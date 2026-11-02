"""GPU curriculum on five focused UE families, with a declared quality task."""
import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import shutil
import subprocess
import threading
import time
import cv2
import numpy as np
import torch
from focused_scenarios import load_suite, case_plan, write_materialized
from hascam.agent import HASCamAgent
from hascam.config import PDQNConfig
from integration.compute import resolve_device, compute_metadata
from integration.detector import sha256
from integration.experiment import make_extractor, rule_action
from integration.focused_environment import FocusedEnvironment, FocusedTransport
from integration.focused_contract import default_policy_path, validate_hascam_frontend
from integration.metrics import iou
from integration.pretrained_detector import create_detector
from integration.runtime import REPRO, ROOT, StandaloneArena, new_run
from integration.video import VideoLog


class GPUMonitor:
    """Whole-device telemetry, including UE rendering; sampled independently."""
    def __init__(self, output, enabled):
        self.output, self.enabled = output, enabled
        self.stopped = threading.Event()

    def __enter__(self):
        if self.enabled:
            self.worker = threading.Thread(target=self.sample, daemon=True)
            self.worker.start()
        return self

    def sample(self):
        with (self.output/"gpu-telemetry.csv").open("w", encoding="utf-8", buffering=1) as stream:
            stream.write("unix_time,gpu_util_percent,memory_used_mib,memory_total_mib,power_w\n")
            while not self.stopped.is_set():
                try:
                    result = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,power.draw",
                                             "--format=csv,noheader,nounits"], capture_output=True, text=True,
                                            timeout=10, creationflags=subprocess.CREATE_NO_WINDOW)
                    if result.returncode == 0:
                        stream.write(f"{time.time():.3f},"+result.stdout.strip()+"\n")
                except (OSError, subprocess.TimeoutExpired):
                    pass
                self.stopped.wait(5)

    def __exit__(self, *exc):
        self.stopped.set()
        if self.enabled:
            self.worker.join(timeout=11)


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def save_agent(agent, path, metadata):
    temporary = Path(str(path) + ".tmp")
    agent.save(temporary, metadata=metadata)
    temporary.replace(path)


def offline_frame(selected, diagnostic):
    """Audit data only; never returned to the policy, reward, or selector."""
    targets = diagnostic.get("targets", [])
    matches = [(iou(selected.bbox, target["gt_bbox_normalized"]), target)
               for target in targets if selected and target.get("gt_in_frame")]
    overlap, target = max(matches, key=lambda item: item[0]) if matches else (0., None)
    matched = target is not None and overlap >= .3
    return {"matched_selected_gt": matched, "selected_gt_iou": overlap,
            "matched_center_error": float(np.linalg.norm(np.asarray(selected.center)*2-1)) if matched else None,
            "matched_area_ratio": selected.area if matched else None,
            "matched_scale_log_error": abs(math.log(max(selected.area, 1e-12)/.05)) if matched else None}


def aggregate(summaries, wall_seconds):
    mean = lambda key: float(np.mean([s[key] for s in summaries if s.get(key) is not None])) if any(s.get(key) is not None for s in summaries) else None
    return {"episodes": len(summaries), "success_rate_gt_verified": mean("success"),
            "quality_completion_estimate_rate": mean("termination_success_estimate"),
            "paper_confidence_condition_rate": mean("paper_confidence_condition_ever"),
            "TC_offline_pooled": sum(s["completed_distinct_count"] for s in summaries)/max(1, sum(s["target_count"] for s in summaries)),
            "mean_center_error_gt_matched": mean("mean_center_error_gt_matched"),
            "mean_scale_log_error_gt_matched": mean("mean_scale_log_error_gt_matched"),
            "selected_gt_match_fraction": mean("selected_gt_match_fraction"),
            "sampled_visible_time_fraction": mean("visible_time_fraction_sampled"),
            "quality_action_fraction": mean("quality_action_fraction"),
            "wall_seconds": wall_seconds, "episodes_detail": summaries}


def run_cases(plan, output, env, agent, args, configuration, training=False, episode_base=0):
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    summaries = []
    update_path = output / "updates.jsonl"
    with (output / "episodes.jsonl").open("w", encoding="utf-8", buffering=1) as episode_log, \
         update_path.open("w", encoding="utf-8", buffering=1) as update_log:
        for local_index, case in enumerate(plan):
            index = episode_base + local_index
            name, seed = case["scenario_name"], case["seed"]
            folder = output / f"episode-{local_index:04d}-{name}-seed{seed}"
            folder.mkdir()
            scenario = write_materialized(name, output / "configs", level=case["level"], split=case["split"])
            if agent:
                agent.set_episode(index if training else agent.completed_episodes)
            phase = "TRAIN" if training else "DEMO" if args.task == "demo" else "EVAL"
            state = env.reset(scenario, seed, f"{args.controller.upper()}_{args.profile.upper()}_{phase}")
            write_json(folder / "initial-observation.json", {"state": state.tolist(),
                "selected_bbox": list(env.selected.bbox) if env.selected else None,
                "detections": [{"bbox": list(d.bbox), "confidence": d.confidence} for d in env.detections]})
            write_json(folder / "scenario.json", env.transport.diagnostics)
            record = not training or local_index == 0 or index % args.video_every == 0
            video = VideoLog(folder, f"{args.controller} | {configuration['detector']['kind']} | {name} | {args.profile}") if record else None
            if video:
                video.write(env.transport.rgb, env.detections, env.selected, "reset; RGB detector state")
            else:
                cv2.imwrite(str(folder / "first-rgb.png"), cv2.cvtColor(env.transport.rgb, cv2.COLOR_RGB2BGR))
            done, info, reward_sum = False, {}, 0.
            audited, durations, policy_times, pipeline_times, update_times, detector_times = [], [], [], [], [], []
            episode_started = time.perf_counter()
            with (folder / "steps.jsonl").open("w", encoding="utf-8", buffering=1) as steps, \
                 (folder / "diagnostics-only.jsonl").open("w", encoding="utf-8", buffering=1) as diagnostics:
                try:
                    for decision in range(env.max_steps):
                        tick = time.perf_counter()
                        action = agent.select_action(state, explore=training) if agent else rule_action(state)
                        policy_seconds = time.perf_counter() - tick
                        next_state, reward, done, info = env.step(*action)
                        environment_seconds = time.perf_counter() - tick - policy_seconds
                        learner_tick = time.perf_counter()
                        if training:
                            agent.observe(state, action, reward, next_state, done)
                            if decision % agent.config.update_interval == 0:
                                update = agent.update()
                                if update:
                                    update_log.write(json.dumps(update | {"episode": index, "environment_steps": agent.environment_steps}) + "\n")
                        learner_seconds = time.perf_counter() - learner_tick
                        reward_sum += reward
                        truth = env.transport.diagnostics
                        diagnostics.write(json.dumps({key: truth.get(key) for key in
                            ("sim_time", "rotation", "fov", "targets", "demo_geometry_follow", "free_preview", "control_mode")}) + "\n")
                        audit = offline_frame(env.selected, truth)
                        audited.append(audit)
                        if video:
                            video.write(env.transport.rgb, env.detections, env.selected,
                                f"step {decision+1} action {action[0]} {action[1]:.3f}s conf {next_state[3]:.3f}")
                        wall = time.perf_counter() - tick
                        info.update(policy_seconds=policy_seconds, environment_seconds=environment_seconds,
                                    learner_update_seconds=learner_seconds, compute_and_record_wall_seconds=wall)
                        if agent:
                            info["policy_selection"] = agent.last_selection
                        presentation_wait = (max(0., action[1] - (time.perf_counter()-tick))
                            if args.task == "demo" and not args.fast and not done else 0.)
                        info["presentation_wait_seconds"] = presentation_wait
                        steps.write(json.dumps(info) + "\n")
                        durations.append(action[1]); policy_times.append(policy_seconds)
                        update_times.append(learner_seconds); pipeline_times.append(wall)
                        detector_times.append(info["detector_seconds"])
                        state = next_state
                        if done:
                            break
                        if presentation_wait:
                            time.sleep(presentation_wait)
                finally:
                    if video:
                        video.close()
            if not video:
                cv2.imwrite(str(folder / "last-rgb.png"), cv2.cvtColor(env.transport.rgb, cv2.COLOR_RGB2BGR))
            matched = [row for row in audited if row["matched_selected_gt"]]
            summary = env.summary(info.get("success", False)) | {
                "episode": index, "scenario": name, "seed": seed, "split": case["split"], "level": case["level"],
                "reward_sum": reward_sum, "terminated": done, "video_recorded": record,
                "network_updates_total": agent.update_count if agent else 0,
                "epsilon": agent.epsilon if agent else None,
                "mean_center_error_gt_matched": float(np.mean([row["matched_center_error"] for row in matched])) if matched else None,
                "mean_scale_log_error_gt_matched": float(np.mean([row["matched_scale_log_error"] for row in matched])) if matched else None,
                "selected_gt_match_fraction": len(matched)/max(1, sum(row["visible_detection"] for row in env.metrics.rows[1:])),
                "mean_policy_seconds": float(np.mean(policy_times)), "mean_detector_seconds": float(np.mean(detector_times)),
                "mean_learner_seconds_per_action": float(np.mean(update_times)),
                "mean_pipeline_seconds_per_action": float(np.mean(pipeline_times)),
                "episode_wall_seconds": time.perf_counter()-episode_started,
                "peak_torch_allocated_bytes": torch.cuda.max_memory_allocated() if args.device == "cuda" else 0}
            write_json(folder / "summary.json", summary)
            summaries.append(summary); episode_log.write(json.dumps(summary) + "\n")
            elapsed = time.perf_counter() - started
            progress = aggregate(summaries, elapsed) | {"current_episode": local_index+1, "planned_episodes": len(plan),
                "eta_seconds_at_observed_episode_rate": elapsed/(local_index+1)*(len(plan)-local_index-1)}
            write_json(output / "progress.json", progress)
            if training:
                agent.set_episode(index+1)
                if (local_index+1) % args.checkpoint_every == 0 or local_index+1 == len(plan):
                    save_agent(agent, output / "checkpoint.pt", configuration)
            print(f"{output.name} {local_index+1}/{len(plan)} {name} steps={summary['action_steps']} "
                  f"GTsuccess={summary['success']} quality={summary['quality_action_fraction']:.1%} "
                  f"matched={summary['selected_gt_match_fraction']:.1%} updates={summary['network_updates_total']} "
                  f"pipeline={summary['mean_pipeline_seconds_per_action']*1000:.1f}ms ETA={progress['eta_seconds_at_observed_episode_rate']/60:.1f}min", flush=True)
    result = aggregate(summaries, time.perf_counter()-started)
    write_json(output / "results.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["train", "evaluate", "demo"], default="demo")
    parser.add_argument("--controller", choices=["hascam", "rule"], default="hascam")
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cuda")
    parser.add_argument("--detector-kind", default="yolo26s_ue")
    parser.add_argument("--detector-checkpoint", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--profile", choices=["quality", "paper"], default="quality")
    parser.add_argument("--train-steps", type=int, default=128)
    parser.add_argument("--eval-steps", type=int, default=1000)
    parser.add_argument("--sim-budget", type=float, default=30.)
    parser.add_argument("--capture-settle", type=float, default=0.)
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--stage-limit", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20261031)
    parser.add_argument("--video-every", type=int, default=20)
    parser.add_argument("--checkpoint-every", type=int, default=16)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--fast", action="store_true", help="skip action-duration playback waits in demo")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    args.device = resolve_device(args.device)
    if min(args.train_steps, args.eval_steps, args.sim_budget, args.video_every, args.checkpoint_every) <= 0:
        parser.error("Budgets and recording/checkpoint intervals must be positive")
    if not 1 <= args.stage_limit <= 3 or (args.episodes is not None and args.episodes < 1):
        parser.error("stage-limit must be 1..3 and episodes must be positive")
    if args.capture_settle < 0:
        parser.error("capture-settle must be nonnegative")
    if args.profile == "paper":
        args.train_steps = args.eval_steps = 128
    if args.task == "train" and (args.controller != "hascam" or args.checkpoint):
        parser.error("Focused curriculum starts a fresh HAS-CAM agent; evaluation accepts checkpoints")
    torch.set_num_threads(4)
    output = args.output or new_run("focused-gpu-" + args.task)
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    if (output / "configuration.json").exists():
        raise FileExistsError("Use a new output folder; preserve previous attempts")
    suite = load_suite()
    detector = create_detector(kind=args.detector_kind, device=args.device, checkpoint=args.detector_checkpoint)
    extractor = make_extractor(args.device)
    checkpoint = args.checkpoint or default_policy_path(REPRO, args.detector_kind, args.profile)
    agent = (HASCamAgent(PDQNConfig(), seed=args.seed, device=args.device) if args.task == "train" else
             HASCamAgent.load(checkpoint, device=args.device, restore_replay=False)) if args.controller == "hascam" else None
    frontend_validation = (validate_hascam_frontend(agent.checkpoint_metadata,
        {"kind": args.detector_kind, **detector.provenance}, extractor.metadata(), args.profile)
        if agent and args.task != "train" else None)
    configuration = {"task": args.task, "controller": args.controller, "task_profile": args.profile,
        "compute": compute_metadata(args.device), "detector": {"kind": args.detector_kind, **detector.provenance},
        "reid": extractor.metadata(), "pdqn": asdict(agent.config) if agent else None,
        "policy_checkpoint": str(checkpoint) if args.task != "train" and agent else None,
        "policy_checkpoint_sha256": sha256(checkpoint) if args.task != "train" and agent else None,
        "suite_sha256": sha256(REPRO / "focused_scenarios/suite.json"), "suite": suite,
        "quality_task_extension": {"active": args.profile == "quality", "confidence_min": .9, "center_error_max": .15, "area_range": [.02,.08],
            "consecutive_actions": 10, "Fmon_commit": "waits for quality predicate", "per_frame_reward": "original author_code VPE",
            "train_decision_budget": args.train_steps, "eval_decision_budget": args.eval_steps,
            "sim_budget_s": args.sim_budget, "boundary_failure_priority": True,
            "budget_note": "end-of-action sampled; may overshoot simulation budget by at most one action"},
        "policy_input": "eight RGB-detector values only; GT is logged offline after action",
        "capture": {"settle_seconds": args.capture_settle, "reset_warmup_captures": 3,
                    "evidence": "verification/capture-settle-gpu; 36 immediate/delayed same-pose comparisons",
                    "warmup_note": "RGB captures only, same simulator t=0 and camera pose; no GT aiming"},
        "platform": "UE5.7.4 replacement; not author Unity CamSim",
        "video": "10fps action-end frames; every 20th training episode and every evaluation episode",
        "fresh_training": args.task == "train", "training_seed": agent.seed if agent else None,
        "requested_seed": args.seed, "frontend_validation": frontend_validation,
        "diagnostic_demo": args.task == "demo",
        "demo_pacing": "action_duration" if args.task == "demo" and not args.fast else "compute_speed",
        "complete_fixed_curriculum": args.task == "train" and args.episodes is None and args.stage_limit == 3,
        "requested_stage_limit": args.stage_limit, "requested_episode_override": args.episodes,
        "resolved_training_plans": [{"stage": stage["name"], "cases": case_plan("train",
            episodes=args.episodes or stage["episodes"], level=stage["level"], names=stage["scenario_names"],
            seed_offset=stage["seed_offset"])} for stage in suite["stages"][:args.stage_limit]] if args.task == "train" else [],
        "resolved_validation_plan": case_plan("validation"), "resolved_test_plan": case_plan("test"),
        "actual_evaluation_plan": case_plan(args.split, episodes=args.episodes) if args.task != "train" else None,
        "source_sha256": {str(p.relative_to(REPRO)): sha256(p) for folder in ("hascam", "integration", "focused_scenarios")
                          for p in sorted((REPRO/folder).glob("*.py"))},
        "ue_module_sha256": sha256(ROOT/"PTZSim/Binaries/Win64/UnrealEditor-PTZSim.dll")}
    write_json(output/"configuration.json", configuration)
    started = time.perf_counter()
    stage_results = []
    with GPUMonitor(output, args.device == "cuda"), StandaloneArena(output, headless=args.headless) as process:
        transport = FocusedTransport(process.port, hud=not args.headless, settle_seconds=args.capture_settle)
        try:
            if args.task == "train":
                agent.save(output/"initial-checkpoint.pt", metadata=configuration, include_replay=False)
                env = FocusedEnvironment(transport, detector, extractor, args.profile, args.eval_steps, args.sim_budget)
                initial = run_cases(case_plan("validation"), output/"validation-initial", env, agent, args, configuration)
                for stage_index, stage in enumerate(suite["stages"][:args.stage_limit]):
                    plan = case_plan("train", episodes=args.episodes or stage["episodes"], level=stage["level"],
                                     names=stage["scenario_names"], seed_offset=stage["seed_offset"])
                    env = FocusedEnvironment(transport, detector, extractor, args.profile, args.train_steps, args.sim_budget)
                    stage_output = output/f"stage-{stage_index+1}-{stage['name']}"
                    trained = run_cases(plan, stage_output, env, agent, args, configuration, True, agent.completed_episodes)
                    env = FocusedEnvironment(transport, detector, extractor, args.profile, args.eval_steps, args.sim_budget)
                    validated = run_cases(case_plan("validation"), output/f"validation-stage-{stage_index+1}", env, agent, args, configuration)
                    stage_results.append({"stage": stage, "training": trained, "validation": validated,
                                          "continuation": "predeclared fixed curriculum; no test-guided model selection"})
                    write_json(output/"stage-results.json", stage_results)
                save_agent(agent, output/"checkpoint.pt", configuration)
                # Publish final curriculum checkpoint, never a test-selected checkpoint.
                result = {"initial_validation": initial, "stages": stage_results}
                if configuration["complete_fixed_curriculum"]:
                    published = default_policy_path(REPRO, args.detector_kind, args.profile)
                    shutil.copy2(output/"checkpoint.pt", published)
                    write_json(published.with_suffix(".provenance.json"), {"source_run": str(output),
                        "sha256": sha256(published), "episodes": agent.completed_episodes,
                        "updates": agent.update_count, "configuration": configuration})
                    env = FocusedEnvironment(transport, detector, extractor, args.profile, args.eval_steps, args.sim_budget)
                    test = run_cases(case_plan("test"), output/"test-final", env, agent, args, configuration)
                    result.update(test=test, checkpoint_sha256=sha256(published))
                else:
                    result["partial_curriculum_note"] = "Explicit subset; no final weight publication or test evaluation"
            else:
                env = FocusedEnvironment(transport, detector, extractor, args.profile, args.eval_steps, args.sim_budget)
                plan = case_plan(args.split, episodes=args.episodes)
                result = run_cases(plan, output/args.split, env, agent, args, configuration)
        finally:
            transport.close()
    result["total_wall_seconds"] = time.perf_counter()-started
    write_json(output/"results.json", result)
    print(f"Finished: {output}", flush=True)
    return output


if __name__ == "__main__":
    main()
