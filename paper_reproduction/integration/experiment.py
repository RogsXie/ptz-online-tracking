"""Reproducible training/evaluation; every controller is labelled explicitly."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
from hascam.agent import HASCamAgent
from hascam.config import PDQNConfig
from hascam.cfff import CFFF
from integration.runtime import ROOT, REPRO, StandaloneArena, new_run
from integration.transport import RGBTransport
from integration.detector import SSDDetector, sha256
from integration.environment import PaperEnvironment
from integration.video import VideoLog
from integration.compute import resolve_device, compute_metadata


def rule_action(state):
    """UE-adapted image-state rule, not PDF Table 4; fixed .2 s, no truth fallback."""
    x, y, area, confidence = state[:4]
    if area < 0:
        return 6, .2  # Recover a wider view after detector loss.
    if x > .07:
        return 3, .2
    if x < -.07:
        return 4, .2
    if y > .07:
        return 1, .2
    if y < -.07:
        return 2, .2
    if area < .05:
        return 5, .2
    if area > .07:
        return 6, .2
    return 0, .2


def make_extractor(device="cpu"):
    from hascam.reid import FastReIDExtractor
    return FastReIDExtractor(device=device)


def run(args):
    args.device = resolve_device(getattr(args, "device", "cpu"))
    requested_seed = args.seed
    args.seed = 0 if args.seed is None else args.seed
    requested_ramp = args.curriculum_ramp
    args.curriculum_ramp = 300 if args.curriculum_ramp is None else args.curriculum_ramp
    if args.controller == "ppo_fixed" and args.task == "train":
        from integration.ppo_baseline import train
        return train(args)
    output = args.output or new_run(f"{args.task}-{args.controller}")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    splits = json.loads((REPRO / "scenarios/splits.json").read_text(encoding="utf-8-sig"))
    partition = "train" if args.task == "train" else args.split
    scenarios = args.scenarios or splits[f"{partition}_scenarios"]
    seeds = splits["seed_partitions"][partition]
    for name in scenarios:
        if name not in splits[f"{partition}_scenarios"] and not args.allow_diagnostic_scenario:
            raise ValueError(f"{name} is not assigned to split {partition}")
    if args.task != "train" and args.episodes > len(seeds):
        raise ValueError("Requested episodes exceed the declared distinct seeds")
    detector = SSDDetector(args.detector, device=args.device)
    extractor = make_extractor(args.device)
    cfff = CFFF(extractor)
    cfg = PDQNConfig(reward_mode=args.reward_mode)
    resolved_checkpoint = None
    ppo_policy = None
    if args.controller == "hascam":
        if args.task == "train" and not args.checkpoint:
            agent = HASCamAgent(cfg, seed=args.seed, device=args.device)
        else:
            path = args.checkpoint or REPRO / "weights/hascam_ue_short.pt"
            if not path.exists():
                raise FileNotFoundError(f"No trained checkpoint {path}. Run --task train first.")
            agent = HASCamAgent.load(path, device=args.device)
            resolved_checkpoint = path
            cfg = agent.config
            if args.task == "train":
                if requested_seed is None:
                    args.seed = agent.seed
                elif args.seed != agent.seed:
                    raise ValueError("Resume training seed differs from checkpoint; omit --seed to reuse its seed")
                if requested_ramp is None:
                    args.curriculum_ramp = agent.checkpoint_metadata.get("curriculum", {}).get("ramp_episodes", 300)
                if args.scenarios is None and agent.checkpoint_metadata.get("scenarios"):
                    scenarios = list(agent.checkpoint_metadata["scenarios"])
            if agent.config.reward_mode != args.reward_mode:
                raise ValueError("Checkpoint reward mode differs; pass its recorded --reward-mode")
    else:
        if args.task == "train":
            raise ValueError("Rule controller has no trainable policy")
        agent = None
        if args.controller == "ppo_fixed":
            from stable_baselines3 import PPO
            resolved_checkpoint = args.checkpoint or REPRO / "weights/ppo_fixed_ssd_short.zip"
            ppo_policy = PPO.load(resolved_checkpoint, device=args.device)
            if ppo_policy.observation_space.shape != (8,):
                raise ValueError("This comparison requires an 8-state SSD PPO; historical 10-state HSV weights use the old launcher")
    for name in scenarios:
        if name not in splits[f"{partition}_scenarios"] and not args.allow_diagnostic_scenario:
            raise ValueError(f"Resolved checkpoint scenarios include {name}, outside {partition}; split override must be explicit")
    configuration = {"task": args.task, "controller": args.controller, "partition": partition,
                     "compute": compute_metadata(args.device),
                     "diagnostic_split_override": args.allow_diagnostic_scenario,
                     "scenarios": scenarios, "episodes": args.episodes, "seed": args.seed,
                     "policy_checkpoint": str(resolved_checkpoint) if resolved_checkpoint else None,
                     "policy_checkpoint_sha256": sha256(resolved_checkpoint) if resolved_checkpoint else None,
                     "detector": detector.provenance, "cfff": cfff.metadata(),
                     "pdqn": asdict(cfg) if agent else None,
                     "task_protocol": {"max_steps": 128, "confidence_threshold": .9,
                                       "consecutive_action_observations": 10, "reward_mode": args.reward_mode},
                     "platform": "UE 5.7.4 replacement; not author Unity CamSim",
                     "policy_input": "8 detector-derived values only, no camera calibration/GT",
                     "termination_task_metadata": "configured target count only; not given to policy",
                     "stage": "short_training_or_verification; no convergence claim",
                     "training_seed_cycle_length": len(seeds) if args.task == "train" else None,
                     "curriculum": {"mode": "static_to_dynamic" if args.task == "train" else "none",
                                    "ramp_episodes": args.curriculum_ramp,
                                    "schedule_assumption": "paper omits ratio schedule; p_dynamic=min(1,completed_episodes/ramp), sample uniformly within group"},
                     "video": "10 fps action-end frames, not real-time action-duration video"}
    if ppo_policy:
        configuration["controller_parameters"] = {
            "algorithm": "SB3 PPO", "duration_s": .2, "gamma": ppo_policy.gamma,
            "gae_lambda": ppo_policy.gae_lambda, "rollout_steps": ppo_policy.n_steps,
            "batch_size": ppo_policy.batch_size, "epochs": ppo_policy.n_epochs,
            "learning_rate": float(ppo_policy.lr_schedule(1.0)),
            "clip_range": float(ppo_policy.clip_range(1.0))}
    elif agent is None:
        configuration["controller_parameters"] = {"algorithm": "UE-adapted SSD rule, not paper Table4",
                                                   "duration_s": .2, "offset_threshold": .07,
                                                   "zoom_in_area_below": .05, "zoom_out_area_above": .07}
    configuration["source_sha256"] = {
        str(path.relative_to(REPRO)): sha256(path)
        for folder, pattern in ((REPRO / "hascam", "*.py"), (REPRO / "integration", "*.py"),
                                (REPRO / "scenarios", "*.json"))
        for path in sorted(folder.glob(pattern))}
    configuration["ue_module_sha256"] = sha256(ROOT / "PTZSim/Binaries/Win64/UnrealEditor-PTZSim.dll")
    configuration["unrealcv_module_sha256"] = sha256(ROOT / "PTZSim/Plugins/UnrealCV/Binaries/Win64/UnrealEditor-UnrealCV.dll")
    (output / "configuration.json").write_text(json.dumps(configuration, indent=2), encoding="utf-8")
    summaries = []
    base_episode = agent.completed_episodes if agent else 0
    started = time.perf_counter()
    with StandaloneArena(output, headless=args.headless) as process:
        transport = RGBTransport(process.port)
        env = PaperEnvironment(transport, detector, cfff, args.reward_mode)
        try:
            with (output / "episodes.jsonl").open("w", encoding="utf-8") as episode_log, \
                 (output / "updates.jsonl").open("w", encoding="utf-8") as update_log:
                for episode in range(args.episodes):
                    if args.task == "train":
                        curriculum_rng = np.random.default_rng(np.random.SeedSequence([args.seed, base_episode + episode]))
                        static = [s for s in scenarios if all(t["path"]["type"] == "static" for t in
                                  json.loads((REPRO / "scenarios" / f"{s}.json").read_text(encoding="utf-8-sig"))["targets"])]
                        dynamic = [s for s in scenarios if s not in static]
                        p_dynamic = min(1., (base_episode + episode) / args.curriculum_ramp)
                        pool = dynamic if curriculum_rng.random() < p_dynamic and dynamic else static
                        if not pool:
                            pool = scenarios
                        name = str(curriculum_rng.choice(pool))
                    else:
                        name = scenarios[episode % len(scenarios)]
                    scene_seed = seeds[(base_episode + episode + args.seed if args.task == "train" else episode + args.seed) % len(seeds)]
                    episode_dir = output / f"episode-{episode:04d}-{name}-seed{scene_seed}"
                    episode_dir.mkdir()
                    if agent is not None:
                        agent.set_episode(base_episode + episode if args.task == "train" else base_episode)
                    state = env.reset(REPRO / "scenarios" / f"{name}.json", scene_seed,
                                      "HAS_CAM_SHORT_TRAIN" if args.task == "train" else
                                      "HAS_CAM_CHECKPOINT" if agent else
                                      "PPO_FIXED_SSD_CHECKPOINT" if ppo_policy else "SSD_RULE_BASELINE")
                    (episode_dir / "initial-observation.json").write_text(json.dumps({
                        "state": state.tolist(), "selected_bbox": list(env.selected.bbox) if env.selected else None,
                        "selected_confidence": env.selected.confidence if env.selected else 0,
                        "detections": [{"bbox": list(d.bbox), "confidence": d.confidence} for d in env.detections],
                        "source": "rendered RGB detector only; simulator diagnostics stored separately"}, indent=2), encoding="utf-8")
                    configuration["cfff"] = cfff.metadata()
                    (output / "configuration.json").write_text(json.dumps(configuration, indent=2), encoding="utf-8")
                    (episode_dir / "scenario.json").write_text(json.dumps(transport.diagnostics,
                                                                         indent=2), encoding="utf-8")
                    video = VideoLog(episode_dir, f"{args.controller.upper()} | SSD + CFFF | {name} | seed {scene_seed}")
                    video.write(transport.rgb, env.detections, env.selected, "reset; no simulator coordinates in policy")
                    reward_sum, done, info = 0., False, {}
                    with (episode_dir / "steps.jsonl").open("w", encoding="utf-8") as step_log, \
                         (episode_dir / "diagnostics-only.jsonl").open("w", encoding="utf-8") as diagnostic_log:
                        try:
                            for index in range(min(args.steps, 128)):
                                tick = time.perf_counter()
                                if agent:
                                    action = agent.select_action(state, explore=args.task == "train")
                                elif ppo_policy:
                                    discrete, _ = ppo_policy.predict(state, deterministic=True)
                                    action = int(discrete), .2
                                else:
                                    action = rule_action(state)
                                policy_seconds = time.perf_counter() - tick
                                next_state, reward, done, info = env.step(*action)
                                info["policy_seconds"] = policy_seconds
                                info["step_wall_seconds"] = time.perf_counter() - tick
                                if agent:
                                    info["policy_selection"] = agent.last_selection
                                update_started = time.perf_counter()
                                if agent is not None and args.task == "train":
                                    agent.observe(state, action, reward, next_state, done)
                                    if index % agent.config.update_interval == 0:
                                        update = agent.update()
                                        if update:
                                            update_log.write(json.dumps(update | {"episode": episode,
                                                                                 "environment_steps": agent.environment_steps}) + "\n")
                                            update_log.flush()
                                info["learner_update_seconds"] = time.perf_counter() - update_started
                                reward_sum += reward
                                diagnostic_log.write(json.dumps({key: transport.diagnostics.get(key) for key in
                                    ("sim_time", "rotation", "fov", "targets", "demo_geometry_follow", "free_preview", "control_mode")}) + "\n")
                                diagnostic_log.flush()
                                video.write(transport.rgb, env.detections, env.selected,
                                            f"step {index+1} action {action[0]} duration {action[1]:.3f}s | conf {next_state[3]:.3f}")
                                info["compute_and_record_wall_seconds"] = time.perf_counter() - tick
                                step_log.write(json.dumps(info) + "\n")
                                step_log.flush()
                                state = next_state
                                if done:
                                    break
                                if not args.fast:
                                    time.sleep(max(0., action[1] - (time.perf_counter() - tick)))
                        finally:
                            video.close()
                    summary = env.metrics.summary(info.get("success", False)) | {
                        "episode": episode, "scenario": name, "seed": scene_seed,
                        "reward_sum": reward_sum, "terminated": done,
                        "interrupted_by_requested_step_limit": not done,
                        "completed_feature_clusters": cfff.completed_count,
                        "network_updates_total": agent.update_count if agent else 0}
                    summaries.append(summary)
                    (episode_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
                    episode_log.write(json.dumps(summary) + "\n")
                    episode_log.flush()
                    if agent is not None and args.task == "train":
                        agent.set_episode(base_episode + episode + 1)
                        agent.save(output / "checkpoint.pt", metadata=configuration)
                    print(f"episode {episode+1}/{args.episodes} {name} seed={scene_seed} "
                          f"steps={summary['action_steps']} visible={summary['TSR_detection_proxy']:.1%} "
                          f"success={summary['success']} updates={summary['network_updates_total']}", flush=True)
        finally:
            transport.close()
    successes = [s for s in summaries if s["success"]]
    result = {"episodes": len(summaries), "SR_offline_gt_verified": len(successes) / max(1, len(summaries)),
              "SR_cfff_estimated": sum(s["termination_success_estimate"] for s in summaries) / max(1, len(summaries)),
              "SS_successful_only": float(np.mean([s["action_steps"] for s in successes])) if successes else None,
              "TC_offline_mean": float(np.mean([s["TC_offline_gt_matched"] for s in summaries])),
              "TC_offline_pooled": sum(s["completed_distinct_count"] for s in summaries) / sum(s["target_count"] for s in summaries),
              "wall_seconds": time.perf_counter() - started, "episodes_detail": summaries,
              "interpretation": "Short UE run, not a reproduced paper benchmark or evidence of convergence."}
    (output / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Results: {output}", flush=True)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["train", "evaluate", "demo"], default="demo")
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    parser.add_argument("--controller", choices=["hascam", "rule", "ppo_fixed"], default="hascam")
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--curriculum-ramp", type=int)
    parser.add_argument("--split", choices=["validation", "test"], default="test")
    parser.add_argument("--scenarios", nargs="+")
    parser.add_argument("--allow-diagnostic-scenario", action="store_true")
    parser.add_argument("--reward-mode", choices=["author_code", "paper_literal"], default="author_code")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--detector", type=Path, default=REPRO / "weights/ssd_ue.pt")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--fast", action="store_true")
    args = parser.parse_args(argv)
    if not 1 <= args.steps <= 128 or args.episodes < 1 or (args.curriculum_ramp is not None and args.curriculum_ramp < 1):
        parser.error("steps must be 1..128; episodes and curriculum ramp positive")
    return run(args)


if __name__ == "__main__":
    main()
