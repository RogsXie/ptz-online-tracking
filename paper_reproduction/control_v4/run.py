"""Bounded static-observation repair; v3 learner/reward/quality gates retained."""
import argparse
from contextlib import contextmanager
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import pickle
import time
import traceback
import cv2
import numpy as np
import torch
from control_curriculum import cases, materialize, stage_spec, load_protocol, assess_episode, gate_aggregate
from control_learning.environment import AtomicTransport
from control_v4.environment import STATE_FEATURES
from control_v4.agent import StaticAgent as ImprovedAgent, control_config
from control_v4.environment import StaticEnvironment as ImprovedEnvironment
from integration.runtime import REPRO, ROOT, StandaloneArena
from integration.pretrained_detector import create_detector
from integration.experiment import make_extractor
from integration.focused_training import GPUMonitor

STAGE = "static_center"
NETWORKS = ("actor_net", "actor_target", "param_net", "param_net_target")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def network_hashes(agent):
    result = {}
    for name in NETWORKS:
        digest = hashlib.sha256()
        for key, value in getattr(agent, name).state_dict().items():
            digest.update(key.encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        result[name] = digest.hexdigest()
    return result


@contextmanager
def frozen(agent, receipt):
    before = network_hashes(agent)
    counters = (agent.environment_steps, agent.update_count, len(agent.memory), agent.exploration_steps)
    rng = pickle.dumps((agent.rng.bit_generator.state, agent.replay_rng.bit_generator.state, agent.python_rng.getstate()))
    old_epsilon = agent.epsilon
    modes, flags = {}, {}
    for name in NETWORKS:
        net = getattr(agent, name)
        modes[name] = net.training
        flags[name] = [p.requires_grad for p in net.parameters()]
        net.eval().requires_grad_(False)
    receipt.update(before=before, four_networks_eval=True, four_networks_requires_grad_false=True)
    try:
        yield
    finally:
        after = network_hashes(agent)
        receipt.update(after=after, networks_unchanged=before == after,
            counters_unchanged=counters == (agent.environment_steps, agent.update_count, len(agent.memory), agent.exploration_steps),
            sampling_rng_unchanged=rng == pickle.dumps((agent.rng.bit_generator.state, agent.replay_rng.bit_generator.state, agent.python_rng.getstate())))
        for name in NETWORKS:
            net = getattr(agent, name)
            net.train(modes[name])
            for parameter, flag in zip(net.parameters(), flags[name]):
                parameter.requires_grad_(flag)
        agent.epsilon = old_epsilon
        if not all(receipt[key] for key in ("networks_unchanged", "counters_unchanged", "sampling_rng_unchanged")):
            raise RuntimeError("Frozen evaluation mutated learning state")


def stop_check(root):
    if (root / "STOP_REQUEST").exists():
        raise InterruptedError("STOP_REQUEST received at an action boundary")


def episode(case, output, transport, detector, extractor, agent, training, root, fixed_action=None):
    output.mkdir(parents=True, exist_ok=False)
    scene = materialize(STAGE, case["split"], case["seed"], output)
    environment = ImprovedEnvironment(transport, detector, extractor, stage_spec(STAGE))
    state = environment.reset(scene, case["seed"])
    write_json(output / "resolved-scene.json", transport.diagnostics)
    cv2.imwrite(str(output / "initial-rgb.jpg"), cv2.cvtColor(transport.rgb, cv2.COLOR_RGB2BGR))
    started, initial_update = time.perf_counter(), agent.update_count
    updates, random_actions, total_reward, wall_steps = [], 0, 0., []
    with (output / "steps.jsonl").open("w", encoding="utf-8", buffering=1) as log, \
         (output / "updates.jsonl").open("w", encoding="utf-8", buffering=1) as update_log:
        log.write(json.dumps(environment.rows[0] | {"initial": True, "state": state.tolist()}) + "\n")
        try:
            for decision in range(stage_spec(STAGE)["max_decisions"]):
                stop_check(root)
                clock = time.perf_counter()
                if fixed_action is not None:
                    action, selection = fixed_action, {"fixed_preflight_only": True, "random": False}
                else:
                    action = agent.select_action(state, explore=training)
                    selection = agent.last_selection.copy()
                next_state, reward, done, row = environment.step(*action)
                if training:
                    agent.observe(state, action, reward, next_state, done)
                # Commit the real transition before attempting any optimization.
                log.write(json.dumps(row | {"decision": decision, "input_state": state.tolist(),
                    "next_state": next_state.tolist(), "policy_selection": selection,
                    "updates_before": agent.update_count}, allow_nan=False) + "\n")
                state = next_state
                if training:
                    for _ in range(2):
                        update = agent.update()
                        if update is not None:
                            update_log.write(json.dumps(update, allow_nan=False) + "\n")
                            updates.append(update)
                wall_steps.append(time.perf_counter() - clock)
                total_reward += reward
                random_actions += int(selection["random"])
                if done:
                    break
        except BaseException as exc:
            write_json(output / "incident.json", {"error": repr(exc), "last_observations": environment.rows[-2:],
                "real_steps": environment.steps, "updates_total": agent.update_count})
            raise
    cv2.imwrite(str(output / "final-rgb.jpg"), cv2.cvtColor(transport.rgb, cv2.COLOR_RGB2BGR))
    assessed = assess_episode(STAGE, environment.rows)
    result = assessed | {"seed": case["seed"], "split": case["split"], "stage": STAGE,
        "reward_sum": total_reward, "steps": environment.steps, "wall_seconds": time.perf_counter()-started,
        "mean_q_loss": float(np.mean([u["q_loss"] for u in updates])) if updates else None,
        "mean_parameter_objective": float(np.mean([u["parameter_objective"] for u in updates])) if updates else None,
        "network_updates": agent.update_count-initial_update, "updates_total": agent.update_count,
        "epsilon": agent.epsilon if training else 0., "random_actions": random_actions,
        "saturated_nonstay": sum(bool(row.get("saturated_nonstay")) for row in environment.rows),
        "stay_actions": sum(row["action_type"] == 0 for row in environment.rows[1:]),
        "wall_step_mean_s": float(np.mean(wall_steps)), "wall_step_p95_s": float(np.quantile(wall_steps, .95)),
        "q_lr": agent.config.actor_lr, "parameter_lr": agent.config.param_lr, "batch_size": agent.config.batch_size}
    write_json(output / "summary.json", result)
    return result


def evaluate(plan, output, transport, detector, extractor, agent, root, official=False):
    output.mkdir(parents=True, exist_ok=False)
    receipt, results = {}, []
    with frozen(agent, receipt):
        for case in plan:
            result = episode(case, output / f"seed{case['seed']}", transport, detector, extractor, agent, False, root)
            results.append(result)
            write_json(output / "progress.json", {"completed": len(results), "total": len(plan),
                "passed": sum(row["passed"] for row in results)})
            print(f"PROBE {len(results)}/{len(plan)} seed={case['seed']} pass={result['passed']} center={result['terminal_center_mean']} reversals={result['direction_reversals']}", flush=True)
    centers = [row["terminal_center_mean"] if row["terminal_center_mean"] is not None else 2. for row in results]
    report = {"purpose": "held-out stage validation" if official else "training-partition diagnostic; not independent validation",
        "passed_cases": sum(row["passed"] for row in results), "total": len(results),
        "center_mean": float(np.mean(centers)), "reward_mean": float(np.mean([row["reward_sum"] for row in results])),
        "reversal_mean": float(np.mean([row["direction_reversals"] for row in results])),
        "collapsed_cases": sum(row["terminal_center_mean"] is None or
            (row["terminal_center_mean"] > .05 and (row["stay_actions"]+row["saturated_nonstay"])/row["steps"] > .8) for row in results),
        "lost_cases": sum(row["longest_loss_run_seconds"] > 1. for row in results),
        "freeze": receipt, "episodes": results}
    if official:
        report["gate"] = gate_aggregate(STAGE, results)
    write_json(output / "report.json", report)
    return report


def review_probe(current, prior, completed):
    reasons = []
    if completed >= 16:
        if current["collapsed_cases"] >= 4:
            reasons.append("greedy policy remains still or mechanically saturated off-center in at least half the probes")
        if current["lost_cases"] >= 4:
            reasons.append("greedy policy loses the target for over1s in at least half the probes")
        if current["reversal_mean"] > 12 and (prior is None or current["passed_cases"] <= prior["passed_cases"]):
            reasons.append("frequent greedy direction reversals without improved strict success")
        if prior and current["reward_mean"] > prior["reward_mean"]+1 and current["center_mean"] > prior["center_mean"]+.01:
            reasons.append("reward improves while RGB centering worsens")
    if completed >= 32 and current["passed_cases"] < current["total"]:
        reasons.append("bounded32-round attempt exhausted; diagnose before any continuation")
    return reasons


def preflight(output, transport, detector, extractor, agent, root):
    output.mkdir(parents=True, exist_ok=False)
    noops = []
    for case in cases(STAGE, "train", 4, offset=8900):
        noops.append(episode(case, output / f"noop-{case['seed']}", transport, detector, extractor, agent, False, root, (0,.25)))
    responses = []
    for action in (1,2,3,4):
        folder = output / f"pulse-{action}"
        scene = materialize(STAGE, "train", 218904, folder)
        env = ImprovedEnvironment(transport, detector, extractor, stage_spec(STAGE))
        env.reset(scene, 218904)
        _, reward, _, row = env.step(action, .1)
        responses.append({"action": action, "initial": env.rows[0], "after": row,
            "center_improvement": env.rows[0]["center_error"]-row["center_error"] if row["visible"] else None, "reward": reward})
    improvements = [row for row in responses if row["center_improvement"] is not None and row["center_improvement"] > .005 and row["reward"] > 0]
    passed = all(not row["passed"] and row["selected_availability"] >= .99 for row in noops) and bool(improvements)
    report = {"passed": passed, "no_op_passes": sum(row["passed"] for row in noops),
        "no_op_cases": noops, "fixed_action_responses": responses, "improving_positive_reward_actions": len(improvements),
        "learning_updates": agent.update_count, "scope": "actuation/reward/no-op sanity only; no learned-control qualification"}
    write_json(output / "report.json", report)
    if not passed:
        raise RuntimeError("Preflight failed: no-op/actuator/reward contract requires repair")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20261102)
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args(argv)
    if args.episodes not in (8,16,24,32):
        parser.error("Static observation repair attempt is bounded to8/16/24/32rounds; inspect evidence before extending")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "configuration.json").exists():
        raise FileExistsError("New attempts require fresh folders")
    torch.set_num_threads(4)
    agent = ImprovedAgent(seed=args.seed)
    detector, extractor = create_detector(device="cuda"), make_extractor("cuda")
    expected = load_protocol()["frozen_frontend"]
    for key in ("threshold", "imgsz", "half"):
        assert detector.provenance[key] == expected[key]
    assert detector.provenance["sha256"] == expected["detector_sha256"]
    assert extractor.metadata()["weights_sha256"] == expected["reid_sha256"]
    before = {str(REPRO / "weights/yolo26s_ue_gpu30.pt"): detector.provenance["sha256"],
        str(extractor.weights): extractor.metadata()["weights_sha256"],
        str(REPRO / "weights/hascam_focused_gpu.pt"): sha256(REPRO / "weights/hascam_focused_gpu.pt")}
    metadata = {"profile": "static_observation_pdqn_v4", "stage": STAGE, "from_scratch": True,
        "pdqn": asdict(agent.config), "updates_per_transition": 2, "warmup_transitions":256,
        "state_features": STATE_FEATURES, "observation_scope":"stationary target, fixed FOV only", "detector":detector.provenance, "reid":extractor.metadata(),
        "protocol_sha256":sha256(REPRO / "control_curriculum/protocol.json"),
        "ue_module_sha256":sha256(ROOT / "PTZSim/Binaries/Win64/UnrealEditor-PTZSim.dll"),
        "train_seeds":[211000,211000+args.episodes-1], "probe_seeds":[219100,219107],
        "preflight_seeds":[218900,218904], "validation_seeds":[510000,510039],
        "probe_interval":8, "validation_trigger":"all8training-partition greedy probes pass",
        "validation_gate":"unchanged38/40 and each quadrant>=9/10; no gate loosening",
        "stop_review":"nonfinite/timing/actuation immediately; collapse/loss/oscillation/reward mismatch from16rounds; mandatory32roundreview",
        "duration_unit":"physical_seconds", "changes_from_author":["continuous image centering objective; no confidence completion bonus",
            "current image geometry with remaining task time; static-only history removal", "physical .04-.25s,8deg/s actuator", "action-isolated Q duration gradients",
            "dense replay updates with projected duration domain", "frozen greedy probes and independent quality validation"],
        "changes_from_v3":["replace action/trend history by remaining task time and reserved zeros; no other learning/objective changes"], "repair_evidence":"verification/improved-control-v3-20261031/oscillation-evidence.json", "comparison":"same training/probe scenes as v3; these are development partitions, not new independent tests",
        "no_GT_policy_or_reward":True, "no_expert_controller":True, "video_recording":False,
        "requested_rounds":args.episodes, "qualified_stages":[], "source_sha256":{}}
    for directory in ("control_v4", "control_v3", "control_learning", "control_curriculum", "hascam", "integration"):
        for path in sorted((REPRO / directory).glob("*.py")):
            relative = path.relative_to(REPRO)
            metadata["source_sha256"][str(relative)] = sha256(path)
            destination = output / "source-snapshot" / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(path.read_bytes())
    write_json(output / "configuration.json", metadata)
    write_json(output / "frozen-before.json", before)
    agent.save(output / "initial.pt", metadata, include_replay=False)
    write_json(output / "status.json", {"state":"starting", "profile":metadata["profile"]})
    history, previous_probe, passed, reasons = [], None, False, []
    try:
        detector.warmup()
        with GPUMonitor(output, True), StandaloneArena(output, headless=args.headless) as arena:
            transport = AtomicTransport(arena.port, hud=not args.headless)
            try:
                write_json(output / "status.json", {"state":"preflight"})
                preflight(output / "preflight", transport, detector, extractor, agent, output)
                write_json(output / "status.json", {"state":"training"})
                with (output / "episodes.jsonl").open("w", encoding="utf-8", buffering=1) as episodes, \
                     (output / "round-metrics.log").open("w", encoding="utf-8", buffering=1) as metrics:
                    for index,case in enumerate(cases(STAGE,"train",args.episodes,offset=1000)):
                        result = episode(case, output / "training" / f"round-{index+1:04d}-seed{case['seed']}",
                            transport, detector, extractor, agent, True, output)
                        agent.completed_episodes += 1
                        result["round"] = agent.completed_episodes
                        history.append(result)
                        episodes.write(json.dumps(result, allow_nan=False) + "\n")
                        def display(value):
                            return "not_updated" if value is None else f"{value:.6f}"
                        line = (f"ROUND {result['round']} Q={display(result['mean_q_loss'])} parameter_max={display(result['mean_parameter_objective'])} "
                            f"reward={result['reward_sum']:.5f} epsilon={result['epsilon']:.5f} steps={result['steps']} "
                            f"updates={result['network_updates']}/{agent.update_count} center={result['terminal_center_mean']} "
                            f"Qlr=0.0005 Paramlr=0.0001 batch=128")
                        metrics.write(line+"\n"); print(line, flush=True)
                        agent.save(output / "last.pt", metadata)
                        write_json(output / "progress.json", {"completed_rounds":len(history), "planned_rounds":args.episodes,
                            "steps":agent.environment_steps, "updates":agent.update_count, "last_round":result,
                            "passed":False, "later_stages_started":False})
                        if (index+1) % 8 == 0:
                            agent.save(output / f"checkpoint-{index+1:04d}.pt", metadata)
                            write_json(output / "status.json", {"state":"training_partition_probe", "round":index+1})
                            probe = evaluate(cases(STAGE,"train",8,offset=9100), output / f"probe-{index+1:04d}",
                                transport, detector, extractor, agent, output)
                            if probe["passed_cases"] == 8:
                                write_json(output / "status.json", {"state":"heldout_validation", "round":index+1})
                                validation = evaluate(cases(STAGE,"validation"), output / f"validation-{index+1:04d}",
                                    transport, detector, extractor, agent, output, official=True)
                                passed = validation["gate"]["passed"]
                                if passed:
                                    metadata["qualified_stages"] = [STAGE]
                                    agent.save(output / "qualified.pt", metadata)
                                else:
                                    reasons = ["independent40-case validation failed; inspect before retry"]
                                break
                            reasons = review_probe(probe, previous_probe, index+1)
                            previous_probe = probe
                            write_json(output / f"review-{index+1:04d}.json", {"stop_for_review":bool(reasons), "reasons":reasons})
                            if reasons:
                                break
                            write_json(output / "status.json", {"state":"training"})
                agent.save(output / "last.pt", metadata)
            finally:
                transport.close()
        write_json(output / "status.json", {"state":"qualified" if passed else "review_required",
            "completed_rounds":len(history), "steps":agent.environment_steps, "updates":agent.update_count,
            "passed":passed, "reasons":reasons or ([] if passed else ["bounded attempt ended"]), "later_stages_started":False})
    except BaseException as exc:
        agent.save(output / "interrupted.pt", metadata)
        write_json(output / "failure.json", {"error":repr(exc), "traceback":traceback.format_exc(),
            "completed_rounds":len(history), "steps":agent.environment_steps, "updates":agent.update_count})
        write_json(output / "status.json", {"state":"stopped_by_user" if isinstance(exc, InterruptedError) else "anomaly",
            "error":repr(exc), "passed":False, "completed_rounds":len(history), "later_stages_started":False})
        raise
    finally:
        after = {path:sha256(path) for path in before}
        write_json(output / "frozen-after.json", after)
        if before != after:
            raise RuntimeError("Frozen frontend or historical baseline changed")
    return passed


if __name__ == "__main__":
    main()
