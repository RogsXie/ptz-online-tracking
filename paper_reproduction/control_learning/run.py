"""Train one declared stage and advance only after measured validation gates."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
import cv2
import numpy as np
import torch
from control_curriculum import cases, materialize, stage_spec, load_protocol, assess_episode, gate_aggregate
from control_learning.agent import ControlAgent, control_config
from control_learning.environment import AtomicTransport, ControlEnvironment
from hascam.agent import ReplayBuffer
from integration.runtime import REPRO, ROOT, StandaloneArena, new_run
from integration.pretrained_detector import create_detector
from integration.experiment import make_extractor
from integration.compute import compute_metadata
from integration.focused_training import GPUMonitor

STATE_FEATURES=["4*normalized_x","4*normalized_y","log(area/.04)/4","confidence_or0",
    "last_action/6","last_duration/.25","10*delta_normalized_x","10*delta_normalized_y"]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def episode(case, output, transport, detector, extractor, agent, training, updates_per_step=2):
    spec = stage_spec(case["stage"])
    output.mkdir(parents=True, exist_ok=False)
    config = materialize(case["stage"], case["split"], case["seed"], output)
    environment = ControlEnvironment(transport, detector, extractor, spec)
    state = environment.reset(config, case["seed"])
    write_json(output / "resolved-scene.json", transport.diagnostics)
    cv2.imwrite(str(output / "initial-rgb.jpg"), cv2.cvtColor(transport.rgb, cv2.COLOR_RGB2BGR))
    started = time.perf_counter()
    total_reward, loss_values = 0., []
    initial_update = agent.update_count
    with (output / "steps.jsonl").open("w", encoding="utf-8", buffering=1) as log, \
         (output / "updates.jsonl").open("w", encoding="utf-8", buffering=1) as updates:
        log.write(json.dumps(environment.rows[0] | {"initial": True, "state": state.tolist()}) + "\n")
        for decision in range(spec["max_decisions"]):
            action = agent.select_action(state, explore=training)
            selection = agent.last_selection.copy()
            next_state, reward, done, row = environment.step(*action)
            if training:
                agent.observe(state, action, reward, next_state, done)
                for _ in range(updates_per_step):
                    update = agent.update()
                    if update:
                        updates.write(json.dumps(update) + "\n")
                        loss_values.append(update["q_loss"])
            log.write(json.dumps(row | {"decision": decision, "input_state": state.tolist(),
                "next_state": next_state.tolist(), "policy_selection": selection,
                "updates_total": agent.update_count}) + "\n")
            total_reward += reward
            state = next_state
            if done:
                break
    cv2.imwrite(str(output / "final-rgb.jpg"), cv2.cvtColor(transport.rgb, cv2.COLOR_RGB2BGR))
    assessed = assess_episode(case["stage"], environment.rows)
    result = assessed | {"seed": case["seed"], "split": case["split"], "stage": case["stage"],
        "reward_sum": total_reward, "steps": environment.steps, "wall_seconds": time.perf_counter()-started,
        "mean_q_loss": float(np.mean(loss_values)) if loss_values else None,
        "network_updates": agent.update_count-initial_update, "updates_total": agent.update_count,
        "epsilon": agent.epsilon if training else 0., "random_actions": sum(
            1 for line in (output / "steps.jsonl").read_text().splitlines() if json.loads(line).get("policy_selection",{}).get("random"))}
    write_json(output / "summary.json", result)
    return result


def validate(stage, output, transport, detector, extractor, agent):
    output.mkdir(parents=True, exist_ok=False)
    old_allowed = agent.allowed_actions
    agent.set_allowed_actions(stage_spec(stage)["allowed_actions"])
    updates_before, steps_before = agent.update_count, agent.environment_steps
    results = []
    try:
        for case in cases(stage, "validation"):
            folder = output / f"seed{case['seed']}"
            result = episode(case, folder, transport, detector, extractor, agent, False)
            results.append(result)
            write_json(output / "progress.json", {"completed":len(results),"total":40,
                "passed":sum(item["passed"] for item in results)})
            print(f"VALIDATE {stage} {len(results)}/40 pass={sum(item['passed'] for item in results)} "
                  f"center={result.get('terminal_center_mean')} steps={result['steps']}", flush=True)
    finally:
        agent.set_allowed_actions(old_allowed)
    aggregate = gate_aggregate(stage, results)
    aggregate.update(frozen_policy_updates=agent.update_count,
        no_learning_updates=updates_before==agent.update_count,
        no_training_steps=steps_before==agent.environment_steps, episodes=results)
    write_json(output / "gate.json", aggregate)
    return aggregate


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", default="static_center")
    parser.add_argument("--task", choices=("train","evaluate","diagnostic"), default="train")
    parser.add_argument("--episodes", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20261101)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--updates-per-step", type=int, default=2)
    parser.add_argument("--validation-interval", type=int, default=32)
    parser.add_argument("--train-seed-offset", type=int, default=0)
    args = parser.parse_args(argv)
    spec, protocol = stage_spec(args.stage), load_protocol()
    if args.episodes < 1 or args.updates_per_step < 1 or args.validation_interval != 32:
        parser.error("Positive budgets required; preregistered validation interval is32")
    if args.episodes>protocol["progression"]["maximum_episodes_per_attempt"]:
        parser.error("Attempt exceeds preregistered stop-and-review budget")
    torch.set_num_threads(4)
    agent = ControlAgent.load(args.checkpoint) if args.checkpoint else ControlAgent(seed=args.seed)
    inherited = getattr(agent, "checkpoint_metadata", {})
    qualified = list(inherited.get("qualified_stages", []))
    if spec["prerequisite"] and spec["prerequisite"] not in qualified:
        raise ValueError("Prior stage validation has not qualified; difficulty advance blocked")
    qualification_history=dict(inherited.get("qualification_history",{}))
    if spec["prerequisite"]:
        evidence_path=qualification_history.get(spec["prerequisite"])
        if not evidence_path or not Path(evidence_path).is_file():
            raise ValueError("Prior qualification evidence is missing")
        evidence=json.loads(Path(evidence_path).read_text(encoding="utf-8"))
        if not evidence.get("passed") or evidence.get("validation",{}).get("stage")!=spec["prerequisite"]:
            raise ValueError("Prior stage evidence has not passed")
    agent.set_allowed_actions(spec["allowed_actions"])
    # Explicit new-stage exploration reset. Same-stage resume preserves it.
    stage_transition=bool(args.checkpoint and inherited.get("stage") != spec["id"])
    if inherited.get("stage") != spec["id"]:
        agent.exploration_steps = 0
    if stage_transition:
        # Stage2 introduces a different scale objective. Old reward/done labels
        # must not be silently mixed into the new MDP's replay buffer.
        agent.memory=ReplayBuffer(agent.config.replay_capacity,agent.replay_rng)
        agent.actor_optimiser=torch.optim.Adam(agent.actor_net.parameters(),lr=agent.config.actor_lr)
        agent.param_net_optimiser=torch.optim.Adam(agent.param_net.parameters(),lr=agent.config.param_lr)
        agent.actor_target.load_state_dict(agent.actor_net.state_dict())
        agent.param_net_target.load_state_dict(agent.param_net.state_dict())
    output = args.output or new_run("gated-control-"+spec["id"])
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "configuration.json").exists():
        raise FileExistsError("Use a new attempt folder; failed experiments are preserved")
    detector = create_detector(device="cuda")
    extractor = make_extractor("cuda")
    expected = protocol["frozen_frontend"]
    assert detector.provenance["sha256"] == expected["detector_sha256"]
    assert detector.provenance["threshold"] == expected["threshold"] and detector.provenance["imgsz"] == expected["imgsz"]
    assert detector.provenance["half"] == expected["half"]
    assert extractor.metadata()["weights_sha256"] == expected["reid_sha256"]
    if args.checkpoint:
        if inherited.get("profile")!="learned_control_v2" or inherited.get("protocol_sha256")!=sha256(REPRO/"control_curriculum/protocol.json"):
            raise ValueError("Checkpoint profile/protocol differs; no silent reinterpretation")
        if inherited.get("state_features")!=STATE_FEATURES or asdict(agent.config)!=asdict(control_config()):
            raise ValueError("Checkpoint state or learning/action unit contract differs")
        original=inherited.get("detector",{})
        for key in ("sha256","threshold","imgsz","half","target_class_ids","target_labels"):
            if original.get(key)!=detector.provenance.get(key):
                raise ValueError("Frozen detector settings differ from parent checkpoint")
        if inherited.get("reid",{}).get("weights_sha256")!=extractor.metadata()["weights_sha256"]:
            raise ValueError("Frozen ReID differs from parent checkpoint")
    detector.warmup()
    metadata = {"profile":"learned_control_v2", "stage":spec["id"], "qualified_stages":qualified,
        "task":args.task, "pdqn":asdict(agent.config), "compute":compute_metadata("cuda"),
        "qualification_history":qualification_history,
        "stage_transition_reset": "replay,optimizers,targets,exploration; onlineweights retained" if stage_transition else None,
        "detector":detector.provenance, "reid":extractor.metadata(),
        "protocol_sha256":sha256(REPRO/"control_curriculum/protocol.json"), "protocol":protocol,
        "source_sha256":{str(path.relative_to(REPRO)):sha256(path)
            for directory in ("control_learning","control_curriculum","hascam") for path in sorted((REPRO/directory).glob("*.py"))},
        "ue_module_sha256":sha256(ROOT/"PTZSim/Binaries/Win64/UnrealEditor-PTZSim.dll"),
        "parent_checkpoint":str(args.checkpoint) if args.checkpoint else None,
        "parent_checkpoint_sha256":sha256(args.checkpoint) if args.checkpoint else None,
        "training_gate":{"last_32_success_rate_min":.90,"last_32_terminal_center_mean_max":.03},
        "changes_from_baseline":["physical duration .04-.25s,8deg/s actuator", "projected Q duration-gradient query",
            "2 replay updates per true RGB transition,batch128", "stage-specific dense image-only reward",
            "normalized8state; loss retains last-seen image history", "continuoussingle-target CFFF without Fmon completion",
            "inactiveactiondurationfeatures fixed atmidpoint; inactiveparametergradients zero"],
        "state_features":STATE_FEATURES,
        "no_expert_controller":True,"no_GT_policy_or_reward":True,"video_recording":False,
        "requested_episodes":args.episodes,"training_seed_offset":args.train_seed_offset,
        "updates_per_step":args.updates_per_step}
    write_json(output/"configuration.json",metadata)
    snapshots=output/"source-snapshot"
    for relative in metadata["source_sha256"]:
        path=snapshots/relative;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes((REPRO/relative).read_bytes())
    before={str(REPRO/"weights/yolo26s_ue_gpu30.pt"):sha256(REPRO/"weights/yolo26s_ue_gpu30.pt"),
        str(REPRO/"weights/hascam_focused_gpu.pt"):sha256(REPRO/"weights/hascam_focused_gpu.pt"),
        str(extractor.weights):extractor.metadata()["weights_sha256"]} if hasattr(extractor,"weights") else {
        str(REPRO/"weights/yolo26s_ue_gpu30.pt"):sha256(REPRO/"weights/yolo26s_ue_gpu30.pt"),
        str(REPRO/"weights/hascam_focused_gpu.pt"):sha256(REPRO/"weights/hascam_focused_gpu.pt")}
    write_json(output/"frozen-before.json",before)
    agent.save(output/"initial.pt",metadata,include_replay=False)
    history=[]
    passed=False
    started=time.perf_counter()
    try:
        with GPUMonitor(output,True), StandaloneArena(output,headless=args.headless) as arena:
            transport=AtomicTransport(arena.port,hud=not args.headless)
            try:
                if args.task=="evaluate":
                    gate=validate(spec["id"],output/"validation",transport,detector,extractor,agent)
                    passed=gate["passed"]
                else:
                    plan=cases(spec["id"],"train",args.episodes,offset=args.train_seed_offset)
                    with (output/"episodes.jsonl").open("w",encoding="utf-8",buffering=1) as log:
                        for index,case in enumerate(plan):
                            folder=output/"training"/f"episode-{index:04d}-seed{case['seed']}"
                            result=episode(case,folder,transport,detector,extractor,agent,args.task=="train",args.updates_per_step)
                            history.append(result);log.write(json.dumps(result)+"\n")
                            agent.completed_episodes+=int(args.task=="train")
                            recent=history[-32:]
                            rate=sum(row["passed"] for row in recent)/len(recent)
                            center=float(np.mean([row["terminal_center_mean"] if row["terminal_center_mean"] is not None else 2. for row in recent]))
                            display_center=result["terminal_center_mean"] if result["terminal_center_mean"] is not None else 2.
                            print(f"TRAIN {spec['id']} {index+1}/{args.episodes} steps={result['steps']} "
                                f"center={display_center:.4f} pass={result['passed']} "
                                f"recent={rate:.1%} eps={agent.epsilon:.3f} updates={agent.update_count}",flush=True)
                            write_json(output/"progress.json",{"episodes":len(history),"planned":args.episodes,
                                "recent_success":rate,"recent_center":center,"updates":agent.update_count,
                                "steps":agent.environment_steps,"wall_seconds":time.perf_counter()-started,
                                "passed":False,"later_stages_started":False})
                            if (index+1)%32==0 or index+1==len(plan):
                                checkpoint=output/f"checkpoint-{index+1:04d}.pt"
                                agent.save(checkpoint,metadata)
                                # No difficulty changes while the training gate is failed.
                                if len(recent)==32 and rate>=.90 and center<=.03 and args.task=="train":
                                    gate=validate(spec["id"],output/f"validation-{index+1:04d}",transport,detector,extractor,agent)
                                    regression=[]
                                    if gate["passed"]:
                                        for prior in qualified:
                                            regression.append(validate(prior,output/f"regression-{index+1:04d}-{prior}",
                                                transport,detector,extractor,agent))
                                    passed=gate["passed"] and all(row["passed"] for row in regression)
                                    write_json(output/f"gate-decision-{index+1:04d}.json",{"passed":passed,
                                        "training_gate_passed":True,"validation":gate,"regressions":regression})
                                    if passed:
                                        metadata["qualified_stages"]=list(dict.fromkeys(qualified+[spec["id"]]))
                                        metadata["qualification_evidence"]=str(output/f"gate-decision-{index+1:04d}.json")
                                        metadata["qualification_history"][spec["id"]]=metadata["qualification_evidence"]
                                        agent.save(output/"qualified.pt",metadata)
                                        break
                                    if gate["passed"] and not all(row["passed"] for row in regression):
                                        # A previous mastered task regressed. Preserve this failed
                                        # attempt; caller must restore last qualified parent.
                                        write_json(output/"regression-stop.json",{"restore_checkpoint":str(args.checkpoint),
                                            "failed_regressions":[row["stage"] for row in regression if not row["passed"]]})
                                        break
                                else:
                                    write_json(output/f"gate-decision-{index+1:04d}.json",{"passed":False,
                                        "training_gate_passed":False,"recent_success":rate,"recent_center":center,
                                        "action":"remain at current stage; no promotion and no final-test access"})
                agent.save(output/"last.pt",metadata)
            finally:
                transport.close()
    finally:
        after={path:sha256(path) for path in before}
        write_json(output/"frozen-after.json",after)
        if before!=after:
            raise RuntimeError("Frozen detector/ReID/failedbaseline changed")
    write_json(output/"results.json",{"stage":spec["id"],"passed":passed,"episodes":len(history),
        "steps_total":agent.environment_steps,"updates_total":agent.update_count,
        "qualified_stages":metadata["qualified_stages"],"later_stages_started":False,
        "wall_seconds":time.perf_counter()-started,
        "next_action":"qualified for next stage" if passed else "diagnose current failed stage; do not advance"})
    print(f"STAGE {spec['id']} passed={passed}; outputs={output}",flush=True)
    return passed


if __name__=="__main__":
    main()
