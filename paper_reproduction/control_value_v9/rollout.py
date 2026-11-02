"""Per-case real RGB rollout; adapted from v8 without changing its gates."""
from collections import deque
import json
import time
import cv2
import numpy as np
from control_coverage_v8.scenes import cases,materialize,stage_spec,initial_zone,category,CORNER_NAMES,layout
from control_coverage_v8.gates import assess_episode,gate_aggregate
from control_coverage_v8.environment import CoverageEnvironment as ImprovedEnvironment
from control_v5.environment import transition_terminal
from control_coverage_v8.run import write_json,frozen,stop_check

STAGE='static_center'

class ValueReviewRequired(RuntimeError):
    pass

def guard_selection(root,agent,selection,training):
    value=selection['greedy_q_bound_excess']
    agent.episode_q_excess.append(value)
    if not training:
        return
    if not hasattr(agent,'q_guard_window'):
        agent.q_guard_window=deque(maxlen=200)
    agent.q_guard_window.append(value)
    if agent.update_count>=256 and len(agent.q_guard_window)==200 and sum(x>1. for x in agent.q_guard_window)>10:
        write_json(root/'value-review.json',dict(reason='persistent Q upper-bound excess',
            update=agent.update_count,environment_steps=agent.environment_steps,last200=list(agent.q_guard_window)))
        raise ValueReviewRequired('More than10 of last200 greedy Q estimates exceed the physical reward bound by1')

def episode(case, output, transport, detector, extractor, agent, training, root, fixed_action=None):
    output.mkdir(parents=True, exist_ok=False)
    scene = materialize(STAGE, case["split"], case["seed"], output)
    environment = ImprovedEnvironment(transport, detector, extractor, stage_spec(STAGE))
    state = environment.reset(scene, case["seed"])
    environment.rows[0]["initial_zone_confirmed"] = initial_zone(environment.rows[0],case["seed"],case["split"])
    # Preserve the actual failing frame and raw detector candidates before any
    # start gate can raise. This is audit output, never an input to the learner.
    write_json(output / "resolved-scene.json", transport.diagnostics)
    cv2.imwrite(str(output / "initial-rgb.png"), cv2.cvtColor(transport.rgb, cv2.COLOR_RGB2BGR))
    write_json(output / "initial-perception.json", {"candidates":[{"bbox":list(d.bbox),"confidence":float(d.confidence)} for d in environment.detections],
        "selected":environment.rows[0],"scope":"diagnostic only; no target/GT policy input"})
    if not environment.rows[0]["initial_zone_confirmed"]:
        write_json(output / "initial-zone-failure.json",environment.rows[0])
        raise RuntimeError("Real RGB initial zone or full-visibility contract failed")
    started, initial_update = time.perf_counter(), agent.update_count
    updates, random_actions, total_reward, wall_steps = [], 0, 0., []
    agent.episode_q_excess = []
    middle_saved = False
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
                    guard_selection(root, agent, selection, training)
                next_state, reward, done, row = environment.step(*action)
                if training:
                    agent.observe(state, action, reward, next_state, transition_terminal(row))
                # Commit the real transition before attempting any optimization.
                log.write(json.dumps(row | {"decision": decision, "input_state": state.tolist(),
                    "next_state": next_state.tolist(), "policy_selection": selection,
                    "updates_before": agent.update_count}, allow_nan=False) + "\n")
                state = next_state
                if not middle_saved and row["sim_time_s"] >= 4.:
                    cv2.imwrite(str(output / "mid-rgb.png"), cv2.cvtColor(transport.rgb,cv2.COLOR_RGB2BGR))
                    middle_saved = True
                if training:
                    for _ in range(1):
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
            cv2.imwrite(str(output / "interrupted-rgb.png"), cv2.cvtColor(transport.rgb, cv2.COLOR_RGB2BGR))
            write_json(output / "incident.json", {"error": repr(exc), "last_observations": environment.rows[-2:],
                "real_steps": environment.steps, "updates_total": agent.update_count})
            raise
    cv2.imwrite(str(output / "final-rgb.png"), cv2.cvtColor(transport.rgb, cv2.COLOR_RGB2BGR))
    assessed = assess_episode(STAGE, environment.rows)
    result = assessed | {"seed": case["seed"], "split": case["split"], "stage": STAGE,
        "coverage_kind":category(case["seed"],case["split"])[0], "coverage":layout(case["seed"],case["split"]), "corner":CORNER_NAMES[category(case["seed"],case["split"])[1]],
        "reward_sum": total_reward, "steps": environment.steps, "wall_seconds": time.perf_counter()-started,
        "mean_q_loss": float(np.mean([u["q_loss"] for u in updates])) if updates else None,
        "mean_parameter_objective": None,
        "network_updates": agent.update_count-initial_update, "updates_total": agent.update_count,
        "epsilon": agent.epsilon if training else 0., "random_actions": random_actions,
        "saturated_nonstay": sum(bool(row.get("saturated_nonstay")) for row in environment.rows),
        "stay_actions": sum(row["action_type"] == 0 for row in environment.rows[1:]),
        "wall_step_mean_s": float(np.mean(wall_steps)), "wall_step_p95_s": float(np.quantile(wall_steps, .95)),
        "q_lr": agent.config.actor_lr, "parameter_lr": None, "batch_size": agent.config.batch_size,
        "parameter_updates": 0, "fixed_pulse_seconds": .04,
        "q_bound_violation_count": sum(x>.01 for x in agent.episode_q_excess),
        "q_bound_diagnostic_count":len(agent.episode_q_excess),
        "max_q_bound_excess":max(agent.episode_q_excess,default=None)}
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
