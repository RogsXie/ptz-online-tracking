"""Real RGB joint centering/zoom rollouts with actual frozen video receipts."""
from collections import deque
import json
import time
import cv2
import numpy as np
from .scenes import cases,materialize,stage_spec,initial_zone,category,CORNER_NAMES,layout
from .gates import assess_episode,gate_aggregate
from .environment import CenterEnvironment as ImprovedEnvironment
from .video import CaseVideo
from control_v5.environment import transition_terminal
from control_coverage_v8.run import write_json,frozen as base_frozen,stop_check
from contextlib import contextmanager
import hashlib,pickle

@contextmanager
def frozen(agent,receipt):
    before=hashlib.sha256(pickle.dumps(agent.memory.history)).hexdigest()
    try:
        with base_frozen(agent,receipt):yield
    finally:
        after=hashlib.sha256(pickle.dumps(agent.memory.history)).hexdigest()
        receipt.update(historical_replay_before=before,historical_replay_after=after,historical_replay_unchanged=before==after)
        if before!=after:raise RuntimeError("Historical rehearsal replay mutated during evaluation")

STAGE='center_repair'

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
    video = CaseVideo(output, case["seed"]) if not training else None
    with (output / "steps.jsonl").open("w", encoding="utf-8", buffering=1) as log, \
         (output / "updates.jsonl").open("w", encoding="utf-8", buffering=1) as update_log:
        log.write(json.dumps(environment.rows[0] | {"initial": True, "state": state.tolist()}) + "\n")
        try:
            if video: video.write(transport.rgb, environment.rows[0])
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
                if video: video.write(transport.rgb, row)
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
        finally:
            if video:
                video.close()
                write_json(output / "video.json", dict(frames=video.frames, fps=25, duration_seconds=video.frames/25, scope="actual frozen development rollout; includes initial observation"))
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
    result.update(terminal_scale_log_error=result['terminal_window']['scale_log_error_mean'],
        initial_area=environment.rows[0]['area_ratio'], final_area=environment.rows[-1]['area_ratio'],
        final_fov=environment.rows[-1]['fov'], zoom_in_actions=sum(r['action_type']==5 for r in environment.rows[1:]),
        zoom_out_actions=sum(r['action_type']==6 for r in environment.rows[1:]),
        initial_task_ready=bool(environment.rows[0]['visible'] and environment.rows[0]['center_error']<=.025
            ))
    write_json(output / "summary.json", result)
    return result


class PolicyReviewRequired(ValueReviewRequired):
    pass


def evaluate(plan, output, transport, detector, extractor, agent, root, official=False):
    output.mkdir(parents=True, exist_ok=False)
    receipt, results = {}, []
    error = None
    try:
        with frozen(agent, receipt):
            for case in plan:
                result = episode(case, output / f"seed{case['seed']}", transport, detector, extractor, agent, False, root)
                results.append(result)
                write_json(output / 'progress.json', dict(completed=len(results), total=len(plan), passed=sum(r['passed'] for r in results)))
                print(f"PROBE {len(results)}/{len(plan)} seed={case['seed']} pass={result['passed']} center={result['terminal_center_mean']} reversals={result['direction_reversals']}", flush=True)
                if result['max_q_bound_excess'] > 1:
                    raise PolicyReviewRequired(f"Q bound exceeded in {case['split']} seed{case['seed']}")
                if not official and result['initial_task_ready'] and not result['passed']:
                    raise PolicyReviewRequired(f"Already-centered development seed{case['seed']} failed: {result['failures']}")
                collapsed = sum(r['terminal_center_mean'] is None or ((r['terminal_center_mean'] > .05) and (r['stay_actions'] + r['saturated_nonstay']) / r['steps'] > .8) for r in results)
                lost = sum(r['longest_loss_run_seconds'] > 1 for r in results)
                if not official and max(collapsed, lost) >= len(plan) / 2:
                    raise PolicyReviewRequired('At least half the planned development cases lost or stalled')
    except BaseException as exc:
        error = repr(exc)
        raise
    finally:
        # The context has already restored modes and verified immutable learning
        # state. Persist its actual live receipt even when an episode raised.
        write_json(output / 'freeze-receipt.json', receipt)
        centers = [r['terminal_center_mean'] if r['terminal_center_mean'] is not None else 2. for r in results]
        report = dict(purpose='held-out stage validation' if official else 'development diagnostic; not independent validation',
                      complete=len(results) == len(plan) and error is None, planned_total=len(plan), total=len(results),
                      passed_cases=sum(r['passed'] for r in results),
                      center_mean=float(np.mean(centers)) if centers else None,
                      scale_error_mean=float(np.mean([r['terminal_scale_log_error'] if r['terminal_scale_log_error'] is not None else 8. for r in results])) if results else None,
                      reward_mean=float(np.mean([r['reward_sum'] for r in results])) if results else None,
                      reversal_mean=float(np.mean([r['direction_reversals'] for r in results])) if results else None,
                      collapsed_cases=sum(r['terminal_center_mean'] is None or ((r['terminal_center_mean'] > .05) and (r['stay_actions'] + r['saturated_nonstay']) / r['steps'] > .8) for r in results),
                      lost_cases=sum(r['longest_loss_run_seconds'] > 1 for r in results),
                      freeze=receipt, episodes=results, interruption=error)
        if official and report['complete']:
            report['gate'] = gate_aggregate(STAGE, results)
        write_json(output / 'report.json', report)

    return report
