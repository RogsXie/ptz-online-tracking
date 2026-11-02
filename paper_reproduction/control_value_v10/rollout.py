"""Original RGB episodes plus immediate review stops and durable freeze receipts."""
import json
import numpy as np
from control_value_v9.rollout import episode, ValueReviewRequired
from control_coverage_v8.run import frozen, write_json
from control_coverage_v8.gates import gate_aggregate


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
                if not official and result['coverage']['layer'] == 0 and not result['passed']:
                    raise PolicyReviewRequired(f"Already-centered development seed{case['seed']} failed: {result['failures']}")
                collapsed = sum(r['terminal_center_mean'] is None or (r['terminal_center_mean'] > .05 and (r['stay_actions'] + r['saturated_nonstay']) / r['steps'] > .8) for r in results)
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
                      reward_mean=float(np.mean([r['reward_sum'] for r in results])) if results else None,
                      reversal_mean=float(np.mean([r['direction_reversals'] for r in results])) if results else None,
                      collapsed_cases=sum(r['terminal_center_mean'] is None or (r['terminal_center_mean'] > .05 and (r['stay_actions'] + r['saturated_nonstay']) / r['steps'] > .8) for r in results),
                      lost_cases=sum(r['longest_loss_run_seconds'] > 1 for r in results),
                      freeze=receipt, episodes=results, interruption=error)
        if official and report['complete']:
            report['gate'] = gate_aggregate('static_center', results)
        write_json(output / 'report.json', report)

    return report
