"""Read-only failure evidence; never changes policy, labels, thresholds or runs."""
import argparse
from collections import Counter
import json
from pathlib import Path
import numpy as np
from control_v3.run import write_json


def complete_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines(keepends=True)
            if line.endswith("\n") and line.strip()]


def analyze_episode(folder):
    rows=complete_rows(folder/"steps.jsonl")
    summary=json.loads((folder/"summary.json").read_text(encoding="utf-8"))
    decisions=rows[1:]
    worsening=[]
    departures=[]
    offcenter_stay=[]
    for before,row in zip(rows,decisions):
        if not before.get("visible") or not row.get("visible"):
            continue
        action=row["action_type"]
        if action in (1,2,3,4) and row["center_error"]>before["center_error"]+.005:
            worsening.append(row["decision"])
        if action and before["center_error"]<=.025 and row["center_error"]>.05:
            departures.append({"decision":row["decision"],"before":before["center_error"],
                "after":row["center_error"],"action":action,"duration":row["duration_s"],
                "q":row["policy_selection"].get("q_values"),"reward":row["reward"]})
        if action==0 and before["center_error"]>.025:
            offcenter_stay.append(row["decision"])
    return {"seed":summary["seed"],"passed":summary["passed"],"failures":summary["failures"],
        "initial_center":summary["initial_center_error"],"terminal_center":summary["terminal_center_mean"],
        "steps":len(decisions),"reward":summary["reward_sum"],"reversals":summary["direction_reversals"],
        "actions":dict(Counter(row["action_type"] for row in decisions)),
        "mean_duration":float(np.mean([row["duration_s"] for row in decisions])),
        "duration_min_max":[min(row["duration_s"] for row in decisions),max(row["duration_s"] for row in decisions)],
        "saturated_nonstay":sum(bool(row.get("saturated_nonstay")) for row in decisions),
        "image_worsening_actions":len(worsening),"offcenter_stay_actions":len(offcenter_stay),
        "departures_from_center":departures,"selected_availability":summary["selected_availability"],
        "wall_step_mean_s":summary["wall_step_mean_s"],"wall_step_p95_s":summary["wall_step_p95_s"]}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run",type=Path)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    results=[]
    for directory in sorted(args.run.glob("probe-*"))+sorted(args.run.glob("validation-*")):
        if not (directory/"report.json").exists():
            continue
        report=json.loads((directory/"report.json").read_text(encoding="utf-8"))
        results.append({"probe":directory.name,"purpose":report["purpose"],
            "freeze":report["freeze"],"passed_cases":report["passed_cases"],"total":report["total"],
            "cases":[analyze_episode(path.parent) for path in sorted(directory.glob("seed*/summary.json"))]})
    write_json(args.output,{"run":str(args.run.resolve()),"read_only":True,"evaluations":results,
        "note":"Recorded RGB endpoints only; observational diagnostics do not establish an untested causal repair"})
    print(json.dumps({"output":str(args.output),"completed_evaluation_sets":len(results)}))


if __name__=="__main__":
    main()
