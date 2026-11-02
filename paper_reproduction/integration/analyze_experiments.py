"""Read saved artifacts only; generate auditable descriptive experiment results."""
import argparse
import json
import math
from pathlib import Path
import numpy as np
from integration.metrics import iou


def read_jsonl(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def summarize(path):
    path = Path(path)
    result = json.loads((path / "results.json").read_text(encoding="utf-8"))
    config = json.loads((path / "configuration.json").read_text(encoding="utf-8"))
    rows = result["episodes_detail"]
    steps = []
    matched_frames = []
    matched_frame_audit_available = True
    line_reset_events = []
    recovery_frames_overlapping_line_reset = 0
    for ep in sorted(path.glob("episode-*")):
        episode_steps = read_jsonl(ep / "steps.jsonl")
        steps.extend(episode_steps)
        initial = json.loads((ep / "scenario.json").read_text(encoding="utf-8"))
        phases = {t["id"]: t.get("phase_s", 0) for t in initial.get("resolved_config", {}).get("targets", [])}
        previous_time = 0.
        previous_visible = json.loads((ep / "summary.json").read_text(encoding="utf-8")).get("IC", 0) > 0
        for step in episode_steps:
            reset_this_frame = False
            for target in initial["scenario_config"]["targets"]:
                loop = target["path"].get("loop_period_s", 0)
                if target["path"]["type"] != "line" or loop <= 0:
                    continue
                phase = phases.get(target["id"], 0)
                cycles = math.floor((step["sim_time_s"] + phase) / loop) - math.floor((previous_time + phase) / loop)
                if cycles > 0:
                    reset_this_frame = True
                    line_reset_events.append({"episode": ep.name, "target": target["id"],
                                              "sim_time_s": step["sim_time_s"], "cycles_crossed": cycles})
            if reset_this_frame and not previous_visible and step["visible_detection"]:
                recovery_frames_overlapping_line_reset += 1
            previous_time, previous_visible = step["sim_time_s"], step["visible_detection"]
        diagnostics = read_jsonl(ep / "diagnostics-only.jsonl")
        if len(diagnostics) != len(episode_steps):
            matched_frame_audit_available = False
            continue
        for step, diagnostic in zip(episode_steps, diagnostics):
            if step["selected_bbox"] is None:
                continue
            candidates = sorted([iou(step["selected_bbox"], t["gt_bbox_normalized"])
                                 for t in diagnostic["targets"] if t["gt_in_frame"]], reverse=True)
            matched = bool(candidates and candidates[0] >= .3 and
                           (len(candidates) < 2 or candidates[0] - candidates[1] >= .05))
            if matched:
                matched_frames.append(step)
    durations = [r["duration_s"] for r in steps]
    target_total = sum(len(json.loads((ep / "scenario.json").read_text(encoding="utf-8"))["scenario_config"]["targets"])
                       for ep in sorted(path.glob("episode-*")))
    total_time = sum(durations)
    all_visible = [r for r in steps if r["visible_detection"]]
    result["aggregate"] = {
        "controller": config["controller"], "partition": config["partition"],
        "episodes": len(rows), "action_steps": len(steps), "sim_time_s": total_time,
        "mean_duration_s": float(np.mean(durations)),
        "duration_min_s": min(durations), "duration_max_s": max(durations),
        "distinct_duration_values_rounded_4dp": len({round(d, 4) for d in durations}),
        "TSR_detection_proxy_pooled": len(all_visible) / max(1, len(steps)),
        "visible_time_fraction_sampled": sum(r["duration_s"] for r in all_visible) / max(1e-12, total_time),
        "mean_center_error_visible_only": float(np.mean([r["center_error_normalized"] for r in all_visible])) if all_visible else None,
        "mean_area_visible_only": float(np.mean([r["target_area_ratio"] for r in all_visible])) if all_visible else None,
        "selected_gt_matched_fraction_of_selected": len(matched_frames) / len(all_visible) if all_visible and matched_frame_audit_available else None,
        "mean_center_error_gt_matched_only": float(np.mean([r["center_error_normalized"] for r in matched_frames])) if matched_frames and matched_frame_audit_available else None,
        "mean_area_gt_matched_only": float(np.mean([r["target_area_ratio"] for r in matched_frames])) if matched_frames and matched_frame_audit_available else None,
        "matched_frame_audit_available": matched_frame_audit_available,
        "IC_mean": float(np.mean([r["IC"] for r in rows])),
        "EC_mean": float(np.mean([r["EC"] for r in rows])),
        "initial_confidence_ge_09_fraction": float(np.mean([r["IC"] >= .9 for r in rows])),
        "strict_SR": result["SR_offline_gt_verified"], "estimated_SR": result["SR_cfff_estimated"],
        "TC_mean": result["TC_offline_mean"], "SS_successful_only": result["SS_successful_only"],
        "TC_pooled": sum(len(r["completed_gt_ids"]) for r in rows) / max(1, target_total),
        "target_instance_total": target_total,
        "false_completion_episodes": sum(r["false_completion_estimate"] for r in rows),
        "rejected_completion_events": sum(not m["accepted"] for r in rows for m in r["completion_matches"]),
        "raw_detector_nonempty_fraction": sum(r["raw_detector_count"] > 0 for r in steps) / max(1, len(steps)),
        "CFFF_filtered_fraction": sum(r["selected_absence_reason"] == "CFFF_filtered" for r in steps) / max(1, len(steps)),
        "recovery_events": sum(len(r["recovery_steps"]) for r in rows),
        "line_path_reset_events": line_reset_events,
        "recovery_frames_temporally_overlapping_line_reset": recovery_frames_overlapping_line_reset,
        "longest_selected_unavailable_seconds": max(r["longest_selected_unavailable_seconds"] for r in rows),
        "mean_detector_seconds": float(np.mean([r["detector_seconds"] for r in steps])),
        "mean_policy_seconds": float(np.mean([r["policy_seconds"] for r in steps])),
        "mean_step_wall_seconds": float(np.mean([r["step_wall_seconds"] for r in steps])),
        "total_wall_seconds": result["wall_seconds"],
        "unique_pairs": [[r["scenario"], r["seed"]] for r in rows],
        "network_updates": len(read_jsonl(path / "updates.jsonl")),
        "action_counts": {str(k): sum(r["action_type"] == k for r in steps) for k in range(7)}}
    a = result["aggregate"]
    (path / "aggregate-derived.json").write_text(json.dumps(a, indent=2), encoding="utf-8")
    if not (path / "aggregate.json").exists():
        (path / "aggregate.json").write_text(json.dumps(a, indent=2), encoding="utf-8")
    return a


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    values = [summarize(path) for path in args.runs]
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "comparison.json").write_text(json.dumps(values, indent=2), encoding="utf-8")
    lines = ["# 真实UE短训练与独立评估", "", "这些是实现验证和描述性结果。训练预算较小且各学习器预算不同，不能据此判断算法优劣或宣称论文性能复现。", "",
             "| 控制器/划分 | N | 动作数 | 严格SR | TC | TSR检测代理 | 中心误差 | 平均面积 | 模拟秒 | 时长均值/范围 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    fmt = lambda v: "无检测" if v is None else f"{v:.4f}"
    for a in values:
        lines.append(f"| {a['controller']}/{a['partition']} | {a['episodes']} | {a['action_steps']} | {a['strict_SR']:.1%} | {a['TC_pooled']:.1%} | {a['TSR_detection_proxy_pooled']:.1%} | {fmt(a['mean_center_error_visible_only'])} | {fmt(a['mean_area_visible_only'])} | {a['sim_time_s']:.2f} | {a['mean_duration_s']:.3f} / {a['duration_min_s']:.3f}–{a['duration_max_s']:.3f} |")
    lines += ["", "严格SR/TC须在离线10帧窗口匹配同一GT；ambiguous匹配拒绝。无目标框时，中心和尺度统计排除该步，TSR包含该步。", "",
              "初始confidence≥.9的比例、虚假完成、未恢复失检、动作开销与每场景/种子明细见comparison.json和各run。直线路径周期重置会产生位置跳变；重置事件与恢复帧的时间重合另列，不能把周期重新出现解释成策略搜索/重捕获。视频为动作结束采样帧，时长以JSONL为准。", "",
              "检测器固定在独立train535上适配、validation64/test64各一次评估。检测数据采集的GT对准只用于采样标签，RL初始化不使用GT。测试复合配置和种子没有进入训练。"]
    (args.output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
