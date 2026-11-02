"""Read-only, repeatable reporting for focused UE GPU runs, including live snapshots.

This module imports no detector, learner, torch, network client or process launcher.
Every value is derived from existing logs. Files inside input runs are never written.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics

REPRO = Path(__file__).resolve().parents[1]
CATEGORIES = {"focus_speed": "variable_speed", "focus_far": "far_small",
              "focus_turn": "turning", "focus_occlusion": "occlusion", "focus_low_light": "low_light"}


def digest(blob):
    return hashlib.sha256(blob).hexdigest()


def read_json(path, warnings=None):
    path = Path(path)
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as error:
        if warnings is not None:
            warnings.append({"file": str(path), "issue": "unreadable_json_snapshot", "detail": str(error)})
        return None


def read_jsonl(path, warnings):
    """Keep line ordinals so a partial/invalid row cannot shift GT alignment."""
    path = Path(path)
    if not path.is_file():
        return {}, None
    blob = path.read_bytes()
    rows = {}
    lines = blob.decode("utf-8-sig").splitlines()
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            rows[index] = json.loads(line)
        except ValueError as error:
            warnings.append({"file": str(path), "line": index+1,
                             "issue": "partial_tail" if index == len(lines)-1 else "invalid_jsonl_row",
                             "detail": str(error)})
    return rows, {"path": str(path), "snapshot_sha256": digest(blob), "snapshot_bytes": len(blob),
                  "valid_rows": len(rows), "read_semantics": "one read of a potentially growing file"}


def finite(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def valid_box(box):
    return (isinstance(box, (list, tuple)) and len(box) == 4 and all(finite(v) for v in box)
            and box[2] > box[0] and box[3] > box[1])


def iou(a, b):
    intersection = max(0., min(a[2], b[2])-max(a[0], b[0])) * max(0., min(a[3], b[3])-max(a[1], b[1]))
    area_a, area_b = (a[2]-a[0])*(a[3]-a[1]), (b[2]-b[0])*(b[3]-b[1])
    return intersection/max(1e-12, area_a+area_b-intersection)


def box_metrics(box):
    area = (box[2]-box[0])*(box[3]-box[1])
    return {"center_error": math.hypot(box[0]+box[2]-1, box[1]+box[3]-1),
            "area_ratio": area, "scale_log_error_to_005": abs(math.log(max(area, 1e-12)/.05))}


def geometry_quality(values):
    return values["center_error"] <= .15 and .02 <= values["area_ratio"] <= .08


def derive_row(step, diagnostic):
    box = step.get("selected_bbox")
    selected = valid_box(box)
    candidates = sorted([(iou(box, target["gt_bbox_normalized"]), target) for target in diagnostic.get("targets", [])
                         if selected and target.get("gt_in_frame") and valid_box(target.get("gt_bbox_normalized"))],
                        key=lambda item: item[0], reverse=True)
    overlap, target = candidates[0] if candidates else (0., None)
    ambiguous = len(candidates) > 1 and overlap-candidates[1][0] < .05
    matched = target is not None and overlap >= .3 and not ambiguous
    det_values = box_metrics(box) if selected else None
    gt_values = box_metrics(target["gt_bbox_normalized"]) if matched else None
    confidence = step.get("confidence", step.get("state", [0, 0, 0, 0])[3])
    targets = diagnostic.get("targets", [])
    return {"duration_s": float(step.get("duration_s", 0.)), "selected": selected,
            "action_type": step.get("action_type"), "policy_random": step.get("policy_selection", {}).get("random"),
            "raw_detector_nonempty": bool(step.get("raw_detector_count", len(step.get("detections", [])))),
            "cfff_filtered": step.get("selected_absence_reason") == "CFFF_filtered",
            "matched": matched, "ambiguous": ambiguous, "match_iou": overlap,
            "matched_gt_id": target.get("id") if matched else None,
            "matched_center_ray_clear": matched and target.get("gt_center_occluded") is False,
            "matched_center_ray_occluded": matched and target.get("gt_center_occluded") is True,
            "any_gt_projection_in_frame": any(t.get("gt_in_frame", False) for t in targets),
            "any_gt_center_ray_occluded": any(t.get("gt_center_occluded", False) for t in targets),
            "selected_quality": bool(selected and confidence >= .9 and geometry_quality(det_values)),
            "matched_gt_geometry_quality": bool(matched and geometry_quality(gt_values)),
            "matched_gt_joint_quality": bool(matched and confidence >= .9 and geometry_quality(gt_values)),
            "detector_box_proxy": det_values if matched else None,
            "gt_amodal_projection": gt_values,
            "gt_amodal_projection_center_ray_clear": gt_values if matched and target.get("gt_center_occluded") is False else None,
            "timings": {key: step[key] for key in ("policy_seconds", "detector_seconds", "environment_seconds",
                         "learner_update_seconds", "compute_and_record_wall_seconds", "environment_and_record_seconds")
                        if finite(step.get(key))}}


class Series:
    def __init__(self):
        self.count, self.total, self.weight, self.weighted = 0, 0., 0., 0.

    def add(self, value, weight):
        if finite(value):
            self.count += 1; self.total += value
            self.weight += weight; self.weighted += value*weight

    def result(self):
        return {"sample_count": self.count, "sampled_duration_s": self.weight,
                "action_mean": self.total/self.count if self.count else None,
                "action_end_time_weighted_mean": self.weighted/self.weight if self.weight else None}


class ActionStats:
    FLAGS = ("selected", "raw_detector_nonempty", "cfff_filtered", "matched", "ambiguous",
             "matched_center_ray_clear", "matched_center_ray_occluded", "any_gt_projection_in_frame",
             "any_gt_center_ray_occluded", "selected_quality", "matched_gt_geometry_quality", "matched_gt_joint_quality")
    def __init__(self):
        self.count, self.duration = 0, 0.
        self.flags = {key: [0, 0.] for key in self.FLAGS}
        self.values, self.timings = {}, {}
        self.actions = {str(index): 0 for index in range(7)}
        self.random_flags = {"true": 0, "false": 0, "unknown": 0}
        self.minimum_duration = self.maximum_duration = None

    def add(self, row):
        duration = max(0., row["duration_s"])
        self.count += 1; self.duration += duration
        action = row.get("action_type")
        if action in range(7):
            self.actions[str(action)] += 1
        random_flag = row.get("policy_random")
        self.random_flags["true" if random_flag is True else "false" if random_flag is False else "unknown"] += 1
        self.minimum_duration = duration if self.minimum_duration is None else min(duration, self.minimum_duration)
        self.maximum_duration = duration if self.maximum_duration is None else max(duration, self.maximum_duration)
        for key in self.FLAGS:
            if row[key]:
                self.flags[key][0] += 1; self.flags[key][1] += duration
        for group in ("detector_box_proxy", "gt_amodal_projection", "gt_amodal_projection_center_ray_clear"):
            for key, value in (row[group] or {}).items():
                self.values.setdefault(group, {}).setdefault(key, Series()).add(value, duration)
        for key, value in row["timings"].items():
            self.timings.setdefault(key, Series()).add(value, duration)

    def result(self):
        selected = self.flags["selected"][0]
        return {"paired_action_count": self.count, "commanded_simulation_seconds": self.duration,
                "executed_action_counts": self.actions, "policy_random_flag_counts": self.random_flags,
                "executed_duration_s": {"minimum": self.minimum_duration, "maximum": self.maximum_duration,
                                        "action_mean": self.duration/self.count if self.count else None},
                "availability_and_quality": {key: {"actions": count, "sampled_duration_s": duration,
                    "action_fraction": count/self.count if self.count else None,
                    "action_end_time_weighted_fraction": duration/self.duration if self.duration else None}
                    for key, (count, duration) in self.flags.items()},
                "gt_match_among_selected_fraction": self.flags["matched"][0]/selected if selected else None,
                "box_metrics": {group: {key: series.result() for key, series in fields.items()}
                                for group, fields in self.values.items()},
                "recorded_timing_seconds": {key: series.result() for key, series in self.timings.items()}}


def recovery(rows, flag):
    run_steps, run_seconds, longest = 0, 0., 0.
    recoveries = []
    for row in rows:
        if row[flag]:
            if run_steps:
                recoveries.append({"sampled_unavailable_actions": run_steps, "sampled_unavailable_seconds": run_seconds})
            run_steps, run_seconds = 0, 0.
        else:
            run_steps += 1; run_seconds += max(0., row["duration_s"])
            longest = max(longest, run_seconds)
    return {"recovery_events": len(recoveries), "recoveries": recoveries,
            "unrecovered_sampled_actions": run_steps, "unrecovered_sampled_seconds": run_seconds,
            "longest_sampled_unavailable_seconds": longest}


def check_video(folder, summary, enabled=True):
    if not summary or not summary.get("terminated"):
        return {"status": "pending_or_unfinished", "checked": False}
    if not summary.get("video_recorded"):
        return {"status": "not_scheduled", "checked": False}
    if not enabled:
        return {"status": "disabled", "checked": False}
    import cv2
    expected_frames = summary["action_steps"]+1
    details = []
    for name in ("sensor-rgb.mp4", "policy-boxed.mp4"):
        path = folder/name
        capture = cv2.VideoCapture(str(path)) if path.exists() else None
        detail = {"file": str(path), "exists": path.is_file(), "expected_frames": expected_frames}
        if capture is not None and capture.isOpened():
            width, height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frames, fps = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), capture.get(cv2.CAP_PROP_FPS)
            first_ok, _ = capture.read()
            capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, frames-1))
            last_ok, _ = capture.read()
            detail.update(width=width, height=height, frames=frames, fps=fps,
                          first_frame_decodes=first_ok, last_frame_decodes=last_ok,
                          passed=width == 960 and height == 540 and frames == expected_frames and abs(fps-10.) < .01 and first_ok and last_ok)
        else:
            detail["passed"] = False
        if capture is not None:
            capture.release()
        details.append(detail)
    pngs = {name: (folder/name).is_file() for name in ("first-rgb.png", "first-boxed.png", "last-boxed.png")}
    passed = all(d["passed"] for d in details) and all(pngs.values())
    return {"status": "passed" if passed else "failed", "checked": True, "videos": details, "png_exists": pngs}


def summarize_episode(folder, warnings, videos=True):
    summary = read_json(folder/"summary.json", warnings)
    scenario = read_json(folder/"scenario.json", warnings) or {}
    steps, steps_source = read_jsonl(folder/"steps.jsonl", warnings)
    diagnostics, diagnostics_source = read_jsonl(folder/"diagnostics-only.jsonl", warnings)
    keys = sorted(set(steps) & set(diagnostics))
    rows = [derive_row(steps[key], diagnostics[key]) for key in keys]
    if len(steps) != len(diagnostics):
        warnings.append({"file": str(folder), "issue": "step_diagnostic_snapshot_row_count_differs",
                         "step_rows": len(steps), "diagnostic_rows": len(diagnostics)})
    completed = bool(summary and summary.get("terminated"))
    if completed and summary.get("action_steps") != len(steps):
        warnings.append({"file": str(folder), "issue": "completed_summary_action_count_mismatch",
                         "summary_steps": summary.get("action_steps"), "logged_steps": len(steps)})
    config = scenario.get("scenario_config", {})
    profile = config.get("focus_profile", {})
    name = (summary or {}).get("scenario") or profile.get("family") or config.get("name") or "unknown"
    seed = (summary or {}).get("seed", scenario.get("seed"))
    maximum_streak, streak, previous_id, previous_line = 0, 0, None, None
    for key, row in zip(keys, rows):
        same_identity = row["matched_gt_id"] == previous_id
        contiguous = previous_line is not None and key == previous_line+1
        streak = (streak+1 if same_identity and contiguous else 1) if row["matched_gt_joint_quality"] else 0
        maximum_streak = max(maximum_streak, streak)
        previous_id, previous_line = row["matched_gt_id"], key
    stats = ActionStats()
    for row in rows:
        stats.add(row)
    record = {"folder": str(folder), "scenario": name, "category": CATEGORIES.get(name, name), "seed": seed,
              "status": "complete" if completed else "unfinished_recorded" if summary else "in_progress_or_unfinalized",
              "observed_action_rows": len(steps), "paired_action_rows": len(rows),
              "summary": summary, "observed_action_metrics": stats.result(),
              "selected_availability_recovery": recovery(rows, "selected"),
              "gt_matched_availability_recovery": recovery(rows, "matched"),
              "offline_gt_projection_joint_quality_max_streak": maximum_streak,
              "terminal_gt_projection_joint_quality_verified": bool(completed and summary.get("success") and streak >= 10),
              "video_verification": check_video(folder, summary, videos),
              "sources": {"steps": steps_source, "diagnostics": diagnostics_source}}
    return record, rows


def episode_rates(records):
    completed = [r for r in records if r["status"] == "complete"]
    zero_resets = sum(r["status"] == "zero_action_reset" for r in records)
    count = len(completed)
    mean_flag = lambda key: sum(bool(r["summary"].get(key)) for r in completed)/count if count else None
    target_total = sum(r["summary"].get("target_count", 0) for r in completed)
    return {"completed_episodes": count,
            "zero_action_resets": zero_resets,
            "unfinished_or_active_episodes": len(records)-count-zero_resets,
            "strict_gt_identity_verified_SR": mean_flag("success"),
            "online_completion_estimate_rate": mean_flag("termination_success_estimate"),
            "paper_confidence_condition_ever_rate": mean_flag("paper_confidence_condition_ever"),
            "TC_offline_identity_matched_pooled": sum(r["summary"].get("completed_distinct_count", 0) for r in completed)/target_total if target_total else None,
            "terminal_gt_projection_joint_quality_verified_rate": sum(r["terminal_gt_projection_joint_quality_verified"] for r in completed)/count if count else None}


def mark_finalized_zero_action_reset(records, final_stats, planned_transitions):
    """Recognize DummyVecEnv's final reset only after full PPO accounting agrees.

    A live zero-action episode or a nonzero unfinished episode remains pending.
    This changes report labels only, never the scientific source records.
    """
    if not final_stats or not records or not finite(planned_transitions) or planned_transitions <= 0:
        return
    completed = [record for record in records if record["status"] == "complete"]
    if not (final_stats.get("actual_transitions") == planned_transitions
            == final_stats.get("model_num_timesteps") == final_stats.get("accounted_transitions")
            == final_stats.get("completed_episode_transitions")
            == sum(record["observed_action_rows"] for record in records)
            and final_stats.get("completed_episodes") == len(completed)
            and final_stats.get("reset_count") == len(completed)+1
            and final_stats.get("unfinished_episode_steps") == 0
            and final_stats.get("interrupted_episodes") == 0
            and final_stats.get("interrupted_episode_transitions") == 0):
        return
    record = records[-1]
    summary, case = record.get("summary") or {}, final_stats.get("unfinished_case") or {}
    sources = record.get("sources", {})
    empty_logs = all(sources.get(key) is not None and sources[key]["snapshot_bytes"] == 0
                     for key in ("steps", "diagnostics"))
    if not (record["status"] == "unfinished_recorded" and summary.get("action_steps") == 0
            and record["observed_action_rows"] == record["paired_action_rows"] == 0 and empty_logs
            and summary.get("terminated") is False and summary.get("record_stop_reason") == "adapter_close"
            and summary.get("unfinished_episode_at_training_end") is False
            and summary.get("interrupted_by_reset") is False
            and case.get("case_index") == summary.get("episode")
            and case.get("scenario_name") == record["scenario"] and case.get("seed") == record["seed"]):
        return
    record["status"] = "zero_action_reset"
    record["pending_transitions"] = 0
    record["classification_note"] = "Final DummyVecEnv reset after all PPO transitions completed; empty action/diagnostic logs and final accounting agree."
    record["video_verification"] = {"status": "zero_action_reset", "checked": False,
                                    "note": "No action was executed; no unfinished transition or pending episode video."}


def phase_specs(config, run):
    if config.get("task") == "train" and config.get("controller") == "hascam":
        plans = config.get("resolved_training_plans", [])
        specs = [{"name": "validation-initial", "kind": "evaluation", "planned_episodes": len(config.get("resolved_validation_plan", []))}]
        for index, plan in enumerate(plans):
            specs.extend([{"name": f"stage-{index+1}-{plan['stage']}", "kind": "training", "planned_episodes": len(plan["cases"])},
                          {"name": f"validation-stage-{index+1}", "kind": "evaluation", "planned_episodes": len(config.get("resolved_validation_plan", []))}])
        if config.get("complete_fixed_curriculum"):
            specs.append({"name": "test-final", "kind": "evaluation", "planned_episodes": len(config.get("resolved_test_plan", []))})
        return specs
    if config.get("task") == "train" and config.get("controller") == "ppo_fixed":
        return [{"name": "train", "kind": "training", "planned_episodes": None,
                 "planned_transitions": config.get("ppo", {}).get("total_steps")},
                {"name": "validation-final", "kind": "evaluation", "planned_episodes": len(config.get("resolved_validation_plan", []))}]
    available = [name for name in ("validation", "test") if (run/name).is_dir()]
    return [{"name": name, "kind": "evaluation", "planned_episodes": None} for name in available] or [
        {"name": "evaluation", "kind": "evaluation", "planned_episodes": None}]


def summarize_phase(run, spec, warnings, videos):
    folder = run/spec["name"]
    if not folder.is_dir():
        return {**spec, "status": "pending", "episode_rates": None, "observed_action_metrics": None, "episodes": []}
    records, all_rows = [], []
    grouped = {}
    for episode in sorted(folder.glob("episode-*")):
        if not episode.is_dir():
            continue
        record, rows = summarize_episode(episode, warnings, videos)
        records.append(record); all_rows.extend(rows)
        grouped.setdefault(record["category"], {"records": [], "rows": []})["records"].append(record)
        grouped[record["category"]]["rows"].extend(rows)
    def action_stats(rows):
        stats = ActionStats()
        for row in rows:
            stats.add(row)
        return stats.result()
    result = read_json(folder/"results.json", warnings)
    progress = read_json(folder/"progress.json", warnings)
    ppo_final = read_json(run/"training-statistics.json", warnings) if spec["name"] == "train" else None
    updates, _ = read_jsonl(folder/"updates.jsonl", warnings)
    has_update_log = (folder/"updates.jsonl").is_file()
    wall = result.get("wall_seconds") if result else None
    completed_rows = sum(r["observed_action_rows"] for r in records if r["status"] == "complete")
    mark_finalized_zero_action_reset(records, ppo_final, spec.get("planned_transitions"))
    rates = episode_rates(records)
    finalized = result is not None or ppo_final is not None
    record_count_mismatch = finalized and spec.get("planned_episodes") is not None and rates["completed_episodes"] != spec["planned_episodes"]
    transition_count_mismatch = ppo_final is not None and sum(r["observed_action_rows"] for r in records) != ppo_final.get("actual_transitions")
    if record_count_mismatch or transition_count_mismatch:
        warnings.append({"file": str(folder), "issue": "finalized_phase_record_count_mismatch",
                         "planned_episodes": spec.get("planned_episodes"), "completed_episode_records": rates["completed_episodes"],
                         "ppo_final_transitions": (ppo_final or {}).get("actual_transitions"),
                         "observed_action_rows": sum(r["observed_action_rows"] for r in records)})
    verified = [r["video_verification"] for r in records]
    return {**spec, "status": "finalized_with_incomplete_records" if record_count_mismatch or transition_count_mismatch else "complete" if finalized else "in_progress_or_unfinalized",
            "episode_rates": rates, "observed_action_rows": sum(r["observed_action_rows"] for r in records),
            "observed_action_metrics": action_stats(all_rows),
            "by_category": {name: {"episode_rates": episode_rates(group["records"]), "observed_action_metrics": action_stats(group["rows"])}
                            for name, group in grouped.items()},
            "phase_wall_seconds_including_resets_recording_checkpoints": wall,
            "completed_actions_per_phase_wall_second": completed_rows/wall if wall else None,
            "sum_episode_wall_seconds_excludes_resets": sum((r["summary"] or {}).get("episode_wall_seconds", 0) for r in records),
            "network_updates_logged_in_phase": len(updates) if has_update_log and spec["name"] != "train" else None,
            "hard_target_syncs_logged_in_phase": sum(bool(r.get("target_synchronised")) for r in updates.values()) if has_update_log and spec["name"] != "train" else None,
            "last_logged_update_count_total": max((r.get("update_count", 0) for r in updates.values()), default=None),
            "last_logged_environment_steps_total": max((r.get("environment_steps", 0) for r in updates.values()), default=None),
            "ppo_final_training_statistics": ppo_final, "progress_snapshot": progress,
            "video_verification_counts": {status: sum(r["status"] == status for r in verified) for status in
                ("passed", "failed", "pending_or_unfinished", "zero_action_reset", "not_scheduled", "disabled")},
            "episodes": records}


def telemetry(path, warnings):
    if not path.is_file():
        return {"status": "pending", "samples": 0}
    blob = path.read_bytes()
    rows = []
    for index, row in enumerate(csv.DictReader(blob.decode("utf-8-sig").splitlines())):
        try:
            values = {key: float(value) for key, value in row.items()}
            if not all(math.isfinite(v) for v in values.values()):
                raise ValueError("nonfinite telemetry")
            rows.append(values)
        except (ValueError, TypeError) as error:
            warnings.append({"file": str(path), "row": index+2, "issue": "invalid_telemetry_snapshot", "detail": str(error)})
    if not rows:
        return {"status": "pending", "samples": 0, "snapshot_sha256": digest(blob)}
    fields = ("gpu_util_percent", "memory_used_mib", "memory_total_mib", "power_w")
    return {"status": "observed", "samples": len(rows), "unix_time_first": rows[0]["unix_time"],
            "unix_time_last": rows[-1]["unix_time"], "snapshot_sha256": digest(blob),
            "statistics": {key: {"mean": statistics.mean(r[key] for r in rows), "min": min(r[key] for r in rows),
                                 "max": max(r[key] for r in rows)} for key in fields},
            "scope": "whole GPU including UE rendering, these models and concurrent processes; not model-only allocation or a CPU speedup"}


def summarize_run(run, videos=True):
    warnings = []
    config = read_json(run/"configuration.json", warnings)
    if config is None:
        return {"run": str(run), "status": "missing_or_unreadable_configuration", "warnings": warnings, "phases": []}
    root_result = read_json(run/"results.json", warnings)
    launcher = read_json(run/"launcher.json", warnings)
    status = "complete" if root_result is not None else "stopped_incomplete" if launcher and launcher.get("owned_process_stopped") else "in_progress_or_unfinalized"
    phases = [summarize_phase(run, spec, warnings, videos) for spec in phase_specs(config, run)]
    rollout_rows, rollout_source = read_jsonl(run/"rollouts.jsonl", warnings)
    update_counts = [p["network_updates_logged_in_phase"] for p in phases if p.get("network_updates_logged_in_phase") is not None]
    sync_counts = [p["hard_target_syncs_logged_in_phase"] for p in phases if p.get("hard_target_syncs_logged_in_phase") is not None]
    learning = ({"actual_pdqn_updates_logged": sum(update_counts) if update_counts else None,
                 "actual_hard_target_syncs_logged": sum(sync_counts) if sync_counts else None,
                 "last_logged_environment_steps_total": max((p["last_logged_environment_steps_total"] for p in phases if p.get("last_logged_environment_steps_total") is not None), default=None)}
                if config.get("controller") == "hascam" and config.get("task") == "train" else
                {"final_training_statistics": read_json(run/"training-statistics.json", warnings),
                 "full_rollouts_collected_logged": len(rollout_rows) if rollout_source else None,
                 "last_epoch_count_before_latest_collected_rollout_training": max((r.get("epoch_updates_before_this_rollout_training", 0) for r in rollout_rows.values()), default=None),
                 "provisional_note": "rollout log is emitted before optimizer training; only final training-statistics authoritatively counts trained rollouts/epochs/optimizer calls"}
                if config.get("controller") == "ppo_fixed" and config.get("task") == "train" else None)
    return {"run": str(run), "status": status, "controller": config.get("controller"),
            "task": config.get("task"), "task_profile": config.get("task_profile"),
            "compute": config.get("compute"), "detector": config.get("detector"), "reid": config.get("reid"),
            "quality_task_extension": config.get("quality_task_extension"), "capture": config.get("capture"),
            "pdqn": config.get("pdqn"), "ppo": config.get("ppo"),
            "budget_comparison": config.get("budget_comparison"),
            "learning_statistics": learning,
            "source_sha256_recorded_at_launch": config.get("source_sha256"),
            "configuration_sha256": digest((run/"configuration.json").read_bytes()),
            "recorded_total_wall_seconds": (root_result or {}).get("total_wall_seconds"),
            "launcher": launcher, "gpu_telemetry": telemetry(run/"gpu-telemetry.csv", warnings),
            "phases": phases, "warnings": warnings}


def detector_evidence(folder):
    output = {}
    for label, name in (("v3_validation", "yolo26s-ue-gpu30-summary.json"),
                        ("v3_test", "yolo26s-ue-gpu30-test-summary.json"),
                        ("focused_validation", "focused-yolo26s_ue-summary.json")):
        path = folder/name
        data = read_json(path)
        if data is not None:
            output[label] = {"path": str(path), "sha256": digest(path.read_bytes()),
                             "metrics": data.get("validation_labelled_only", data.get("all")),
                             "model_sha256": data.get("provenance", {}).get("sha256"),
                             "selection": data.get("selection")}
        else:
            output[label] = {"status": "pending", "path": str(path)}
    return output


def fmt(value, percent=False):
    if value is None:
        return "pending / 无样本"
    return f"{value:.1%}" if percent else f"{value:.4f}"


def render_markdown(report):
    lines = ["# 五类 UE GPU 实验的离线记录", "", f"生成时间：{report['generated_utc']}。这是读文件时的快照；运行中阶段会明确标为 pending 或未结束。", "",
             "默认 quality 是显式任务扩展：连续10次检测置信度≥.9、径向中心误差≤.15、面积比例.02–.08才完成并提交Fmon。paper模式保留原confidence-only/128决策规则，两者不可混报。严格SR在在线完成之外核验连续10次同一GT身份（IoU≥.3，前两名IoU差≥.05）；它并不等于用GT几何再次验证中心/尺度。后者另列终点GT投影联合质量。", "",
             "HAS-CAM课程按回合预算，PPO按实际transition预算，固定规则没有学习；预算、检测器或协议不同的行不作算法排名。网络参数变化、优化次数和硬同步只证明学习计算实际发生，不能单独证明控制或任务质量改善；须结合留出验证和动作分布。", "",
             "| Run / 阶段 | 协议 | 状态 | 完成回合 | 已记录动作 | 严格身份SR | 在线完成 | 论文置信条件 | 终点GT投影质量 |",
             "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for run in report["runs"]:
        label = Path(run["run"]).name
        for phase in run["phases"]:
            rates = phase.get("episode_rates") or {}
            lines.append(f"| {label}/{phase['name']} | {run.get('task_profile', 'unknown')} | {phase['status']} | {rates.get('completed_episodes', 'pending')} | {phase.get('observed_action_rows', 'pending')} | {fmt(rates.get('strict_gt_identity_verified_SR'), True)} | {fmt(rates.get('online_completion_estimate_rate'), True)} | {fmt(rates.get('paper_confidence_condition_ever_rate'), True)} | {fmt(rates.get('terminal_gt_projection_joint_quality_verified_rate'), True)} |")
    lines += ["", "动作级指标包含已记录的活动回合；成功率只以已终止回合为分母。每一类和每一seed的明细在comparison.json，未结束回合不记失败。完整PPO训练后由DummyVecEnv额外重置的零动作记录单列zero_action_reset，既不计成功率分母，也不算待处理回合。", "",
              "| Run / 阶段 | 配对动作 | GT匹配存在/动作 | GT匹配存在/采样时间 | 检测框中心proxy | GT投影中心 | GT投影面积 | GT投影尺度log误差 |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for run in report["runs"]:
        for phase in run["phases"]:
            stats = phase.get("observed_action_metrics")
            if not stats:
                continue
            flags = stats["availability_and_quality"]
            boxes = stats["box_metrics"]
            field = lambda group, key: boxes.get(group, {}).get(key, {}).get("action_mean")
            lines.append(f"| {Path(run['run']).name}/{phase['name']} | {stats['paired_action_count']} | {fmt(flags['matched']['action_fraction'], True)} | {fmt(flags['matched']['action_end_time_weighted_fraction'], True)} | {fmt(field('detector_box_proxy', 'center_error'))} | {fmt(field('gt_amodal_projection', 'center_error'))} | {fmt(field('gt_amodal_projection', 'area_ratio'))} | {fmt(field('gt_amodal_projection', 'scale_log_error_to_005'))} |")
    lines += ["", "检测框proxy和GT投影几何均限定在选中框匹配GT的动作；没有匹配的动作不填0，并另报可用率、样本数和采样时长，避免条件均值隐藏丢失。JSON同时保存动作均值和按命令动作时长加权的动作末采样均值；这些都不是连续时间可见性。", "",
              "| Run / 验证或测试阶段 | 类别 | 完成回合 | 严格身份SR | GT匹配存在/动作 | GT投影中心误差 | GT投影尺度log误差 |",
              "|---|---|---:|---:|---:|---:|---:|"]
    for run in report["runs"]:
        for phase in run["phases"]:
            if phase["kind"] != "evaluation":
                continue
            for category, group in phase.get("by_category", {}).items():
                rates, stats = group["episode_rates"], group["observed_action_metrics"]
                gt = stats["box_metrics"].get("gt_amodal_projection", {})
                lines.append(f"| {Path(run['run']).name}/{phase['name']} | {category} | {rates['completed_episodes']} | {fmt(rates['strict_gt_identity_verified_SR'], True)} | {fmt(stats['availability_and_quality']['matched']['action_fraction'], True)} | {fmt(gt.get('center_error', {}).get('action_mean'))} | {fmt(gt.get('scale_log_error_to_005', {}).get('action_mean'))} |")
    lines += ["",
              "GT框是车辆世界轴对齐包围盒八角的未裁剪完整车体投影近似，可超出[0,1]；其中心不等于三维车辆中心投影，转弯时AABB可能变松。gt_in_frame仅表示八角在前方且框与画幅相交。gt_center_occluded仅是相机到AABB中心的一条射线；false不证明车体已完整出现或恢复跟踪。JSON额外列出中心射线清晰子集，但不把它称为可见mask。", "",
              "失检/恢复分别按选中检测存在与离线GT匹配存在计算。前者会把误检当存在；恢复秒数是动作末采样按命令时长累积，不是精确连续遮挡时长。", "",
              "| Run | GPU采样数 | 平均利用率 | 设备显存峰值MiB | 并发采样重叠 |",
              "|---|---:|---:|---:|---|"]
    for run in report["runs"]:
        gpu = run.get("gpu_telemetry", {})
        values = gpu.get("statistics", {})
        lines.append(f"| {Path(run['run']).name} | {gpu.get('samples', 0)} | {fmt(values.get('gpu_util_percent', {}).get('mean'))} | {fmt(values.get('memory_used_mib', {}).get('max'))} | {', '.join(gpu.get('overlapping_run_samples', [])) or '无已记录重叠'} |")
    lines += ["", "GPU遥测包含UE渲染和所有并发进程，不能当单模型显存、单模型利用率或CPU加速比。JSON按阶段记录真实动作、更新、硬同步、PPO完整rollout/epoch/optimizer计数和阶段墙钟。日志中的pipeline计时截止写step行前；完整阶段墙钟还包含重置、记录和checkpoint。", ""]
    for run in report["runs"]:
        learned = run.get("learning_statistics") or {}
        if "actual_pdqn_updates_logged" in learned:
            lines.append(f"- {Path(run['run']).name}：已记录P-DQN更新 {learned['actual_pdqn_updates_logged']} 次，硬同步 {learned['actual_hard_target_syncs_logged']} 次，最近更新行的累计真实交互 {learned['last_logged_environment_steps_total']} 条（末尾未触发更新的动作另计入阶段动作数）。")
        elif "final_training_statistics" in learned:
            final = learned["final_training_statistics"]
            detail = (f"最终真实交互 {final.get('actual_transitions')}、训练完整rollout {final.get('rollout_updates')}、epoch计数 {final.get('sb3_epoch_updates')}、独立hook计数optimizer调用 {final.get('optimizer_steps')}"
                      if final else f"最终训练统计 pending；已收集完整rollout日志 {learned['full_rollouts_collected_logged']} 条，此日志在优化前写入，不能当已完成优化次数")
            lines.append(f"- {Path(run['run']).name}：{detail}。")
    lines += ["", "## 检测器证据", ""]
    for label, item in report["detector_evidence"].items():
        metrics = item.get("metrics") or {}
        lines.append(f"- {label}：AP50={fmt(metrics.get('AP50_101_point'))}，P@.15={fmt(metrics.get('precision_at_015'))}，R@.15={fmt(metrics.get('recall_at_015'))}。来源 `{item['path']}`，文件SHA256 `{item.get('sha256', 'pending')}`。")
    lines += ["", "固定最终epoch30模型和阈值；训练来源、预训练来源、模型SHA、CUDA设备与逐run来源SHA保存在comparison.json及原配置。检测AP来自独立静态图像标签，不能代替闭环策略成功率。", "", "## 记录核验", ""]
    for run in report["runs"]:
        counts = {key: sum(phase.get("video_verification_counts", {}).get(key, 0) for phase in run["phases"]) for key in ("passed", "failed", "pending_or_unfinished", "zero_action_reset", "not_scheduled", "disabled")}
        lines.append(f"- {Path(run['run']).name}：视频回合核验 {counts}；数据快照警告 {len(run.get('warnings', []))} 条。")
    lines += ["", "视频仅核验已终止且计划录制的回合：两MP4均960×540、10fps、动作数+1帧，首末帧可解码，首RGB/首带框/末带框PNG存在；真实活动或非零未完成视频标pending，不判失败。已完成PPO末尾的零动作额外重置单列zero_action_reset，无待处理transition。10fps是动作末帧展示节奏，并非动作持续时间录像。", ""]
    return "\n".join(lines)


def build_report(runs, videos=True, evidence_folder=None):
    values = [summarize_run(Path(run).resolve(), videos) for run in runs]
    for run in values:
        gpu = run.get("gpu_telemetry", {})
        overlaps = []
        for other in values:
            alt = other.get("gpu_telemetry", {})
            if other is not run and gpu.get("samples") and alt.get("samples") and max(gpu["unix_time_first"], alt["unix_time_first"]) <= min(gpu["unix_time_last"], alt["unix_time_last"]):
                overlaps.append(Path(other["run"]).name)
        gpu["overlapping_run_samples"] = overlaps
    return {"schema_version": 1, "generated_utc": datetime.now(timezone.utc).isoformat(),
            "read_only_inputs": True, "snapshot_note": "Each file read once; active files can grow between reads. Paired JSONL rows retain line ordinals.",
            "runs": values, "detector_evidence": detector_evidence(Path(evidence_folder or REPRO/"verification/pretrained-detectors-gpu"))}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--skip-video-checks", action="store_true")
    parser.add_argument("--detector-evidence", type=Path)
    args = parser.parse_args(argv)
    output = args.output.resolve()
    if any(output.is_relative_to(path.resolve()) for path in args.runs):
        parser.error("Output must be outside every input run; run logs are read-only")
    report = build_report(args.runs, not args.skip_video_checks, args.detector_evidence)
    output.mkdir(parents=True, exist_ok=True)
    for name, text in (("comparison.json", json.dumps(report, ensure_ascii=False, indent=2)), ("REPORT.md", render_markdown(report))):
        path = output/name
        temporary = output/(name+".tmp")
        temporary.write_text(text+"\n", encoding="utf-8")
        temporary.replace(path)
    print(json.dumps({"output": str(output), "runs": [{"run": r["run"], "status": r["status"], "warnings": len(r.get("warnings", []))} for r in report["runs"]]}, ensure_ascii=False, indent=2))
    return output


if __name__ == "__main__":
    main()
