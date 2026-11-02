"""Offline evaluation only. None of these diagnostic values enters the agent."""
import math
from collections import deque
import numpy as np


def iou(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    inter = np.maximum(0, np.minimum(a[2:], b[2:]) - np.maximum(a[:2], b[:2])).prod()
    area_a = np.maximum(0, a[2:] - a[:2]).prod()
    area_b = np.maximum(0, b[2:] - b[:2]).prod()
    return float(inter / max(1e-12, area_a + area_b - inter))


def centered_box_iou(box):
    box = np.asarray(box, float)
    wh = box[2:] - box[:2]
    return iou(box, np.r_[.5 - wh / 2, .5 + wh / 2])


class EpisodeMetrics:
    def __init__(self, expected_count):
        self.expected_count = expected_count
        self.rows = []
        self.completed_ids = set()
        self.completion_matches = []
        self.lost_run = 0
        self.recovery_steps = []
        self.recovery_seconds = []
        self.lost_duration = 0
        self.max_lost_duration = 0
        self.gt_streak = deque(maxlen=10)

    def add(self, selected, diagnostic, duration, completion=None):
        # GT is used only for this offline identity/coverage audit.
        targets = diagnostic.get("targets", [])
        candidates = sorted([(iou(selected.bbox, target.get("gt_bbox_normalized", [0]*4)),
                              target["id"]) for target in targets if target.get("gt_in_frame")],
                             reverse=True) if selected else []
        match = candidates[0] if candidates else (0, None)
        ambiguous = len(candidates) > 1 and match[0] - candidates[1][0] < .05
        matched_id = match[1] if match[0] >= .3 and not ambiguous else None
        if duration > 0:
            self.gt_streak.append(matched_id if selected and selected.confidence >= .9 else None)
        if completion is not None:
            accepted = len(self.gt_streak) == 10 and matched_id is not None and all(
                value == matched_id for value in self.gt_streak)
            if accepted:
                self.completed_ids.add(matched_id)
            self.completion_matches.append({"best_iou": match[0], "target_id": matched_id,
                                            "ambiguous_iou_match": ambiguous,
                                            "streak_gt_ids": list(self.gt_streak),
                                            "accepted": accepted})
        visible = selected is not None
        if visible:
            box = np.asarray(selected.bbox)
            center = (box[:2] + box[2:]) - 1
            area = float(np.prod(box[2:] - box[:2]))
            error = float(np.linalg.norm(center))
            if self.lost_run and duration > 0:
                self.recovery_steps.append(self.lost_run)
                self.recovery_seconds.append(self.lost_duration)
            self.lost_run = 0
            self.lost_duration = 0
        elif duration > 0:
            area, error = None, None
            self.lost_run += 1
            self.lost_duration += duration
            self.max_lost_duration = max(self.max_lost_duration, self.lost_duration)
        else:
            area, error = None, None
        row = {"visible_detection": visible, "confidence": selected.confidence if visible else 0,
               "center_error_normalized": error, "target_area_ratio": area,
               "centered_box_iou": centered_box_iou(selected.bbox) if visible else 0,
               "duration_s": float(duration), "sim_time_s": diagnostic["sim_time"],
               "gt_any_in_frame": any(t.get("gt_in_frame", False) for t in targets),
               "gt_any_center_occluded": any(t.get("gt_center_occluded", False) for t in targets)}
        self.rows.append(row)
        return row

    def summary(self, task_success):
        active = self.rows[1:] if len(self.rows) > 1 else self.rows
        detected = [r for r in active if r["visible_detection"]]
        sim_time = sum(r["duration_s"] for r in active)
        weighted_visible = sum(r["duration_s"] for r in active if r["visible_detection"])
        mean = lambda key: float(np.mean([r[key] for r in detected])) if detected else None
        verified_success = bool(task_success and len(self.completed_ids) == self.expected_count)
        return {"success": verified_success, "termination_success_estimate": bool(task_success),
                "false_completion_estimate": bool(task_success and not verified_success),
                "target_count": self.expected_count, "completed_distinct_count": len(self.completed_ids),
                "action_steps": len(active),
                "TSR_detection_proxy": len(detected) / len(active) if active else 0,
                "visible_time_fraction_sampled": weighted_visible / sim_time if sim_time else 0,
                "sim_time_s": sim_time, "mean_duration_s": sim_time / len(active) if active else 0,
                "mean_center_error": mean("center_error_normalized"),
                "mean_target_area_ratio": mean("target_area_ratio"),
                "mean_centered_box_iou": mean("centered_box_iou"),
                "IC": self.rows[0]["confidence"] if self.rows else 0,
                "EC": self.rows[-1]["confidence"] if self.rows else 0,
                "TC_offline_gt_matched": len(self.completed_ids) / self.expected_count,
                "completed_gt_ids": sorted(self.completed_ids),
                "completion_matches": self.completion_matches,
                "recovery_steps": self.recovery_steps, "recovery_seconds": self.recovery_seconds,
                "unrecovered_lost_steps": self.lost_run,
                "unrecovered_lost_seconds": self.lost_duration,
                "longest_selected_unavailable_seconds": self.max_lost_duration,
                "selected_unavailable_events": len(self.recovery_steps) + int(self.lost_run > 0),
                "note": "TSR uses selected detector availability; time visibility sampled at action ends. TC requires 10 action frames with confidence>=0.9 matching the same offline full-car GT at IoU>=0.3; ambiguous top-two IoU gap<0.05 rejected. Policy never reads IDs."}
