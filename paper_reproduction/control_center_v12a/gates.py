"""Evaluate the frozen, preregistered RGB-control validation gates.

No policy, detector or simulator calls occur here. `matched` is offline identity
evidence and must never be included in the learner's state or reward. Timing
weights are action-end samples; these metrics are not continuous pixel truth.
"""
import math
import hashlib
from pathlib import Path

from .scenes import cases, load_protocol, stage_spec
from .scenes import layout


def _finite(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def _weighted_quantile(values, quantile):
    values = sorted(values)
    total = sum(weight for _, weight in values)
    if total <= 0:
        return None
    cursor = 0.
    for value, weight in values:
        cursor += weight
        if cursor >= quantile * total:
            return value
    return values[-1][0]


def _window(rows, start, end, scale_target=.04):
    errors, scale_errors, areas = [], [], []
    total = visible = matched = in_center = 0.
    for before, row in zip(rows, rows[1:]):
        weight = max(0., min(end, row["sim_time_s"]) - max(start, before["sim_time_s"]))
        if weight <= 0:
            continue
        total += weight
        valid = bool(row.get("visible")) and _finite(row.get("center_error"))
        visible += weight * valid
        matched += weight * (valid and bool(row.get("matched")))
        if valid:
            error = float(row["center_error"])
            errors.append((error, weight))
            in_center += weight * (error <= .05)
            area = row.get("area_ratio")
            if _finite(area) and area > 0:
                areas.append((float(area), weight))
                scale_errors.append((abs(math.log(area / scale_target)), weight))
    weighted_mean = lambda pairs: (sum(value * weight for value, weight in pairs) / sum(weight for _, weight in pairs)) if pairs else None
    return {"start_s": start, "end_s": end, "duration_s": total,
        "center_mean": weighted_mean(errors), "center_p95": _weighted_quantile(errors, .95),
        "center_fraction_within_0_05": in_center / total if total else 0.,
        "selected_availability": visible / total if total else 0.,
        "matched_availability": matched / total if total else 0.,
        "scale_log_error_mean": weighted_mean(scale_errors),
        "area_p05": _weighted_quantile(areas, .05), "area_p95": _weighted_quantile(areas, .95)}


def _window_passed(metrics, spec):
    checks = {
        "minimum_stable_hold": metrics["duration_s"] >= spec["stable_hold_seconds"] - 1e-6,
        "center_mean": _finite(metrics["center_mean"]) and metrics["center_mean"] <= spec["stable_center_mean_max"],
        "center_p95": _finite(metrics["center_p95"]) and metrics["center_p95"] <= spec["stable_center_p95_max"],
        "center_band_fraction": metrics["center_fraction_within_0_05"] >= spec["stable_center_fraction_within_0_05_min"],
        "selected_availability": metrics["selected_availability"] >= spec["selected_availability_min"],
        "correct_target_availability": metrics["matched_availability"] >= spec["offline_matched_availability_min"]}
    if spec["scale_required"]:
        lo, hi = spec["stable_area_range"]
        checks.update(scale_log_error=_finite(metrics["scale_log_error_mean"]) and metrics["scale_log_error_mean"] <= spec["stable_scale_log_error_mean_max"],
                      area_lower=_finite(metrics["area_p05"]) and metrics["area_p05"] >= lo,
                      area_upper=_finite(metrics["area_p95"]) and metrics["area_p95"] <= hi)
    return all(checks.values()), checks


def _motion_metrics(rows):
    previous = {}
    reversals, unwarranted, overshoots = [], [], []
    magnitudes, zoom_magnitudes = [], []
    for index in range(1, len(rows)):
        before, row = rows[index - 1], rows[index]
        action = int(row.get("action_type", 0))
        if action not in (1, 2, 3, 4, 5, 6):
            continue
        axis, sign = (("x", 1 if action == 3 else -1) if action in (3, 4)
                      else ("y", -1 if action == 2 else 1) if action in (1, 2)
                      else ("zoom", 1 if action == 5 else -1))
        started = before["sim_time_s"]
        (magnitudes if axis != "zoom" else zoom_magnitudes).append(8. * (row["sim_time_s"] - started))
        source_error, final_error = before.get(axis), row.get(axis)
        if _finite(source_error) and _finite(final_error) and source_error * final_error < 0 and min(abs(source_error), abs(final_error)) >= .03:
            overshoots.append({"row": index, "axis": axis, "before": source_error, "after": final_error})
        old = previous.get(axis)
        if old and old["sign"] != sign:
            event = {"row": index, "axis": axis, "seconds_since_previous_axis_command": started - old["start"]}
            reversals.append(event)
            if (started - old["start"] <= .5 + 1e-9 and _finite(source_error) and _finite(final_error)
                    and abs(source_error) <= .03 and abs(final_error) - abs(source_error) >= .02):
                unwarranted.append(event | {"before_error": source_error, "after_error": final_error})
        previous[axis] = {"sign": sign, "start": started}
    return {"direction_reversals": len(reversals), "rapid_unwarranted_reversals": len(unwarranted),
            "direction_reversals_by_axis": {axis: sum(event["axis"] == axis for event in reversals) for axis in ("x", "y", "zoom")},
            "overshoots": len(overshoots), "maximum_angular_command_deg": max(magnitudes, default=0.),
            "maximum_fov_command_deg": max(zoom_magnitudes, default=0.),
            "reversal_events": reversals, "rapid_unwarranted_reversal_events": unwarranted,
            "overshoot_events": overshoots}


def assess_episode(stage, rows):
    """Return an explicit per-case gate, independent of training reward.

    Required row fields: sim_time_s, center_error, x, y, area_ratio, visible,
    matched, action_type, duration_s, fov. Initial row has t=0/duration=0.
    Caller adds the known stage/seed to the returned dictionary for aggregation.
    """
    protocol = load_protocol()
    raw_spec = stage_spec(stage)
    spec = protocol["default_episode_gate"] | raw_spec
    common = {"stage": raw_spec["id"], "metric_version": "gated_rgb_control_v2_1",
              "protocol_sha256": hashlib.sha256((Path(__file__).parent / "protocol.json").read_bytes()).hexdigest()}
    if len(rows) < 2:
        return common | {"passed": False, "failures": ["At least an initial observation and one action are required"], "checks": {"rows": False}}
    times = [row.get("sim_time_s") for row in rows]
    valid_times = all(_finite(value) for value in times) and abs(times[0]) <= 1e-6 and all(b > a for a, b in zip(times, times[1:]))
    if not valid_times:
        return common | {"passed": False, "failures": ["Invalid/nonmonotonic simulation times"], "checks": {"time_grid": False}}
    end = times[-1]
    valid_durations = all(_finite(row.get("duration_s")) and .04 - 1e-6 <= row["duration_s"] <= .25 + 1e-6
                          and abs(row["duration_s"] - (row["sim_time_s"] - before["sim_time_s"])) <= 1e-5
                          for before, row in zip(rows, rows[1:]))
    finite_fov = all(_finite(row.get("fov")) for row in rows)
    rgb_fields_valid = all((not row.get("visible")) or (
        all(_finite(row.get(key)) for key in ("center_error", "x", "y", "area_ratio"))
        and row["center_error"] >= 0 and 0 < row["area_ratio"] <= 1
        and abs(row["center_error"] - math.hypot(row["x"], row["y"])) <= 1e-5)
        for row in rows)
    fov_change = max(row["fov"] for row in rows) - min(row["fov"] for row in rows) if finite_fov else float("inf")
    loss_run = longest_loss = 0.
    for before, row in zip(rows, rows[1:]):
        loss_run = loss_run + row["sim_time_s"] - before["sim_time_s"] if not row.get("visible") else 0.
        longest_loss = max(longest_loss, loss_run)
    full = _window(rows, 0., end)
    terminal = _window(rows, max(0., end - spec["stable_hold_seconds"]), end)
    terminal_passed, terminal_checks = _window_passed(terminal, spec)
    settling = None
    stable = None
    for row in rows:
        start = row["sim_time_s"]
        if start > spec["settling_seconds_max"] + 1e-9:
            break
        if not row.get("visible") or not row.get("matched") or not _finite(row.get("center_error")) or row["center_error"] > spec["stable_center_p95_max"]:
            continue
        window = _window(rows, start, end)
        passed, _ = _window_passed(window, spec)
        if passed:
            settling, stable = start, window
            break
    motion = _motion_metrics(rows)
    motion["direction_reversals_per_10s"] = motion["direction_reversals"] / end * 10.
    checks = {"time_grid": valid_times, "physical_duration_bounds_and_timing": valid_durations,
              "complete_horizon": end >= spec["horizon_seconds"] - 1e-6,
              "horizon_overshoot": end <= spec["horizon_seconds"] + .25 + 1e-6,
              "decision_budget": len(rows) - 1 <= spec["max_decisions"],
              "allowed_actions": all(int(row.get("action_type", -1)) in spec["allowed_actions"] for row in rows[1:]),
              "finite_fov": finite_fov, "initial_rgb_zone_confirmed":rows[0].get("initial_zone_confirmed") is True, "settle_before_limit_and_sustain": settling is not None,
              "rgb_fields_and_raw_offset_scale": rgb_fields_valid,
              "terminal_correct_target": bool(rows[-1].get("visible")) and bool(rows[-1].get("matched")),
              "terminal_stability": terminal_passed,
              "full_selected_availability": full["selected_availability"] >= spec["selected_availability_min"],
              "full_matched_availability": full["matched_availability"] >= spec["offline_matched_availability_min"],
              "maximum_unintended_loss": longest_loss <= spec["unintended_loss_run_seconds_max"] + 1e-6,
              "rapid_unwarranted_reversals": motion["rapid_unwarranted_reversals"] <= spec["rapid_unwarranted_reversals_max"]}
    if "fov_change_deg_max" in spec:
        checks["fixed_fov"] = fov_change <= spec["fov_change_deg_max"]
    if "direction_reversals_per_10s_max" in spec:
        checks["total_direction_reversal_frequency"] = motion["direction_reversals_per_10s"] <= spec["direction_reversals_per_10s_max"] + 1e-9
    # Actual RGB occlusion events must be audited separately before this stage
    # can pass; do not infer them from a center-ray flag or silently skip loss.
    if raw_spec["id"] == "occlusion":
        checks["occlusion_events_and_recovery_evidence"] = False
    passed = all(checks.values())
    return common | {"passed": passed, "checks": checks, "failures": [key for key, value in checks.items() if not value],
        "settling_time_s": settling, "full_duration_s": end, "action_steps": len(rows) - 1,
        "initial_center_error": rows[0].get("center_error"),
        "terminal_center_mean": terminal["center_mean"], "terminal_center_p95": terminal["center_p95"],
        "selected_availability": full["selected_availability"], "matched_availability": full["matched_availability"],
        "longest_loss_run_seconds": longest_loss, "fov_change_deg": fov_change,
        "terminal_window": terminal, "terminal_checks": terminal_checks, "stable_suffix": stable, "full_metrics": full,
        **motion}


def gate_aggregate(stage, episodes):
    """All120 declared cases plus quadrant/distance/scale/intersection gates."""
    protocol = load_protocol()
    expected = {c['seed'] for c in cases(stage, 'validation')}
    seeds = [r.get('seed') for r in episodes]
    digest = hashlib.sha256((Path(__file__).parent/'protocol.json').read_bytes()).hexdigest()
    groups = {'quadrant': {}, 'layer': {}, 'scale_group': {}, 'group': {}}
    for seed in sorted(expected):
        item = layout(seed, 'validation')
        for key in groups:
            groups[key].setdefault(str(item[key]), {'passed': 0, 'total': 0})['total'] += 1
    for row in episodes:
        if row.get('seed') in expected and row.get('passed') is True:
            item = layout(row['seed'], 'validation')
            for key in groups:
                groups[key][str(item[key])]['passed'] += 1
    successes = sum(row.get('passed') is True for row in episodes)
    limits = protocol['validation_group_gates']
    checks = {
        'complete_expected_validation': set(seeds) == expected and len(seeds) == 120,
        'no_duplicate_seeds': len(seeds) == len(set(seeds)),
        'stage_matches': all(r.get('stage') == stage for r in episodes),
        'protocol_matches': all(r.get('protocol_sha256') == digest for r in episodes),
        'overall_95_percent': len(episodes) == 120 and successes >= 114,
        'every_quadrant': min(g['passed'] for g in groups['quadrant'].values()) >= limits['quadrant_min'],
        'every_distance': min(g['passed'] for g in groups['layer'].values()) >= limits['layer_min'],
        'every_scale': min(g['passed'] for g in groups['scale_group'].values()) >= limits['scale_min'],
        'every_intersection': min(g['passed'] for g in groups['group'].values()) >= limits['cell_min'],
        'all_near_center_hold': groups['layer']['0']['passed'] == limits['near_center_min'],
    }
    return dict(stage=stage, passed=all(checks.values()), checks=checks,
                failed_checks=[k for k, v in checks.items() if not v],
                passed_episodes=successes, episodes=len(episodes), expected_episodes=120,
                groups=groups, protocol_sha256=digest, later_stage_promotion=False)


