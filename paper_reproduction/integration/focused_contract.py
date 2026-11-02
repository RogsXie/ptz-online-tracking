"""Prevent silent task/detector changes when loading focused HAS-CAM weights."""
from pathlib import Path


def default_policy_path(repro, kind="yolo26s_ue", profile="quality"):
    if kind not in {"yolo26s_ue", "yolo26s", "yolo11s", "ssd"} or profile not in {"quality", "paper"}:
        raise ValueError("Unknown focused detector or task profile")
    backend = "" if kind == "yolo26s_ue" else "_" + kind
    task = "" if profile == "quality" else "_paper"
    return Path(repro)/"weights"/f"hascam_focused{backend}{task}_gpu.pt"


def validate_hascam_frontend(metadata, detector, reid, profile, reward_mode="author_code"):
    if metadata.get("task_profile") != profile:
        raise ValueError("Checkpoint task profile differs or is legacy; use its original experiment entry")
    if metadata.get("pdqn", {}).get("reward_mode") != reward_mode:
        raise ValueError("Checkpoint reward mode differs from the focused environment")
    specifications = (
        ("detector", detector, "sha256", ("kind", "sha256", "backend", "threshold", "imgsz", "half", "target_class_ids", "target_labels")),
        ("reid", reid, "weights_sha256", ("name", "weights_sha256", "feature_dim_actual", "input_size", "input_color")),
    )
    for name, current, hash_key, keys in specifications:
        original = metadata.get(name, {})
        if not original.get(hash_key) or not current.get(hash_key):
            raise ValueError(f"Missing {name} weight hash")
        for key in keys:
            if original.get(key) != current.get(key):
                raise ValueError(f"Checkpoint {name} mismatch for {key}")
    if profile == "quality":
        predicate = metadata.get("quality_task_extension", {})
        for key, value in {"confidence_min": .9, "center_error_max": .15,
                           "area_range": [.02, .08], "consecutive_actions": 10}.items():
            if predicate.get(key) != value:
                raise ValueError(f"Checkpoint completion predicate differs for {key}")
    return {"task_profile_matches": True, "detector_weights_and_settings_match": True,
            "reid_weights_and_settings_match": True, "reward_and_completion_match": True}
