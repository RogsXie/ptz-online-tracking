"""Deterministic UE scenarios for the preregistered learned-control curriculum.

These are new experiments. Seeds used to diagnose the failed 320-episode policy
are excluded. Camera reset is configured independently of the realized target;
no simulator location, GT box or target identity enters a policy observation.
"""
from copy import deepcopy
from pathlib import Path
import hashlib
import json
import random

HERE = Path(__file__).resolve().parent
FOCUSED = HERE.parent / "focused_scenarios"
ALIASES = {"stage1": "static_center", "stage2": "static_zoom", "stage3": "uniform_motion",
           "stage4": "variable_speed", "stage5": "turning", "stage6": "occlusion", "stage7": "low_light"}


def load_protocol():
    return json.loads((HERE / "protocol.json").read_text(encoding="utf-8"))


def stage_spec(stage):
    stage = ALIASES.get(str(stage), str(stage))
    for item in load_protocol()["stages"]:
        if item["id"] == stage:
            return item
    raise KeyError(f"Unknown control stage: {stage}")


def _partition(spec, split):
    if split not in ("train", "validation", "final_test"):
        raise KeyError(split)
    index = spec["number"]
    base = {"train": 200_000, "validation": 500_000, "final_test": 1_000_000}[split] + index * 10_000
    count = 10_000 if split == "train" else 40
    return base, count


def cases(stage, split, count=None, offset=0, allow_final_test=False):
    spec = stage_spec(stage)
    if split == "final_test" and not allow_final_test:
        raise ValueError("Final-test seeds are reserved. Complete all validation gates before explicit final release.")
    base, capacity = _partition(spec, split)
    count = count if count is not None else (256 if split == "train" else capacity)
    if not isinstance(count, int) or not isinstance(offset, int) or count <= 0 or offset < 0 or count + offset > capacity:
        raise ValueError("Case count/offset must stay inside the disjoint seed partition")
    return [{"stage": spec["id"], "scenario_name": f"control_{spec['id']}", "split": split,
             "case_index": offset + index, "seed": base + offset + index} for index in range(count)]


def _base_config():
    # Reuse the verified UE JSON schema and environment, without editing its
    # historical scenario or importing split-specific target data.
    return json.loads((FOCUSED / "focus_far.json").read_text(encoding="utf-8-sig"))


def resolve(stage, split, seed, allow_final_test=False):
    spec = stage_spec(stage)
    if split == "final_test" and not allow_final_test:
        raise ValueError("Final-test materialization is blocked until explicit final release")
    base, capacity = _partition(spec, split)
    if not isinstance(seed, int) or not base <= seed < base + capacity:
        raise ValueError("Seed is outside this stage/split partition")
    rng = random.Random(seed)
    ordinal = seed - base
    # Four balanced initial offset quadrants in each forty-seed validation set.
    yaw_sign, pitch_sign = (-1, -1) if ordinal % 4 == 0 else (-1, 1) if ordinal % 4 == 1 else (1, -1) if ordinal % 4 == 2 else (1, 1)
    config = deepcopy(_base_config())
    config.update(name=f"control_{spec['id']}", split=split, focus_category=spec["category"],
                  focus_level="gated", difficulty=spec["number"], task="learned_rgb_control_v2")
    config["features"] = ["single_vehicle", "rgb_closed_loop", "fixed_seeded_camera", "no_gt_aim", spec["id"]]
    camera = config["camera"]
    camera.update(location_cm=[-2200, 0, 900], rotation_deg=[-6., 0., 0.], rotation_jitter_deg=[0., 0., 0.],
                  pan_deg_s=8., tilt_deg_s=8., zoom_fov_deg_s=8., fov_deg=60., fov_min_deg=8., fov_max_deg=60.,
                  yaw_limits_deg=[-40., 40.], pitch_limits_deg=[-25., 20.], legacy_target_center_reset=False)
    config["randomization"] = {"position_jitter_cm": 0., "speed_scale_range": [1., 1.]}
    car = config["targets"][0]
    car.update(id="car_0", color_rgb=[.75, .025, .012], scale=1.8)
    car["path"] = {"type": "static", "origin_cm": [rng.uniform(4800, 5200), rng.uniform(-25, 25), 8.]}
    config["occluders"] = []
    if spec["id"] == "static_center":
        # Local foundation only: bounded mechanics keep a small-error task from
        # degenerating into exploration of blank sky. Fixed limits are not GT aim.
        camera.update(fov_deg=60., fov_min_deg=60., fov_max_deg=60., zoom_fov_deg_s=8.,
                      pitch_limits_deg=[-12., 0.], yaw_limits_deg=[-10., 10.])
        camera["rotation_deg"] = [-6. + pitch_sign * rng.uniform(1., 3.2), yaw_sign * rng.uniform(1.2, 4.), 0.]
    elif spec["id"] == "static_zoom":
        camera["rotation_deg"] = [-6. + pitch_sign * rng.uniform(.8, 2.8), yaw_sign * rng.uniform(1., 3.5), 0.]
        camera["fov_deg"] = rng.uniform(45., 60.)
        car["scale"] = rng.uniform(1.4, 1.8)
        car["path"]["origin_cm"] = [rng.uniform(4800, 8500), rng.uniform(-80, 80), 8.]
    else:
        camera["rotation_deg"] = [-6. + pitch_sign * rng.uniform(.3, 1.), yaw_sign * rng.uniform(.3, 1.), 0.]
        camera["fov_deg"] = rng.uniform(20., 24.)
        car["scale"] = rng.uniform(1.3, 1.6)
        depth = rng.uniform(4500, 5100)
        direction = -1 if ordinal % 2 == 0 else 1
        speed = rng.uniform(75., 110.)
        car["path"] = {"type": "line", "origin_cm": [depth, rng.uniform(-150, 150), 8.],
                       "velocity_cm_s": [0., direction * speed, 0.]}
        if spec["id"] in ("variable_speed", "occlusion", "low_light"):
            car["path"]["velocity_cm_s"] = [0., direction * rng.uniform(130., 165.), 0.]
            car["path"]["speed_profile"] = [[0, .5], [3, 1.4], [6, .25], [9, 1.8], [12, 0],
                                             [13, -.7], [17, -1.2], [22, .7], [27, .3], [32, .5]]
        if spec["id"] == "turning":
            # Closed continuous rectangle, no line wrap or positional teleport.
            car["path"] = {"type": "waypoints", "origin_cm": [depth, 0., 8.],
                           "waypoints_cm": [[depth, 0., 8.], [depth, direction * 900., 8.],
                                            [depth + 1000., direction * 900., 8.], [depth + 1000., 0., 8.]],
                           "speed_cm_s": rng.uniform(165., 210.)}
        if spec["id"] == "occlusion":
            # The ray to the moving car crosses this opaque wall in the first
            # half of the episode. Actual RGB hide/reappearance is checked in UE.
            config["occluders"] = [{"kind": "wall", "location_cm": [1800., direction * 520., 220.],
                                     "size_cm": [100., 280., 440.], "rotation_deg": [0., 0., 0.],
                                     "color_rgb": [.38, .40, .42]}]
        if spec["id"] == "low_light":
            config["environment"].update(directional_lux=.25, sky_intensity=.08, exposure=.55)
    protocol_bytes = (HERE / "protocol.json").read_bytes()
    config["control_protocol"] = {"name": "gated_rgb_control_v2", "stage": spec["id"],
        "scene_version": "static-center-local-limits-v2" if spec["id"] == "static_center" else "initial-preregistered-v1",
        "number": spec["number"], "seed": seed, "allowed_actions": spec["allowed_actions"],
        "fixed_fov": spec["id"] == "static_center", "target_area_ratio": .04,
        "protocol_sha256": hashlib.sha256(protocol_bytes).hexdigest(),
        "scope": "new calibrated control experiment, not claimed author/paper-equivalent",
        "initialization": "fixed nominal camera with independent seeded offsets; never reads target/GT to aim",
        "detector": "frozen yolo26s_ue_gpu30.pt; threshold/image size/weights unchanged",
        "reid": "existing verified frozen FastReID configuration"}
    return config


def materialize(stage, split, seed, output_dir, allow_final_test=False):
    config = resolve(stage, split, seed, allow_final_test)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{config['name']}-{split}-seed{seed}.json"
    text = json.dumps(config, indent=2, ensure_ascii=False) + "\n"
    if path.exists() and path.read_text(encoding="utf-8-sig") != text:
        raise FileExistsError(f"Do not overwrite a different materialized scenario: {path}")
    if not path.exists():
        path.write_text(text, encoding="utf-8")
    return path
