"""Integration checks against the running UE arena (no mocked simulator)."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from ptz_env import ArenaClient


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=9001)
    args = parser.parse_args()
    output = Path(__file__).resolve().parents[1] / "verification"
    output.mkdir(parents=True, exist_ok=True)
    arena = ArenaClient(port=args.port)
    checks = {}
    try:
        first = arena.reset(42)
        start = arena.state.copy()
        cv2.imwrite(str(output / "vehicle-first.png"), cv2.cvtColor(arena.rgb, cv2.COLOR_RGB2BGR))
        checks["image_resolution"] = arena.rgb.shape == (540, 960, 3)
        checks["visual_target_detected"] = bool(first["visible"])
        arena.step(0, 0, 0)
        checks["target_moves"] = np.linalg.norm(np.array(arena.state["target_location"]) - start["target_location"]) > 1
        checks["synchronous_time"] = abs(arena.state["sim_time"] - 0.2) < 1e-5
        arena.reset(42)
        repeated = arena.state.copy()
        checks["seed_reproducible"] = all(np.allclose(start[k], repeated[k]) for k in
            ["target_location", "rotation", "fov", "sim_time"])
        arena.step(1, 0, 0)
        checks["pan_control"] = abs(arena.state["rotation"][1] - start["rotation"][1] - 6) < 1e-3
        checks["camera_position_fixed"] = arena.state["camera_location"] == start["camera_location"]
        arena.reset(42)
        arena.step(0, -1, 0)
        checks["tilt_control"] = abs(arena.state["rotation"][0] - start["rotation"][0] + 4) < 1e-3
        arena.reset(42)
        arena.step(0, 0, 1)
        checks["zoom_control"] = abs(arena.state["fov"] - 54) < 1e-3
        cv2.imwrite(str(output / "vehicle-zoom.png"), cv2.cvtColor(arena.rgb, cv2.COLOR_RGB2BGR))
        arena.step(0, 0, 0, duration=5)
        checks["duration_limit"] = abs(arena.state["sim_time"] - 1.2) < 1e-5
        checks = {key: bool(value) for key, value in checks.items()}
        result = dict(checks=checks, ok=all(checks.values()), first_image_measurement=first,
                      endpoint=f"127.0.0.1:{args.port}",
                      note="Simulator geometry used only here to validate motion and seed reproducibility")
        (output / "tracking-scene-verification.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2))
        if not result["ok"]:
            raise SystemExit("Tracking scene verification failed")
    finally:
        arena.reset(42)
        arena.close()


if __name__ == "__main__":
    main()
