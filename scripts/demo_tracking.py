"""Bounded image-feedback demo. Proportional control, not a learned policy."""
import argparse
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np

from ptz_env import ArenaClient
from ptz_overlay import draw_overlay


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--port", type=int, default=9001)
    parser.add_argument("--realtime", action="store_true",
                        help="Pace the preview to its 0.2-second simulation steps")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "runs" / "visual-demo")
    args = parser.parse_args(argv)
    if args.steps < 1:
        parser.error("--steps must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    arena = ArenaClient(port=args.port)
    writer = None
    boxed_writer = None
    history = []
    started = time.monotonic()
    try:
        m = arena.reset(args.seed)
        cv2.imwrite(str(args.output / "first.png"), cv2.cvtColor(arena.rgb, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(args.output / "first-boxed.png"),
                    cv2.cvtColor(draw_overlay(arena.rgb, m), cv2.COLOR_RGB2BGR))
        writer = cv2.VideoWriter(str(args.output / "tracking.mp4"),
                                cv2.VideoWriter_fourcc(*"mp4v"), 5, (960, 540))
        if not writer.isOpened():
            raise RuntimeError("Could not open MP4 writer")
        boxed_writer = cv2.VideoWriter(str(args.output / "tracking-boxed.mp4"),
                                      cv2.VideoWriter_fourcc(*"mp4v"), 5, (960, 540))
        if not boxed_writer.isOpened():
            raise RuntimeError("Could not open annotated MP4 writer")
        for step in range(args.steps):
            frame_started = time.monotonic()
            if m["visible"]:
                cx, cy = m["center"]
                tangent = math.tan(math.radians(arena.state["fov"] / 2))
                pan = np.clip(0.6 * math.degrees(math.atan(cx * tangent)) / 6, -1, 1)
                tilt = np.clip(-0.6 * math.degrees(math.atan(cy * tangent * 540 / 960)) / 4, -1, 1)
                zoom = np.clip(-0.5 * math.log(max(m["area"], 1e-6) / 0.025), -1, 1)
                if m["center_error"] > 0.6:
                    zoom = min(zoom, -0.5)
            else:
                # A simple search fallback. It cannot guarantee reacquisition.
                pan, tilt, zoom = 0.25 * math.sin(step / 15), 0, -1
            m = arena.step(pan, tilt, zoom)
            writer.write(cv2.cvtColor(arena.rgb, cv2.COLOR_RGB2BGR))
            overlay = draw_overlay(arena.rgb, m, arena.state["sim_time"])
            boxed_writer.write(cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            history.append(dict(step=step, sim_time=arena.state["sim_time"], **m))
            if (step + 1) % 25 == 0:
                print(f"step={step+1}/{args.steps} visible={m['visible']} error={m['center_error']:.3f}", flush=True)
            if args.realtime:
                time.sleep(max(0.0, 0.2 - (time.monotonic() - frame_started)))
        cv2.imwrite(str(args.output / "last.png"), cv2.cvtColor(arena.rgb, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(args.output / "last-boxed.png"),
                    cv2.cvtColor(draw_overlay(arena.rgb, m, arena.state["sim_time"]), cv2.COLOR_RGB2BGR))
    finally:
        if writer is not None:
            writer.release()
        if boxed_writer is not None:
            boxed_writer.release()
        arena.close()
    summary = dict(controller="HSV image feedback + proportional PTZ; no RL",
                   seed=args.seed, steps=len(history), sim_seconds=args.steps * 0.2,
                   wall_seconds=time.monotonic() - started,
                   visible_fraction=float(np.mean([m["visible"] for m in history])),
                   mean_center_error=float(np.mean([m["center_error"] for m in history])),
                   trace=history)
    (args.output / "demo-results.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "trace"}, indent=2))


if __name__ == "__main__":
    main()
