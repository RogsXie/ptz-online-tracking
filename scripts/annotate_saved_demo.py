"""Overlay saved detections on their matching raw frames; no UE rerun needed."""
import json
from pathlib import Path

import cv2

from ptz_env import image_measurement
from ptz_overlay import draw_overlay


def main():
    folder = Path(__file__).resolve().parents[1] / "runs" / "visual-demo"
    data = json.loads((folder / "demo-results.json").read_text(encoding="utf-8"))
    capture = cv2.VideoCapture(str(folder / "tracking.mp4"))
    writer = cv2.VideoWriter(str(folder / "tracking-boxed.mp4"),
                            cv2.VideoWriter_fourcc(*"mp4v"), 5, (960, 540))
    if not capture.isOpened() or not writer.isOpened():
        raise RuntimeError("Could not open raw video or annotated output")
    count = 0
    try:
        for row in data["trace"]:
            ok, bgr = capture.read()
            if not ok:
                raise RuntimeError("Video ended before its detection trace")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            overlay = draw_overlay(rgb, row, row["sim_time"])
            writer.write(cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            count += 1
        extra, _ = capture.read()
        if extra:
            raise RuntimeError("Video has frames not covered by the detection trace")
    finally:
        capture.release()
        writer.release()
    for filename in ["first", "last"]:
        rgb = cv2.cvtColor(cv2.imread(str(folder / f"{filename}.png")), cv2.COLOR_BGR2RGB)
        measurement = image_measurement(rgb)
        overlay = draw_overlay(rgb, measurement, 0 if filename == "first" else data["sim_seconds"])
        cv2.imwrite(str(folder / f"{filename}-boxed.png"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    print(f"Annotated {count} matching frames: {folder / 'tracking-boxed.mp4'}")


if __name__ == "__main__":
    main()
