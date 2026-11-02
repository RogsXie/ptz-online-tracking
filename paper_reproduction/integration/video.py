"""Store untouched RGB and a separately labelled diagnostic presentation."""
from pathlib import Path
import cv2
import numpy as np


class VideoLog:
    def __init__(self, output, label, fps=10):
        self.output, self.label, self.fps = Path(output), label, fps
        self.raw = self.annotated = None
        self.frames = 0
        self.last_shown = None

    def write(self, rgb, detections, selected, text=""):
        h, w = rgb.shape[:2]
        if self.raw is None:
            codec = cv2.VideoWriter_fourcc(*"mp4v")
            self.raw = cv2.VideoWriter(str(self.output / "sensor-rgb.mp4"), codec, self.fps, (w, h))
            self.annotated = cv2.VideoWriter(str(self.output / "policy-boxed.mp4"), codec, self.fps, (w, h))
            if not self.raw.isOpened() or not self.annotated.isOpened():
                raise RuntimeError("Video encoder unavailable")
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        self.raw.write(bgr)
        shown = bgr.copy()
        for detection in detections:
            x1, y1, x2, y2 = (np.array(detection.bbox) * [w, h, w, h]).astype(int)
            color = (40, 255, 80) if detection is selected else (120, 160, 220)
            cv2.rectangle(shown, (x1, y1), (x2, y2), color, 2)
            cv2.putText(shown, f"car {detection.confidence:.3f}", (x1, max(65, y1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, .45, color, 1)
        cv2.drawMarker(shown, (w // 2, h // 2), (255, 160, 60), cv2.MARKER_CROSS, 18, 1)
        cv2.rectangle(shown, (0, 0), (w, 52), (15, 20, 25), -1)
        cv2.putText(shown, self.label, (10, 21), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1)
        cv2.putText(shown, text[:140], (10, 43), cv2.FONT_HERSHEY_SIMPLEX, .42, (220, 220, 220), 1)
        self.annotated.write(shown)
        if self.frames == 0:
            cv2.imwrite(str(self.output / "first-rgb.png"), bgr)
            cv2.imwrite(str(self.output / "first-boxed.png"), shown)
        self.last_shown = shown
        self.frames += 1

    def close(self):
        if self.raw:
            self.raw.release()
            self.annotated.release()
        if self.last_shown is not None:
            cv2.imwrite(str(self.output / "last-boxed.png"), self.last_shown)
