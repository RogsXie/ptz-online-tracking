"""Sensor-only detections. No simulator identity or location is accepted."""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Detection:
    """One detector result: normalized xyxy, actual detector probability/score."""

    bbox: tuple[float, float, float, float]
    confidence: float
    class_id: int = 0
    label: str = "car"

    def __post_init__(self):
        box = tuple(float(v) for v in self.bbox)
        if len(box) != 4 or not all(math.isfinite(v) and 0 <= v <= 1 for v in box):
            raise ValueError("Detection bbox must be four finite normalized xyxy coordinates")
        if box[2] <= box[0] or box[3] <= box[1]:
            raise ValueError("Detection bbox must have positive area")
        if not math.isfinite(self.confidence) or not 0 <= self.confidence <= 1:
            raise ValueError("Detection confidence must be a detector score in [0,1]")
        object.__setattr__(self, "bbox", box)
        object.__setattr__(self, "confidence", float(self.confidence))

    @property
    def center(self):
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) * 0.5, (y1 + y2) * 0.5)

    @property
    def area(self):
        x1, y1, x2, y2 = self.bbox
        return (x2 - x1) * (y2 - y1)

    @property
    def center_inside(self):
        x1, y1, x2, y2 = self.bbox
        return x1 <= .5 <= x2 and y1 <= .5 <= y2

    def as_dict(self):
        return {"bbox": list(self.bbox), "confidence": self.confidence,
                "class_id": self.class_id, "label": self.label}
