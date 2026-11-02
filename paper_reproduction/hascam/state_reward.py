"""Table 1 states and two explicit readings of the conflicting VPE reward.

paper_literal is exactly printed Eqs 5--11 (including the adverse position/
in-box signs); author_code uses the costs executed by the official environment.
Both are exposed so an experiment cannot silently change the objective.
"""
import math
import numpy as np
from .detection import Detection


LOST_STATE = np.array([-2., -2., -1., 0., 0., 0., 0., 0.], dtype=np.float32)


class StateBuilder:
    """Pure detector-to-state transform; Table 1, not source's area*100 bug."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.previous_center = None

    def build(self, detection: Detection | None, last_action=(0, 0.), reset=False):
        if reset:
            self.reset()
        if detection is None:
            self.previous_center = None
            return LOST_STATE.copy()
        cx, cy = detection.center
        x, y = 2 * cx - 1, 2 * cy - 1
        trend_x, trend_y = (0., 0.) if self.previous_center is None else (
            2 * (cx - self.previous_center[0]), 2 * (cy - self.previous_center[1]))
        action, duration = last_action
        if reset:
            action, duration, trend_x, trend_y = 0, 0., 0., 0.
        if not 0 <= int(action) < 7 or not math.isfinite(duration) or duration < 0:
            raise ValueError("Invalid preceding action")
        self.previous_center = (cx, cy)
        return np.array([x, y, detection.area, detection.confidence,
                         int(action) / 7., duration, trend_x, trend_y], dtype=np.float32)


class VPEReward:
    def __init__(self, mode="author_code"):
        if mode not in {"author_code", "paper_literal"}:
            raise ValueError("reward mode must be author_code or paper_literal")
        self.mode = mode
        self.reset()

    def components(self, detection):
        if detection is None:
            return {"confidence": 1., "position": 1., "area": 1., "inbox": 1.}
        x, y = (2 * c - 1 for c in detection.center)
        a = detection.area
        confidence = 1 - detection.confidence
        position = .5 * (x*x + y*y)
        if a >= .05:
            area = 1 - math.exp(-(a-.05) / .1)
        else:
            area = ((a-.05) / .05) ** 2
        inbox = 0. if detection.center_inside else 1.
        if self.mode == "paper_literal":
            position = 1 - position
            if a < .05:
                area = 1 - area
            inbox = 1 - inbox
        return {"confidence": confidence, "position": position,
                "area": area, "inbox": inbox}

    def reset(self, initial_detection=None):
        # Source resets previous penalties to zero. Paper says improvement only
        # after a captured preceding frame. We require actual captured history.
        self.previous = self.components(initial_detection) if initial_detection is not None else None

    def step(self, detection: Detection | None):
        c = self.components(detection)
        bonus = 0.
        if detection is not None and self.previous is not None:
            old = self.previous
            if old["position"] - c["position"] > .2:
                bonus += .5
            if old["area"] - c["area"] > .15:
                bonus += .5
            if old["confidence"] - c["confidence"] > .1:
                bonus += .5
            # Printed Eq11 compares p(inBbox) increases. Actual author reward
            # uses a signed in-box penalty but its old!=-1 guard blocks this
            # bonus. Keep those two readings explicit.
            if self.mode == "paper_literal" and c["inbox"] > old["inbox"]:
                bonus += .5
            if self.mode == "author_code":
                # Exact guards in source prevent bonuses when old cost==1.
                bonus = sum(.5 for key, threshold in (("position", .2), ("area", .15), ("confidence", .1))
                            if old[key] != 1 and old[key] - c[key] > threshold)
        self.previous = c if detection is not None else None
        return {"reward": -sum(c.values()) + bonus if detection is not None else -4.,
                "components": c, "improvement_bonus": bonus, "mode": self.mode}

    @staticmethod
    def terminal_adjustment(success: bool, steps: int) -> float:
        """Eq12: failure has priority at the 128-step boundary."""
        return 40. if success and steps < 128 else -40.
