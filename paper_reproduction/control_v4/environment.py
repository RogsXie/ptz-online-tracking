"""Static-only current-image state with observable finite-horizon time.

No action is selected here. History removal is limited to a stationary target,
fixed-FOV task; this representation is not qualified for moving targets.
"""
import math
import numpy as np
from control_v3.environment import ImprovedEnvironment

STATE_FEATURES = ["4*normalized_x", "4*normalized_y", "log(area/.04)/4",
    "confidence_or0", "remaining_action_time/horizon", "reserved0", "reserved0", "reserved0"]


class StaticState:
    def __init__(self, horizon):
        self.horizon = float(horizon)
        self.elapsed = 0.
        self.last_seen = np.zeros(3, np.float32)

    def build(self, detection, action=(0,0.)):
        self.elapsed += float(action[1])
        if detection is not None:
            self.last_seen = np.array([2*detection.center[0]-1, 2*detection.center[1]-1,
                np.clip(math.log(detection.area/.04)/4,-2,2)], np.float32)
        x,y,area = self.last_seen
        return np.array([np.clip(4*x,-2,2),np.clip(4*y,-2,2),area,
            detection.confidence if detection is not None else 0.,
            max(0.,1.-self.elapsed/self.horizon),0.,0.,0.],np.float32)


class StaticEnvironment(ImprovedEnvironment):
    def __init__(self, transport, detector, extractor, specification):
        if specification["id"] != "static_center":
            raise ValueError("v4 static observation has not been qualified for other stages")
        super().__init__(transport,detector,extractor,specification)

    def reset(self, scenario, seed):
        super().reset(scenario,seed)
        self.transport.mode("STATIC_OBSERVATION_PDQN_V4")
        self.builder = StaticState(self.spec["horizon_seconds"])
        return self.builder.build(self.selected)
