"""Reuse image-only dense control objective and corrected action history.

    Extra simulator telemetry is an execution audit, never a learner feature.
"""
import math
from control_learning.environment import ControlEnvironment


class ImprovedEnvironment(ControlEnvironment):
    def reset(self, scenario, seed):
        state = super().reset(scenario, seed)
        self.transport.mode("ISOLATED_PDQN_V3_RGB_CONTROL")
        return state

    def step(self, action, duration):
        before = self.rows[-1]
        state, reward, done, row = super().step(action, duration)
        elapsed = row["sim_time_s"] - before["sim_time_s"]
        if not math.isfinite(reward) or abs(elapsed - duration) > 1e-5:
            raise RuntimeError("Nonfinite reward or action/simulation time mismatch")
        if self.spec["id"] == "static_center":
            expected = list(before["rotation"])
            if action in (1, 2):
                expected[0] = min(0., max(-12., expected[0] + (1 if action == 2 else -1) * 8 * duration))
            elif action in (3, 4):
                expected[1] = min(10., max(-10., expected[1] + (1 if action == 3 else -1) * 8 * duration))
            error = max(abs(a-b) for a,b in zip(expected, row["rotation"]))
            if error > 1e-3 or abs(row["fov"] - 60) > 1e-4:
                raise RuntimeError(f"Actuator response mismatch: {error} degrees")
            row["actuator_error_deg"] = error
        unchanged = max(abs(a-b) for a,b in zip(before["rotation"], row["rotation"])) < 1e-5 and abs(row["fov"]-before["fov"]) < 1e-5
        row["saturated_nonstay"] = action != 0 and unchanged
        return state, reward, done, row
