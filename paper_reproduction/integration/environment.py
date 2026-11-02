"""Paper state/reward/task protocol on the explicitly different UE platform."""
from hascam.state_reward import StateBuilder, VPEReward
from integration.metrics import EpisodeMetrics


class PaperEnvironment:
    def __init__(self, transport, detector, cfff, reward_mode="author_code", max_steps=128):
        self.transport, self.detector, self.cfff = transport, detector, cfff
        self.reward = VPEReward(mode=reward_mode)
        self.builder = StateBuilder()
        self.max_steps = max_steps
        self.steps = 0

    def reset(self, scenario, seed, mode):
        self.transport.scenario(scenario)
        rgb = self.transport.reset(seed)
        self.transport.mode(mode)
        # The known task target count is used for termination only, never state/action.
        config = self.transport.diagnostics["scenario_config"]
        self.expected_count = len(config["targets"])
        self.steps = 0
        self.cfff.reset()
        self.builder.reset()
        self.detections = self.detector(rgb)
        self.selected = self.cfff.choose(rgb, self.detections, count_step=False)
        self.reward.reset(self.selected)
        self.observation = self.builder.build(self.selected, reset=True)
        self.metrics = EpisodeMetrics(self.expected_count)
        self.metrics.add(self.selected, self.transport.diagnostics, 0)
        self.transport.overlay(self.selected)
        return self.observation.copy()

    def step(self, action_type, duration):
        rgb = self.transport.step(action_type, duration)
        if self.transport.diagnostics.get("demo_geometry_follow") or self.transport.diagnostics.get("free_preview"):
            raise RuntimeError("Policy episode unexpectedly entered simulator preview mode")
        self.steps += 1
        self.detections = self.detector(rgb)
        self.selected = self.cfff.choose(rgb, self.detections)
        if self.cfff.last_event.get("new_target", False):
            self.builder.reset()
            self.reward.reset()
        result = self.reward.step(self.selected)
        self.observation = self.builder.build(self.selected, last_action=(action_type, duration))
        success = self.cfff.completed_count >= self.expected_count and self.steps < self.max_steps
        done = success or self.steps >= self.max_steps
        adjustment = self.reward.terminal_adjustment(success, self.steps) if done else 0
        event = self.cfff.last_event
        row = self.metrics.add(self.selected, self.transport.diagnostics, duration,
                               event.get("completed_detection") if event.get("completed") else None)
        self.transport.overlay(self.selected)
        # This row is for logging; learner receives only the eight-vector, reward and done.
        info = row | {"reward": float(result["reward"] + adjustment),
                      "reward_components": result["components"], "terminal_adjustment": adjustment,
                      "success": success, "completed_features": self.cfff.completed_count,
                      "expected_task_count": self.expected_count,
                      "action_type": int(action_type), "duration_s": float(duration),
                      "state": self.observation.tolist(),
                      "selected_bbox": list(self.selected.bbox) if self.selected else None,
                      "detections": [{"bbox": list(d.bbox), "confidence": d.confidence} for d in self.detections],
                      "detector_seconds": self.detector.last_seconds}
        info["raw_detector_count"] = len(self.detections)
        info["selected_absence_reason"] = (None if self.selected else
                                           "detector_empty" if not self.detections else "CFFF_filtered")
        return self.observation.copy(), info["reward"], done, info
