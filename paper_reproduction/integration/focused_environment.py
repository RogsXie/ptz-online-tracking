"""Explicit quality-task extension of HAS-CAM; original PaperEnvironment stays intact."""
import math
import numpy as np
from hascam.cfff import CFFF, unit
from hascam.state_reward import VPEReward
from integration.environment import PaperEnvironment
from integration.transport import RGBTransport


def quality_of(detection, center_tolerance=.15, area_min=.02, area_max=.08):
    if detection is None:
        return False
    error = math.hypot(*(2 * value - 1 for value in detection.center))
    return bool(detection.confidence >= .9 and error <= center_tolerance
                and area_min <= detection.area <= area_max)


class QualityCFFF(CFFF):
    """Same appearance selection, but commit Fmon only after quality completion.

    This is a declared task extension, not a claim about the printed Algorithm 1.
    The predicate reads only image detection, never simulator truth.
    """
    def __init__(self, extractor, center_tolerance=.15, area_min=.02, area_max=.08):
        self.center_tolerance, self.area_min, self.area_max = center_tolerance, area_min, area_max
        super().__init__(extractor)

    def choose(self, rgb, detections, count_step=True):
        selected = super().choose(rgb, detections, count_step=False)
        if selected is None or not count_step:
            return selected
        eligible = quality_of(selected, self.center_tolerance, self.area_min, self.area_max)
        self.confidence_streak = self.confidence_streak + 1 if eligible else 0
        if self.confidence_streak >= self.consecutive_steps:
            self.monitored.append(unit(np.mean(self.current_features, axis=0)))
            self.completed_count += 1
            self.last_event.update(completed=True, completion_count=self.completed_count,
                                   completed_detection=selected)
            self.current_feature, self.current_features, self.confidence_streak = None, [], 0
        return selected

    def metadata(self):
        return super().metadata() | {"completion_mode": "quality_task_extension",
                                    "center_error_max": self.center_tolerance,
                                    "area_ratio_range": [self.area_min, self.area_max],
                                    "paper_difference": "Fmon commit waits for center/area/confidence, not confidence alone"}


class QualityVPEReward(VPEReward):
    @staticmethod
    def terminal_adjustment(success, steps):
        # Episode limits are explicitly supplied by the focused protocol.
        return 40. if success else -40.


class FocusedTransport(RGBTransport):
    def __init__(self, port, hud=True, settle_seconds=0., reset_warmup_captures=3):
        super().__init__(port, settle_seconds=settle_seconds)
        self.hud_enabled = hud
        self.reset_warmup_captures = reset_warmup_captures

    def reset(self, seed):
        rgb = super().reset(seed)
        # Rendering history settles at the same simulator time and fixed pose.
        for _ in range(self.reset_warmup_captures):
            rgb = self.capture()
        return rgb

    def overlay(self, selected):
        if self.hud_enabled:
            return super().overlay(selected)


class FocusedEnvironment(PaperEnvironment):
    def __init__(self, transport, detector, extractor, profile="quality", max_steps=128,
                 sim_budget=30., reward_mode="author_code"):
        if profile not in {"paper", "quality"}:
            raise ValueError("Unknown focused task profile")
        if max_steps < 1 or sim_budget <= 0:
            raise ValueError("Focused budgets must be positive")
        if profile == "paper" and max_steps != 128:
            raise ValueError("Paper profile requires its original 128-decision boundary")
        cfff = QualityCFFF(extractor) if profile == "quality" else CFFF(extractor)
        super().__init__(transport, detector, cfff, reward_mode, max_steps)
        self.profile, self.sim_budget = profile, sim_budget
        if profile == "quality":
            self.reward = QualityVPEReward(mode=reward_mode)

    def reset(self, scenario, seed, mode):
        observation = super().reset(scenario, seed, mode)
        if self.profile == "quality" and self.expected_count != 1:
            raise ValueError("Focused quality protocol requires exactly one target")
        self.paper_streak = self.quality_streak = 0
        self.paper_condition_ever = self.quality_condition_ever = False
        self.first_paper_step = self.first_quality_step = None
        self.quality_action_count = 0
        return observation

    def step(self, action_type, duration):
        observation, reward, done, info = super().step(action_type, duration)
        confident = self.selected is not None and self.selected.confidence >= .9
        quality = quality_of(self.selected)
        self.paper_streak = self.paper_streak + 1 if confident else 0
        self.quality_streak = self.quality_streak + 1 if quality else 0
        self.quality_action_count += int(quality)
        if self.paper_streak >= 10 and not self.paper_condition_ever:
            self.paper_condition_ever, self.first_paper_step = True, self.steps
        if self.quality_streak >= 10 and not self.quality_condition_ever:
            self.quality_condition_ever, self.first_quality_step = True, self.steps
        time_limit = self.profile == "quality" and self.transport.diagnostics["sim_time"] >= self.sim_budget
        budget_limit = self.profile == "quality" and (time_limit or self.steps >= self.max_steps)
        if budget_limit:
            # Both budget boundaries have failure priority over coincident completion.
            if done:
                reward += -40. - info["terminal_adjustment"]
            else:
                reward -= 40.
            info.update(reward=reward, success=False, terminal_adjustment=-40.)
            done = True
            if self.cfff.last_event.get("completed"):
                self.metrics.completed_ids.clear()
                self.metrics.completion_matches[-1].update(accepted=False, rejected_by_budget=True)
                self.cfff.monitored.pop()
                self.cfff.completed_count -= 1
                self.cfff.last_event.update(completed=False, completion_count=self.cfff.completed_count,
                                           completion_rejected_by_budget=True)
                info["completed_features"] = self.cfff.completed_count
        info.update(task_profile=self.profile, quality_now=quality,
                    paper_confidence_streak=self.paper_streak, quality_streak=self.quality_streak,
                    paper_condition_ever=self.paper_condition_ever,
                    quality_condition_ever=self.quality_condition_ever,
                    budget_time_reached=time_limit, budget_decisions_reached=self.steps >= self.max_steps)
        return observation, reward, done, info

    def summary(self, success):
        return self.metrics.summary(success) | {
            "task_profile": self.profile, "paper_confidence_condition_ever": self.paper_condition_ever,
            "quality_condition_ever": self.quality_condition_ever,
            "quality_action_fraction": self.quality_action_count / max(1, self.steps),
            "first_paper_condition_action": self.first_paper_step,
            "first_quality_condition_action": self.first_quality_step,
            "decision_budget": self.max_steps, "simulation_budget_s": self.sim_budget,
            "simulation_budget_overshoot_s": max(0., self.transport.diagnostics["sim_time"] - self.sim_budget),
            "note_task": "quality profile changes completion and terminal budget; VPE per-frame cost/state/networks remain audited original variants"}
