"""Continuous single-target RGB task; GT is used only in returned audit rows."""
import json
import math
import struct
import cv2
import numpy as np
from hascam.cfff import CFFF
from integration.focused_environment import FocusedTransport
from integration.metrics import iou


class AtomicTransport(FocusedTransport):
    """Raw sensor transport for perception; no video recording or replay."""
    def capture(self):
        packet = self.client.request("vget /ptz/frame", timeout=30)
        if not isinstance(packet, bytes) or len(packet) < 16:
            raise RuntimeError("Raw UE sensor packet unavailable")
        magic, length, width, height = struct.unpack_from("<4I", packet)
        offset = 16 + length
        if magic != 0x315A5450 or len(packet) != offset + width * height * 4 or (width, height) != (960, 540):
            raise RuntimeError("Invalid PTZ1 RGB sensor packet")
        self.diagnostics = json.loads(packet[16:offset].decode("utf-8"))
        bgra = np.frombuffer(packet, np.uint8, offset=offset).reshape(height, width, 4)
        self.rgb = cv2.cvtColor(bgra, cv2.COLOR_BGRA2RGB)
        return self.rgb


class ControlState:
    def __init__(self, max_duration=.25):
        self.max_duration = max_duration
        self.last_center = None
        self.last_seen = np.zeros(3, np.float32)
        self.trend = np.zeros(2, np.float32)

    def build(self, detection, action=(0, 0.)):
        if detection is not None:
            center = np.array([2 * value - 1 for value in detection.center], np.float32)
            self.trend = np.zeros(2, np.float32) if self.last_center is None else center - self.last_center
            self.last_center = center
            self.last_seen = np.array([*center, np.clip(math.log(detection.area / .04) / 4, -2, 2)], np.float32)
        # Preserve last-seen location/scale/trend on loss. Confidence=0 explicitly
        # marks unobserved state; there is no simulator-derived position guess.
        x, y, area = self.last_seen
        return np.array([np.clip(4*x, -2, 2), np.clip(4*y, -2, 2), area,
            detection.confidence if detection is not None else 0., action[0] / 6.,
            action[1] / self.max_duration, *np.clip(10*self.trend, -2, 2)], np.float32)


class ControlEnvironment:
    def __init__(self, transport, detector, extractor, specification):
        self.transport, self.detector, self.extractor = transport, detector, extractor
        self.spec = specification
        self.cfff = CFFF(extractor)

    def observe(self):
        self.detections = self.detector(self.transport.rgb)
        # Continuous tracking does not blacklist its target after ten confident
        # samples. ReID model/threshold/fusion are the frozen verified frontend.
        self.selected = self.cfff.choose(self.transport.rgb, self.detections, count_step=False)
        if self.selected is None:
            return {"visible": False, "center_error": None, "x": None, "y": None, "area_ratio": None}
        x, y = [2 * value - 1 for value in self.selected.center]
        return {"visible": True, "center_error": math.hypot(x, y), "x": x, "y": y, "area_ratio": self.selected.area}

    def audit_row(self, observation, action, reward=0., parts=None):
        truth = self.transport.diagnostics
        matched = bool(self.selected is not None and any(
            iou(self.selected.bbox, target.get("gt_bbox_normalized", (0,0,0,0))) >= .3
            for target in truth["targets"]))
        return observation | {"sim_time_s": truth["sim_time"], "matched": matched,
            "action_type": int(action[0]), "duration_s": float(action[1]), "fov": truth["fov"],
            "rotation": truth["rotation"], "reward": reward, "reward_parts": parts,
            "selected_bbox": list(self.selected.bbox) if self.selected else None,
            "confidence": self.selected.confidence if self.selected else 0.,
            "detector_seconds": self.detector.last_seconds,
            "gt_usage": "offline match flag only; never reward, state or action"}

    def reset(self, scenario, seed):
        self.transport.scenario(scenario)
        self.transport.reset(seed)
        self.transport.mode("LEARNED_CONTROL_V2_TRAINING_DIAGNOSTIC")
        self.cfff.reset()
        self.builder = ControlState()
        self.steps = 0
        self.previous = self.observe()
        self.rows = [self.audit_row(self.previous, (0,0.))]
        return self.builder.build(self.selected)

    def step(self, action, duration):
        if action not in self.spec["allowed_actions"]:
            raise ValueError("Action violates preregistered stage mask")
        self.transport.step(action, duration)
        if self.transport.diagnostics.get("demo_geometry_follow") or self.transport.diagnostics.get("free_preview"):
            raise RuntimeError("Image-policy task entered a preview mode")
        observation = self.observe()
        self.steps += 1
        # Dense image-only task objective. Fixed-FOV foundation ignores scale.
        def cost(value):
            if not value["visible"]:
                return 2.
            return value["center_error"] + (.20 * abs(math.log(value["area_ratio"]/.04)) if self.spec["scale_required"] else 0.)
        progress = 20. * (cost(self.previous) - cost(observation))
        if observation["visible"]:
            centered = observation["center_error"] <= .025
            scaled = not self.spec["scale_required"] or .025 <= observation["area_ratio"] <= .055
            rate = (1. if centered and scaled else -.2) - 6.*cost(observation)
        else:
            rate = -8.
        effort = -.02 if action != 0 else 0.
        reward = progress + rate * duration + effort
        done = self.transport.diagnostics["sim_time"] >= self.spec["horizon_seconds"] or self.steps >= self.spec["max_decisions"]
        state = self.builder.build(self.selected, (action,duration))
        row = self.audit_row(observation, (action,duration), reward,
            {"image_progress": progress, "image_rate": rate, "duration": duration, "effort": effort})
        self.rows.append(row)
        self.previous = observation
        return state, float(reward), done, row
