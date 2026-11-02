"""Image-based PTZ prototype: red-vehicle HSV detector and Gymnasium wrapper.

This is a controlled-scene baseline, not HAS-CAM or a general object detector.
Simulator target coordinates and projected ground-truth boxes are never used
by the policy observation or reward.
"""
import json
import math
import time
from pathlib import Path

import cv2
import gymnasium as gym
import numpy as np
from unrealcv import Client


def detect_vehicle(rgb):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, (0, 110, 45), (12, 255, 255))
    mask |= cv2.inRange(hsv, (168, 110, 45), (179, 255, 255))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < 20:
        return None
    x, y, w, h = cv2.boundingRect(contour)
    return np.array([x, y, x + w, y + h], dtype=np.float32) / np.array(
        [rgb.shape[1], rgb.shape[0], rgb.shape[1], rgb.shape[0]], dtype=np.float32
    )


def image_measurement(rgb):
    box = detect_vehicle(rgb)
    if box is None:
        return dict(visible=False, center=[0.0, 0.0], width=0.0, height=0.0,
                    area=0.0, center_error=math.sqrt(2), bbox=None)
    cx, cy = (box[:2] + box[2:]) - 1.0
    w, h = box[2:] - box[:2]
    return dict(visible=True, center=[float(cx), float(cy)], width=float(w),
                height=float(h), area=float(w * h),
                center_error=float(math.hypot(cx, cy)), bbox=box.tolist())


class ArenaClient:
    def __init__(self, port=9000, settle_seconds=0.03):
        self.client = Client(("127.0.0.1", port))
        self.client.connect(timeout=5)
        if not self.client.isconnected():
            raise RuntimeError("Open PTZSim and press Play before connecting")
        self.settle_seconds = settle_seconds
        self.state = self.rpc("vget /ptz/state")
        # Keep the old HSV/PPO comparison tied to its explicit single-car scene.
        legacy = Path(__file__).resolve().parents[1] / "paper_reproduction/scenarios/legacy_ellipse.json"
        if legacy.exists() and "scenario_name" in self.state:
            self.state = self.rpc(f"vset /ptz/scenario {legacy.as_posix()}")
        self.rgb = None

    def rpc(self, command):
        reply = self.client.request(command, timeout=15)
        if isinstance(reply, bytes):
            reply = reply.decode("utf-8")
        if not isinstance(reply, str) or reply.lower().startswith("error"):
            raise RuntimeError(f"{command}: {reply!r}. Open PTZSim and press Play.")
        return json.loads(reply)

    def reset(self, seed):
        self.state = self.rpc(f"vset /ptz/reset {int(seed)}")
        return self.observe()

    def step(self, pan, tilt, zoom, duration=0.2):
        values = np.asarray([pan, tilt, zoom, duration], dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("PTZ actions must be finite")
        self.state = self.rpc("vset /ptz/step " + " ".join(f"{x:.6f}" for x in values))
        return self.observe()

    def observe(self):
        if self.settle_seconds:
            time.sleep(self.settle_seconds)
        camera = int(self.state["camera_id"])
        if camera < 0:
            raise RuntimeError("PTZ capture camera was not registered")
        reply = self.client.request(f"vget /camera/{camera}/lit png", timeout=20)
        if not isinstance(reply, bytes):
            raise RuntimeError(f"Camera returned {reply!r}")
        bgr = cv2.imdecode(np.frombuffer(reply, dtype=np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError("Camera response is not a valid image")
        self.rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        measurement = image_measurement(self.rgb)
        # HUD presentation only. Detection always reads the untouched sensor RGB.
        box = measurement["bbox"] if measurement["visible"] else [0, 0, 0, 0]
        self.rpc("vset /ptz/detection " + " ".join(f"{x:.6f}" for x in box)
                 + f" {int(measurement['visible'])}")
        return measurement

    def close(self):
        self.client.disconnect()


class PTZTrackingEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 5}
    # Hold, pan left/right, tilt up/down, zoom in/out. Fixed 0.2 s baseline.
    ACTIONS = np.array([[0, 0, 0], [-1, 0, 0], [1, 0, 0], [0, 1, 0],
                        [0, -1, 0], [0, 0, 1], [0, 0, -1]], dtype=float)

    def __init__(self, port=9000, max_steps=200, render_mode=None):
        super().__init__()
        self.action_space = gym.spaces.Discrete(len(self.ACTIONS))
        self.observation_space = gym.spaces.Box(-1, 1, shape=(10,), dtype=np.float32)
        self.render_mode = render_mode
        self.arena = ArenaClient(port=port)
        self.max_steps = max_steps
        self.previous_center = np.zeros(2)
        self.steps = 0
        self.lost_steps = 0

    def observation(self, measurement):
        cx, cy = measurement["center"]
        delta = np.array([cx, cy]) - self.previous_center
        self.previous_center = np.array([cx, cy])
        pitch, yaw, _ = self.arena.state["rotation"]
        obs = [cx, cy, measurement["width"], measurement["height"],
               float(measurement["visible"]), yaw / 65, (pitch + 17.5) / 27.5,
               (self.arena.state["fov"] - 54) / 36, delta[0] / 2, delta[1] / 2]
        return np.clip(obs, -1, 1).astype(np.float32)

    def info(self, measurement):
        # Explicit allowlist: never expose simulator ground truth to the policy.
        return dict(measurement, sim_time=self.arena.state["sim_time"],
                    fov=self.arena.state["fov"], observation_source="rendered RGB + HSV red-body bbox")

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        scene_seed = int(self.np_random.integers(0, 2**30))
        measurement = self.arena.reset(scene_seed)
        self.previous_center = np.array(measurement["center"])
        self.steps = self.lost_steps = 0
        return self.observation(measurement), self.info(measurement)

    def step(self, action):
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action}")
        m = self.arena.step(*self.ACTIONS[int(action)], duration=0.2)
        self.steps += 1
        self.lost_steps = 0 if m["visible"] else self.lost_steps + 1
        if m["visible"]:
            scale_error = min(2.0, abs(math.log(max(m["area"], 1e-6) / 0.025)))
            reward = 1.0 - 0.8 * m["center_error"] - 0.4 * scale_error
        else:
            reward = -1.0
        return (self.observation(m), float(reward), self.lost_steps >= 15,
                self.steps >= self.max_steps, self.info(m))

    def render(self):
        return self.arena.rgb.copy()

    def close(self):
        self.arena.close()
