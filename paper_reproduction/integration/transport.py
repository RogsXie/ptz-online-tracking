"""Raw RGB transport. No HSV processing and no simulator labels in policy input."""
import json
import time
import cv2
import numpy as np
from unrealcv import Client

# PDF Eq.1 order; Unreal positive tilt pitches camera upward.
ACTION_RATES = ((0, 0, 0), (0, -1, 0), (0, 1, 0), (1, 0, 0),
                (-1, 0, 0), (0, 0, 1), (0, 0, -1))


class RGBTransport:
    def __init__(self, port, settle_seconds=0.025):
        self.client = Client(("127.0.0.1", int(port)))
        self.client.connect(timeout=5)
        if not self.client.isconnected():
            raise ConnectionError("Standalone PTZ arena unavailable")
        self.settle_seconds = settle_seconds
        self.diagnostics = self.rpc("vget /ptz/state")
        self.rgb = None

    def rpc(self, command):
        reply = self.client.request(command, timeout=30)
        if isinstance(reply, bytes):
            reply = reply.decode("utf-8")
        if not isinstance(reply, str) or reply.lower().startswith("error"):
            raise RuntimeError(f"UE request failed: {command}: {reply!r}")
        return json.loads(reply)

    def scenario(self, path):
        self.diagnostics = self.rpc(f"vset /ptz/scenario {str(path).replace(chr(92), '/')}")

    def reset(self, seed):
        self.diagnostics = self.rpc(f"vset /ptz/reset {int(seed)}")
        return self.capture()

    def mode(self, label):
        if not label.replace("_", "").replace("-", "").isalnum():
            raise ValueError("HUD labels use alphanumeric characters and underscores")
        self.rpc(f"vset /ptz/mode {label}")

    def step(self, action_type, duration):
        if not 0 <= int(action_type) < 7 or not 0.04 <= duration <= 1:
            raise ValueError("Paper action outside seven types / [0.04,1] seconds")
        p, t, z = ACTION_RATES[int(action_type)]
        self.diagnostics = self.rpc(f"vset /ptz/step {p} {t} {z} {duration:.9f}")
        return self.capture()

    def capture(self):
        if self.settle_seconds:
            time.sleep(self.settle_seconds)
        cid = int(self.diagnostics["camera_id"])
        reply = self.client.request(f"vget /camera/{cid}/lit png", timeout=30)
        if not isinstance(reply, bytes):
            raise RuntimeError(f"Camera did not return PNG: {reply!r}")
        image = cv2.imdecode(np.frombuffer(reply, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError("Invalid PNG response")
        self.rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return self.rgb

    def overlay(self, selected):
        box = selected.bbox if selected is not None else (0, 0, 0, 0)
        # UnrealCV [float] routes accept decimal notation, not scientific e-05.
        self.rpc("vset /ptz/detection " + " ".join(f"{float(v):.12f}" for v in box)
                 + f" {int(selected is not None)}")

    def close(self):
        self.client.disconnect()
