"""I/O bindings for the original UnityCameraControlEnv class on UE."""
import collections
import json
import struct
from types import SimpleNamespace
import cv2
import gymnasium as gym
import numpy as np
from scipy.spatial.distance import cosine as scipy_cosine
import torch
from torch.nn import functional as F
from author_aligned.source import environment_class
from integration.transport import ACTION_RATES
from integration.focused_environment import FocusedTransport

COMMANDS = ("STAY", "DOWN", "UP", "RIGHT", "LEFT", "ZOOMIN", "ZOOMOUT")


class AtomicTransport(FocusedTransport):
    """Decode a same-state RGB sensor packet; no v2 state/reward dependency."""
    def capture(self):
        packet=self.client.request("vget /ptz/frame",timeout=30)
        if not isinstance(packet,bytes) or len(packet)<16:
            raise RuntimeError("Raw UE sensor packet unavailable")
        magic,length,width,height=struct.unpack_from("<4I",packet)
        offset=16+length
        if magic!=0x315A5450 or len(packet)!=offset+width*height*4 or (width,height)!=(960,540):
            raise RuntimeError("Invalid PTZ1 sensor packet")
        self.diagnostics=json.loads(packet[16:offset].decode("utf-8"))
        bgra=np.frombuffer(packet,np.uint8,offset=offset).reshape(height,width,4)
        self.rgb=cv2.cvtColor(bgra,cv2.COLOR_BGRA2RGB)
        return self.rgb


class UEBridge:
    def __init__(self, transport, seconds_per_parameter=.2, execution_boundary="reject"):
        if seconds_per_parameter != .2:
            raise ValueError("Only the explicitly user-approved raw/5 mapping is configured")
        if execution_boundary not in ("reject", "clip_box"):
            raise ValueError("Unknown execution boundary")
        self.transport = transport
        self.seconds_per_parameter = seconds_per_parameter
        self.execution_boundary = execution_boundary
        self.seed = None
        self.scenario = None
        self.target_count = None
        self.last_action = None

    def SendData(self, message):
        command, value = message.split(",", 1)
        if command == "RESET":
            self.transport.scenario(self.scenario)
            self.transport.reset(self.seed)
            self.last_action = None
            return
        raw = float(value)
        if not np.isfinite(raw):
            raise ValueError("Nonfinite author parameter")
        bounded = min(5., max(.3, raw))
        out_of_box = not .3 - 1e-7 <= raw <= 5. + 1e-7
        if out_of_box and self.execution_boundary == "reject":
            raise ValueError(f"Author raw parameter {raw} is outside declared Box[0.3,5]; execution requires an explicit boundary policy")
        executed = bounded if self.execution_boundary == "clip_box" else raw
        duration = executed * self.seconds_per_parameter
        action = COMMANDS.index(command)
        before = self.transport.diagnostics["sim_time"]
        pan, tilt, zoom = ACTION_RATES[action]
        self.transport.rpc(f"vset /ptz/step {pan} {tilt} {zoom} {duration:.12f}")
        self.transport.capture()
        self.last_action = {"action_type": action, "raw_parameter": raw,
            "requested_seconds": raw * self.seconds_per_parameter,
            "executed_parameter": executed, "duration_s": duration,
            "measured_seconds": self.transport.diagnostics["sim_time"] - before,
            "execution_clipped": out_of_box, "execution_boundary": self.execution_boundary}
        if abs(self.last_action["measured_seconds"] - duration) > 1e-5:
            raise RuntimeError("UE execution time differs from declared mapping")

    def ReadReceivedData(self):
        # Known task cardinality, not a visible/GT-detected count. The original
        # environment needs it for sequential target completion/terminal reward.
        return f"ActionDone|{self.target_count}#0%{self.seed}"


class SensorCV2:
    def __init__(self, transport):
        self.transport = transport

    def imread(self, _):
        return cv2.cvtColor(self.transport.rgb, cv2.COLOR_RGB2BGR)

    def __getattr__(self, name):
        if name in ("imshow", "namedWindow", "resizeWindow", "waitKey"):
            return lambda *args, **kwargs: None
        return getattr(cv2, name)


class FrozenFeatures:
    def __init__(self, extractor):
        self.extractor = extractor

    def run_on_image(self, bgr):
        # Matches the author's FeatureExtractionDemo preprocessing exactly;
        # the verified frozen inference model replaces obsolete framework I/O.
        rgb = bgr[:, :, ::-1]
        image = cv2.resize(rgb, (256, 256), interpolation=cv2.INTER_CUBIC)
        tensor = torch.as_tensor(image.astype("float32").transpose(2, 0, 1)[None],
                                 device=self.extractor.device)
        with torch.inference_mode():
            return self.extractor.model(tensor)


def bind_environment(transport, detector, extractor, execution_boundary="reject"):
    bridge = UEBridge(transport, execution_boundary=execution_boundary)
    detected = {"boxes": [], "error": None}

    def detect(*args, **kwargs):
        try:
            rows = detector(transport.rgb)
            detected["boxes"] = rows
            detected["error"] = None
            # Only format conversion. Detector remains at .15; original env's
            # additional score_thr=.3 predicate is preserved, not tuned.
            return [np.array([[d.bbox[0]*960, d.bbox[1]*540,
                               d.bbox[2]*960, d.bbox[3]*540, d.confidence]
                              for d in rows], dtype=np.float32).reshape(-1, 5)]
        except Exception as error:
            detected["error"] = error
            raise

    namespace = {"__name__": "_author_environment_on_ue", "gym": gym, "spaces": gym.spaces,
        "np": np, "torch": torch, "F": F, "sock": bridge, "cv2": SensorCV2(transport),
        "bbox_deque": collections.deque(maxlen=4), "setup_cfg": lambda args: None,
        "args": SimpleNamespace(parallel=False), "detect": detect,
        "FeatureExtractionDemo": lambda cfg, parallel=False: FrozenFeatures(extractor),
        # Current SciPy requires 1-D cosine inputs; original gives (1,2048).
        "cosine": lambda a, b: scipy_cosine(np.asarray(a).ravel(), np.asarray(b).ravel())}
    env = environment_class(namespace)(a_p=0, a_r=0, e_thres=.5, interactive=False)
    env.io_bridge, env.detected = bridge, detected
    return env
