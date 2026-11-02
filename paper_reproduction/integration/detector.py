"""Real SSD probabilities. Simulator boxes are never used during inference.

SSD is one of the paper's detector families. Torchvision/COCO initialization
and UE adaptation differ from the authors' MMDetection / Unity 535-image set.
"""
from pathlib import Path
import hashlib
import time
import numpy as np
import torch
from torchvision.models.detection import ssd300_vgg16, SSD300_VGG16_Weights
from hascam import Detection

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()


class SSDDetector:
    def __init__(self, checkpoint=None, threshold=0.15, device="cpu", threads=4):
        torch.set_num_threads(threads)
        torch.hub.set_dir(str(ROOT / "weights" / "torch_hub"))
        self.device = torch.device(device)
        self.threshold = float(threshold)
        if checkpoint:
            self.model = ssd300_vgg16(weights=None, weights_backbone=None)
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            self.model.load_state_dict(payload["model"])
            self.provenance = payload["provenance"] | {
                "checkpoint": str(Path(checkpoint).resolve()), "sha256": sha256(checkpoint),
                "epochs_completed": payload.get("epochs_completed"),
                "requested_epochs": payload["provenance"].get("epochs")}
            self.provenance.pop("epochs", None)
        else:
            self.model = ssd300_vgg16(weights=SSD300_VGG16_Weights.DEFAULT)
            path = ROOT / "weights/torch_hub/checkpoints/ssd300_vgg16_coco-b556d3b4.pth"
            self.provenance = {"model": "torchvision SSD300 VGG16 COCO_V1",
                               "adapted_to_UE": False,
                               "sha256": sha256(path) if path.exists() else None}
        self.model.to(self.device).eval()
        self.last_seconds = 0.0

    def __call__(self, rgb):
        start = time.perf_counter()
        tensor = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))).to(
            self.device, dtype=torch.float32) / 255.0
        with torch.inference_mode():
            result = self.model([tensor])[0]
        h, w = rgb.shape[:2]
        output = []
        for box, score, label in zip(result["boxes"].cpu().numpy(),
                                     result["scores"].cpu().numpy(),
                                     result["labels"].cpu().numpy()):
            if int(label) != 3 or float(score) < self.threshold:
                continue
            normalized = np.clip(box / [w, h, w, h], 0, 1)
            if normalized[2] > normalized[0] and normalized[3] > normalized[1]:
                output.append(Detection(bbox=tuple(float(x) for x in normalized),
                                        confidence=float(score), class_id=3))
        self.last_seconds = time.perf_counter() - start
        return output
