"""Portable inference for the official FastReID VeRi SBS R50-IBN checkpoint.

The computation and tensor names follow the FastReID sources bundled by the
HAS-CAM author (Apache-2.0). Only inference-required layers are implemented,
avoiding the old training engine's dependencies on CUDA, Apex and Faiss.
All inference tensors must load strictly; no random feature fallback exists.

Paper prose says 4096 dimensions; author config and release are 2048. The
actual 2048-dimensional vector is returned and reported, never padded/doubled.
"""
from pathlib import Path
import hashlib
import math
import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .detection import Detection


DEFAULT_WEIGHTS = Path(__file__).resolve().parent / "weights" / "veri_sbs_R50-ibn.pth"
OFFICIAL_URL = "https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/veri_sbs_R50-ibn.pth"


class IBN(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.half = channels // 2
        self.IN = nn.InstanceNorm2d(self.half, affine=True)
        self.BN = nn.BatchNorm2d(channels-self.half)

    def forward(self, x):
        a, b = torch.split(x, self.half, 1)
        return torch.cat((self.IN(a.contiguous()), self.BN(b.contiguous())), 1)


class NonLocal(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # The pinned FastReID implementation uses reduc_ratio//reduc_ratio=1.
        self.g = nn.Conv2d(channels, 1, 1)
        self.theta = nn.Conv2d(channels, 1, 1)
        self.phi = nn.Conv2d(channels, 1, 1)
        self.W = nn.Sequential(nn.Conv2d(1, channels, 1), nn.BatchNorm2d(channels))

    def forward(self, x):
        n, _, h, w = x.shape
        g = self.g(x).view(n, 1, -1).permute(0, 2, 1)
        theta = self.theta(x).view(n, 1, -1).permute(0, 2, 1)
        phi = self.phi(x).view(n, 1, -1)
        attention = torch.matmul(theta, phi) / (h*w)
        y = torch.matmul(attention, g).permute(0, 2, 1).contiguous().view(n, 1, h, w)
        return self.W(y) + x


class Bottleneck(nn.Module):
    def __init__(self, inputs, planes, stride=1, ibn=False):
        super().__init__()
        self.conv1 = nn.Conv2d(inputs, planes, 1, bias=False)
        self.bn1 = IBN(planes) if ibn else nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes*4, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes*4)
        self.downsample = nn.Sequential(nn.Conv2d(inputs, planes*4, 1, stride, bias=False),
                                        nn.BatchNorm2d(planes*4)) if stride != 1 or inputs != planes*4 else None

    def forward(self, x):
        identity = x if self.downsample is None else self.downsample(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return F.relu(self.bn3(self.conv3(x)) + identity)


class R50IBNBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(3, 2, ceil_mode=True)
        previous = 64
        counts = [3, 4, 6, 3]
        for stage, (planes, blocks) in enumerate(zip([64, 128, 256, 512], counts), 1):
            stride = 2 if stage in (2, 3) else 1
            layers = [Bottleneck(previous, planes, stride, ibn=stage < 4)]
            layers.extend(Bottleneck(planes*4, planes, ibn=stage < 4) for _ in range(blocks-1))
            setattr(self, f"layer{stage}", nn.Sequential(*layers))
            previous = planes*4
            nl_count = {1:0, 2:2, 3:3, 4:0}[stage]
            setattr(self, f"NL_{stage}", nn.ModuleList(NonLocal(previous) for _ in range(nl_count)))
            setattr(self, f"NL_{stage}_idx", list(range(blocks-nl_count, blocks)))

    def forward(self, x):
        x = self.maxpool(F.relu(self.bn1(self.conv1(x))))
        for stage in range(1, 5):
            nl_indices = getattr(self, f"NL_{stage}_idx")
            for i, block in enumerate(getattr(self, f"layer{stage}")):
                x = block(x)
                if i in nl_indices:
                    x = getattr(self, f"NL_{stage}")[nl_indices.index(i)](x)
        return x


class GeM(nn.Module):
    def __init__(self):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1)*3)

    def forward(self, x):
        return F.adaptive_avg_pool2d(x.clamp(min=1e-6).pow(self.p), 1).pow(1/self.p)


class ReIDModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = R50IBNBackbone()
        self.heads = nn.Module()
        self.heads.pool_layer = GeM()
        self.heads.bottleneck = nn.Sequential(nn.BatchNorm2d(2048))
        self.register_buffer("pixel_mean", torch.tensor([123.675,116.28,103.53]).view(1,3,1,1))
        self.register_buffer("pixel_std", torch.tensor([58.395,57.12,57.375]).view(1,3,1,1))

    def forward(self, x):
        x = (x-self.pixel_mean)/self.pixel_std
        features = self.backbone(x)
        return self.heads.bottleneck(self.heads.pool_layer(features))[...,0,0]


class FastReIDExtractor:
    def __init__(self, weights=DEFAULT_WEIGHTS, device="cpu", expected_dim=None):
        self.weights = Path(weights)
        if not self.weights.is_file() or self.weights.stat().st_size < 1_000_000:
            raise FileNotFoundError(f"Real VeRi checkpoint required: {self.weights}; download {OFFICIAL_URL}")
        self.device = torch.device(device)
        self.model = ReIDModel().to(self.device)
        checkpoint = torch.load(self.weights, map_location="cpu", weights_only=False)
        state = checkpoint["model"] if "model" in checkpoint else checkpoint
        state = {k.removeprefix("module."): torch.as_tensor(v) for k,v in state.items()}
        # Only the training classifier is unused during feature inference.
        self.unused_training_tensors = sorted(k for k in state if k in {"heads.weight", "heads.classifier.weight"})
        for key in self.unused_training_tensors:
            del state[key]
        # FastReID v0.1.1 release uses an old BN counter name while its actual
        # neck weights/statistics already use bottleneck.0 names.
        self.remapped_tensor_names = {}
        if "heads.bnneck.num_batches_tracked" in state:
            old, new = "heads.bnneck.num_batches_tracked", "heads.bottleneck.0.num_batches_tracked"
            state[new] = state.pop(old)
            self.remapped_tensor_names[old] = new
        self.model.load_state_dict(state, strict=True)
        self.model.eval().requires_grad_(False)
        self.feature_dim = 2048
        if expected_dim is not None and expected_dim != self.feature_dim:
            raise ValueError(f"Checkpoint output is {self.feature_dim}, requested {expected_dim}; paper's 4096 is unsupported by author config")
        self.sha256 = hashlib.sha256(self.weights.read_bytes()).hexdigest()

    def __call__(self, rgb, detection: Detection):
        h, w = rgb.shape[:2]
        x1, y1, x2, y2 = detection.bbox
        left, top = max(0, int(x1*w)), max(0, int(y1*h))
        right, bottom = min(w, max(left+1, math.ceil(x2*w))), min(h, max(top+1, math.ceil(y2*h)))
        crop = rgb[top:bottom, left:right]
        if not crop.size:
            raise ValueError("Empty detector crop")
        image = cv2.resize(crop, (256,256), interpolation=cv2.INTER_CUBIC)
        tensor = torch.as_tensor(image.astype(np.float32).transpose(2,0,1)[None], device=self.device)
        with torch.inference_mode():
            feature = F.normalize(self.model(tensor), dim=1)[0].cpu().numpy()
        if feature.shape != (self.feature_dim,) or not np.isfinite(feature).all():
            raise RuntimeError("Invalid pretrained ReID output")
        return feature

    def metadata(self):
        return {"name": "FastReID_SBS_R50_IBN_VeRi_official_release", "weights": str(self.weights),
                "weights_sha256": self.sha256, "weights_source": OFFICIAL_URL,
                "pretrained": True, "training_data": "VeRi-776 (FastReID public checkpoint)",
                "feature_dim_actual": self.feature_dim, "feature_dim_paper_prose": 4096,
                "dimension_conflict": True, "input_size": [256,256], "input_color": "RGB",
                "device": str(self.device), "unused_training_tensors": self.unused_training_tensors,
                "compatibility_tensor_remaps": self.remapped_tensor_names,
                "architecture": "ResNet50 + IBN + 5 nonlocal blocks + GeM + BN neck",
                "inference_load": "strict_all_inference_tensors"}
