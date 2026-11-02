"""Official pretrained YOLO detector with explicit CUDA and RGB-only inference.

This is an alternative detector backend, not the detector used in the original
paper. Official weights and any UE adaptation must have separate provenance.
"""
from pathlib import Path
import hashlib
import os
import sys
import time
import numpy as np
import torch
from hascam import Detection

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1048576),b""):
            digest.update(chunk)
    return digest.hexdigest()


class YOLODetector:
    """call(rgb)->list[Detection], compatible with the SSD environment interface."""
    def __init__(self,checkpoint=None,threshold=.15,device="cuda",imgsz=640,
                 target_labels=("car",),half=False):
        # Keep package settings local; do not auto-install/upgrade dependencies.
        os.environ.setdefault("YOLO_CONFIG_DIR",str(ROOT/"verification/pretrained-detectors-gpu/ultralytics-settings"))
        os.environ.setdefault("YOLO_AUTOINSTALL","false")
        from ultralytics import YOLO, __version__ as ultralytics_version
        self.checkpoint = Path(checkpoint or ROOT/"weights/yolo26s.pt").resolve()
        if not self.checkpoint.is_file():
            raise FileNotFoundError(f"Download audited official weights first: {self.checkpoint}")
        self.device = torch.device(device)
        if self.device.type=="cuda" and not torch.cuda.is_available():
            raise RuntimeError(f"CUDA requested but unavailable in {sys.executable}")
        self.threshold = float(threshold)
        if not 0<self.threshold<1:
            raise ValueError("Detector threshold must lie in (0,1)")
        self.imgsz = int(imgsz)
        self.half = bool(half)
        if self.half and self.device.type!="cuda":
            raise ValueError("half precision requires CUDA in this backend")
        self.model = YOLO(str(self.checkpoint),task="detect")
        self.model.to(self.device)
        self.model.model.eval()
        self.class_ids = [int(index) for index,name in self.model.names.items()
                          if name in set(target_labels)]
        if not self.class_ids:
            raise ValueError(f"Requested target labels {target_labels} absent from model names {self.model.names}")
        source_file = self.checkpoint.with_suffix(".source.json")
        source = __import__("json").loads(source_file.read_text(encoding="utf-8")) if source_file.exists() else {}
        self.provenance = {"model":self.checkpoint.stem,"backend":"ultralytics",
                           "ultralytics":ultralytics_version,"checkpoint":str(self.checkpoint),
                           "sha256":sha256(self.checkpoint),"weight_source":source,
                           "adapted_to_UE":bool(source.get("adapted_to_UE",False)),
                           "device":str(self.device),"python":sys.executable,
                           "torch":torch.__version__,"cuda_build":torch.version.cuda,
                           "target_class_ids":self.class_ids,"target_labels":list(target_labels),
                           "threshold":self.threshold,"imgsz":self.imgsz,"half":self.half,
                           "inference_inputs":"rendered RGB only; no GT or HSV",
                           "timing_scope":"synchronized CUDA wall time including preprocessing, forward, postprocessing and Detection conversion",
                           "difference_from_paper":("Ultralytics model with UE GPU domain adaptation, not author MMDetection detector"
                                                   if source.get("adapted_to_UE",False) else
                                                   "Ultralytics model and COCO pretrained weights, not author MMDetection detector")}
        if self.device.type=="cuda":
            self.provenance.update(gpu=torch.cuda.get_device_name(self.device),
                                   capability=list(torch.cuda.get_device_capability(self.device)))
        self.last_seconds = 0.

    def __call__(self,rgb):
        if not isinstance(rgb,np.ndarray) or rgb.ndim!=3 or rgb.shape[2]!=3 or rgb.dtype!=np.uint8:
            raise ValueError("Expected HxWx3 uint8 RGB sensor image")
        if self.device.type=="cuda":
            torch.cuda.synchronize(self.device)
        start=time.perf_counter()
        # Ultralytics numpy sources follow OpenCV BGR; its internal transform
        # converts BGR to model RGB. Never feed RGB as if it were BGR.
        bgr=np.ascontiguousarray(rgb[:,:,::-1])
        extra={"half":True} if self.half else {}
        with torch.inference_mode():
            result=self.model.predict(source=bgr,device=str(self.device),imgsz=self.imgsz,
                                      conf=self.threshold,classes=self.class_ids,verbose=False,
                                      save=False,max_det=100,iou=.7,**extra)[0]
        h,w=rgb.shape[:2]
        output=[]
        if result.boxes is not None:
            for box,score,label in zip(result.boxes.xyxy.cpu().numpy(),
                                       result.boxes.conf.cpu().numpy(),result.boxes.cls.cpu().numpy()):
                normalized=np.clip(box/[w,h,w,h],0,1)
                if normalized[2]>normalized[0] and normalized[3]>normalized[1]:
                    output.append(Detection(tuple(float(v) for v in normalized),float(score),
                                            int(label),self.model.names[int(label)]))
        if self.device.type=="cuda":
            torch.cuda.synchronize(self.device)
        self.last_seconds=time.perf_counter()-start
        return output

    def warmup(self,iterations=3):
        sample=np.zeros((540,960,3),np.uint8)
        for _ in range(int(iterations)):
            self(sample)


def create_detector(kind="yolo26s_ue",checkpoint=None,device="cuda",threshold=.15,imgsz=960):
    """Explicit backend factory; a missing adapted weight never falls back silently."""
    defaults={"yolo26s_ue":ROOT/"weights/yolo26s_ue_gpu30.pt",
              "yolo26s":ROOT/"weights/yolo26s.pt","yolo11s":ROOT/"weights/yolo11s.pt",
              "ssd":ROOT/"weights/ssd_ue.pt"}
    if kind not in defaults:
        raise ValueError(f"Unknown detector backend {kind!r}; choose {tuple(defaults)}")
    path=Path(checkpoint) if checkpoint else defaults[kind]
    if not path.is_file():
        raise FileNotFoundError(f"Requested detector checkpoint not ready: {path}")
    if kind=="yolo26s_ue" and not path.with_suffix(".source.json").is_file():
        raise FileNotFoundError(f"Adapted detector provenance is not published yet: {path.with_suffix('.source.json')}")
    if kind=="ssd":
        from integration.detector import SSDDetector
        return SSDDetector(path,device=device,threshold=threshold)
    return YOLODetector(path,device=device,threshold=threshold,imgsz=imgsz)
