"""Frozen YOLO weights; RGB-only confidence arbitration across two scales."""
import time
import cv2
import numpy as np
from hascam.detection import Detection
from integration.pretrained_detector import create_detector as base_detector
from integration.metrics import iou

class MultiScaleDetector:
    def __init__(self,**kwargs):
        self.base=base_detector(**kwargs);self.checkpoint=self.base.checkpoint
        self.provenance=self.base.provenance|dict(preprocessing_profile='full_and_half_rgb_confidence_nms_v13',
            scales=[1.,.5],selection='confidence descending cross-scale NMS IoU>.5',no_gt_or_fov_input=True)
        self.last_seconds=0.;self.last_fallback=False;self.last_scales=[]
    def __call__(self,rgb):
        start=time.perf_counter();h,w=rgb.shape[:2];hh,ww=h//2,w//2;x0,y0=(w-ww)//2,(h-hh)//2
        canvas=np.full_like(rgb,114);canvas[y0:y0+hh,x0:x0+ww]=cv2.resize(rgb,(ww,hh),interpolation=cv2.INTER_AREA)
        candidates=[(d,1.) for d in self.base(rgb)]
        for d in self.base(canvas):
            b=(np.asarray(d.bbox)*[w,h,w,h]-[x0,y0,x0,y0])/[ww,hh,ww,hh];b=np.clip(b,0,1)
            if b[2]>b[0] and b[3]>b[1]:candidates.append((Detection(tuple(b),d.confidence,d.class_id,d.label),.5))
        chosen=[];scales=[]
        for d,scale in sorted(candidates,key=lambda v:v[0].confidence,reverse=True):
            if not any(d.class_id==old.class_id and iou(d.bbox,old.bbox)>.5 for old in chosen):
                chosen.append(d);scales.append(scale)
        self.last_seconds=time.perf_counter()-start;self.last_scales=scales;self.last_fallback=bool(scales and scales[0]==.5)
        return chosen
    def warmup(self,iterations=3):
        for _ in range(iterations):self(np.zeros((540,960,3),np.uint8))

def create_detector(**kwargs):return MultiScaleDetector(**kwargs)
