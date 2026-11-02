"""RGB-only scale fallback around the unchanged, frozen YOLO checkpoint."""
import time
import cv2
import numpy as np
from hascam.detection import Detection
from integration.pretrained_detector import create_detector as original_detector


class ScaleFallbackDetector:
    def __init__(self,**kwargs):
        self.base=original_detector(**kwargs)
        self.checkpoint=self.base.checkpoint
        self.provenance=self.base.provenance | {'preprocessing_profile':'full_rgb_then_half_size_rgb_fallback_v11',
            'fallback_trigger':'no detections on full RGB', 'fallback_scale':.5,
            'padding_rgb':[114,114,114], 'no_gt_or_fov_input':True,
            'coordinate_mapping':'inverse of centered half-size affine transform; output in original image coordinates'}
        self.last_seconds=0.;self.last_fallback=False

    def __call__(self,rgb):
        started=time.perf_counter();output=self.base(rgb);self.last_fallback=False
        if not output:
            h,w=rgb.shape[:2];small=cv2.resize(rgb,(w//2,h//2),interpolation=cv2.INTER_AREA)
            canvas=np.full_like(rgb,114);x0=(w-small.shape[1])//2;y0=(h-small.shape[0])//2
            canvas[y0:y0+small.shape[0],x0:x0+small.shape[1]]=small
            self.last_fallback=True
            for det in self.base(canvas):
                b=np.asarray(det.bbox)*[w,h,w,h]
                b=(b-[x0,y0,x0,y0])/[small.shape[1],small.shape[0],small.shape[1],small.shape[0]]
                b=np.clip(b,0,1)
                if b[2]>b[0] and b[3]>b[1]:output.append(Detection(tuple(b),det.confidence,det.class_id,det.label))
        self.last_seconds=time.perf_counter()-started
        return output

    def warmup(self,iterations=3):
        for _ in range(iterations):self(np.zeros((540,960,3),np.uint8))


def create_detector(**kwargs):return ScaleFallbackDetector(**kwargs)
