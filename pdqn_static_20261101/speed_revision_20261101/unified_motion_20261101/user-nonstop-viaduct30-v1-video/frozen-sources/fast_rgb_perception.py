"""Image-only fast path. No new motor rule and no target-state oracle.

Run full-size frozen YOLO once; retain original multiscale/ROI recovery when
confidence or RGB association is weak. During strong single-candidate image
association, reuse an actual ReID feature only while its age is <=150ms. A changed,
ambiguous or lost appearance always triggers the real extractor immediately.
"""
import speed_bootstrap
import time,math
import numpy as np
from unified_perception import FoveatedDetector

class Detector(FoveatedDetector):
    def __init__(self,base):
        super().__init__(base);self.provenance.update(preprocessing_profile='single_full_with_multiscale_recovery_v1',
            fast_path='One originalFP32fullRGBpass; original full/half/ROI recovery onloss or confidence<.65. Weights and threshold unchanged.')
        self.fast_calls=self.recovery_calls=0;self.current_candidate_count=0
    def __call__(self,rgb):
        start=time.perf_counter();d=list(self.base.base(rgb))
        if self.lost or not d or max(x.confidence for x in d)<.65:
            self.recovery_calls+=1;d=super().__call__(rgb)
        else:self.fast_calls+=1
        self.current_candidate_count=len(d);self.last_detections=list(d);self.last_seconds=time.perf_counter()-start;return d

class PeriodicReID:
    """One geometrically associated candidate; cached feature age <=150ms."""
    def __init__(self,base,period=.15):
        self.base=base;self.period=period;self.transport=None;self.association=None;self.detector=None;self.last=None;self.real_calls=self.cached_calls=0
        self.sha256=base.sha256
    def __getattr__(self,k):return getattr(self.base,k)
    def metadata(self):return self.base.metadata()|dict(runtime_refresh_period_s=self.period,
        refresh_rule='Strong consecutive singlecandidate RGBassociation: cachedfeature<=150ms; ambiguous/lowconfidence/lost/new reset forcesrealReID; weights unchanged.')
    def reset(self):self.last=None
    def __call__(self,rgb,det):
        diag=self.transport.diagnostics if self.transport else {};now=diag.get('sim_time',0.)
        stable=(self.association is not None and self.association.base.current_feature is not None and
            self.detector.lost==0 and self.detector.current_candidate_count==1)
        if self.last and stable and not getattr(self,'force_real',False):
            age=now-self.last['time'];ratio=det.area/max(self.last['area'],1e-12)
            # Independent appearance memory is reused only briefly, after the
            # unchanged image-motion Selector has already admitted a candidate.
            if 0<=age<=self.period and det.confidence>=.8 and .65<=ratio<=1.55:
                self.cached_calls+=1;return self.last['feature'].copy()
        f=self.base(rgb,det);self.real_calls+=1
        self.last=dict(time=now,area=det.area,feature=f.copy());return f
