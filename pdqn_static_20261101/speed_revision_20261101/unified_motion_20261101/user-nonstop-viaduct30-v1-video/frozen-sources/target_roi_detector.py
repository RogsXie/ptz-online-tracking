"""Target-aware RGB crop recovery; unrelated confident cars cannot disable it."""
import time,math
import numpy as np
from fast_rgb_perception import Detector as PreviousDetector
from unified_perception import bearing,project
from hascam.detection import Detection
from integration.metrics import iou

class Detector(PreviousDetector):
    def __init__(self,base):
        super().__init__(base);self.roi_calls=0;self.roi_ids=set()
        self.provenance.update(target_aware_recovery=True,
            fast_path='FullRGB each observation; crop recovery when lockedtarget lacks nearby>=.55confidence detection, independent of othervehicleconfidence. RGB/owncamera prediction only.')

    def __call__(self,rgb):
        start=time.perf_counter();d=list(self.base.base(rgb));self.roi_ids=set();self.fast_calls+=1
        if self.track and self.transport:
            diag=self.transport.diagnostics;age=max(0.,diag['sim_time']-self.track['time'])
            predicted=self.track['bearing']+self.track['velocity']*age
            tangent=math.tan(math.radians(diag['fov']/2))**2
            expected=self.track.get('angular_area',self.track['area']*tangent)/max(1e-12,tangent)
            label=self.track.get('class_id',0)
            strong=False
            for x in d:
                delta=bearing(x.center,diag)-predicted;delta[0]=(delta[0]+180)%360-180
                if x.class_id==label and x.confidence>=.55 and np.linalg.norm(delta)<max(.8,2*age) and .4<=x.area/max(expected,1e-12)<=2.5:
                    strong=True;break
            if not strong:
                self.roi_calls+=1;self.recovery_calls+=1
                h,w=rgb.shape[:2];cx,cy=project(predicted,diag)
                aspect=self.track.get('box_aspect',1.4)
                bw=math.sqrt(max(1e-9,expected)*w*h*aspect);bh=bw/max(.1,aspect)
                rw=int(min(w,max(160.,min(640.,max(5*bw,4*bh*w/h)))))
                rh=min(h,int(round(rw*h/w)))
                left=int(np.clip(cx*w-rw/2,0,w-rw));top=int(np.clip(cy*h-rh/2,0,h-rh))
                for x in self.base.base(rgb[top:top+rh,left:left+rw]):
                    b=(np.asarray(x.bbox)*[rw,rh,rw,rh]+[left,top,left,top])/[w,h,w,h]
                    z=Detection(tuple(np.clip(b,0,1)),x.confidence,x.class_id,x.label)
                    d.append(z);self.roi_ids.add(id(z))
        elif not d or max(x.confidence for x in d)<.65:
            self.recovery_calls+=1;d=list(self.base(rgb))
        chosen=[]
        for x in sorted(d,key=lambda x:x.confidence,reverse=True):
            if not any(x.class_id==y.class_id and iou(x.bbox,y.bbox)>.5 for y in chosen):chosen.append(x)
        self.current_candidate_count=len(chosen);self.last_detections=chosen;self.last_seconds=time.perf_counter()-start
        return chosen
