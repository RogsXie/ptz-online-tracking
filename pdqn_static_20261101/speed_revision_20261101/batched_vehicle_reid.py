"""Same frozen FastReID crops/weights, evaluated together without feature caching."""
import math,time
import cv2,numpy as np,torch
from torch.nn import functional as F

class BatchFeatures:
    def __init__(self,periodic,chunk=8):
        self.periodic=periodic;self.extractor=getattr(periodic,'base',periodic);self.chunk=chunk
        self.calls=0;self.features=0;self.total_s=0.

    def __call__(self,rgb,detections):
        if not detections:return []
        start=time.perf_counter();h,w=rgb.shape[:2];images=[]
        for d in detections:
            x1,y1,x2,y2=d.bbox
            left,top=max(0,int(x1*w)),max(0,int(y1*h))
            right,bottom=min(w,max(left+1,math.ceil(x2*w))),min(h,max(top+1,math.ceil(y2*h)))
            crop=rgb[top:bottom,left:right]
            if not crop.size:raise ValueError('Empty detector crop')
            image=cv2.resize(crop,(256,256),interpolation=cv2.INTER_CUBIC)
            images.append(image.astype(np.float32).transpose(2,0,1))
        output=[]
        for i in range(0,len(images),self.chunk):
            tensor=torch.as_tensor(np.stack(images[i:i+self.chunk]),device=self.extractor.device)
            with torch.inference_mode():features=F.normalize(self.extractor.model(tensor),dim=1).cpu().numpy()
            if features.shape[1]!=2048 or not np.isfinite(features).all():raise RuntimeError('Invalid batched ReID')
            output.extend(features)
        if hasattr(self.periodic,'real_calls'):self.periodic.real_calls+=len(output)
        self.calls+=1;self.features+=len(output);self.total_s+=time.perf_counter()-start
        return output
