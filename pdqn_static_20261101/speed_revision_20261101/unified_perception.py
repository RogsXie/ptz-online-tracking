"""Frozen RGB detector with foveated small-target detection and association."""
import speed_bootstrap
import math,time
from collections import deque
import numpy as np
from hascam.detection import Detection
from integration.metrics import iou

def bearing(center,diag):
    pitch,yaw,_=map(math.radians,diag['rotation']);tan=math.tan(math.radians(diag['fov']/2));x,y=[2*v-1 for v in center]
    f=np.array([math.cos(pitch)*math.cos(yaw),math.cos(pitch)*math.sin(yaw),math.sin(pitch)])
    right=np.array([-math.sin(yaw),math.cos(yaw),0.]);up=np.array([-math.sin(pitch)*math.cos(yaw),-math.sin(pitch)*math.sin(yaw),math.cos(pitch)])
    ray=f+x*tan*right-y*tan*540/960*up
    return np.array([math.degrees(math.atan2(ray[1],ray[0])),math.degrees(math.atan2(ray[2],math.hypot(ray[0],ray[1])))])

def project(angles,diag):
    by,bp=map(math.radians,angles);pitch,yaw,_=map(math.radians,diag['rotation']);tan=math.tan(math.radians(diag['fov']/2))
    ray=np.array([math.cos(bp)*math.cos(by),math.cos(bp)*math.sin(by),math.sin(bp)])
    f=np.array([math.cos(pitch)*math.cos(yaw),math.cos(pitch)*math.sin(yaw),math.sin(pitch)])
    right=np.array([-math.sin(yaw),math.cos(yaw),0.]);up=np.array([-math.sin(pitch)*math.cos(yaw),-math.sin(pitch)*math.sin(yaw),math.cos(pitch)])
    depth=float(ray@f)
    if depth<=0:return (.5,.5)
    return ((ray@right/depth/tan+1)/2,(-ray@up/depth/(tan*540/960)+1)/2)

class FoveatedDetector:
    def __init__(self,base):
        self.base=base;self.provenance=base.provenance|dict(preprocessing_profile='full_half_and_RGB_history_fovea2x_v1',fovea='RGB/camera predictedROI, coldquadrants, no GT or scenevelocity')
        self.transport=None;self.track=None;self.lost=0;self.last_seconds=0.
    def __getattr__(self,key):return getattr(self.base,key)
    def reset_tracking(self):self.track=None;self.lost=0
    def __call__(self,rgb):
        start=time.perf_counter();h,w=rgb.shape[:2];hh,ww=h//2,w//2;candidates=list(self.base(rgb));crops=[]
        if self.track and self.transport:
            diag=self.transport.diagnostics;dt=max(0.,float(diag['sim_time'])-self.track['time']);angles=self.track['bearing']+self.track['velocity']*dt
            cx,cy=project(angles,diag)
            if self.track['area']<.015 or self.lost:
                x=int(np.clip(cx*w-ww/2,0,w-ww));y=int(np.clip(cy*h-hh/2,0,h-hh));crops.append((x,y))
        if self.lost>=5:crops.extend([(0,0),(w-ww,0),(0,h-hh),(w-ww,h-hh)])
        for x,y in dict.fromkeys(crops):
            for detection in self.base.base(rgb[y:y+hh,x:x+ww]):
                b=(np.asarray(detection.bbox)*[ww,hh,ww,hh]+[x,y,x,y])/[w,h,w,h]
                candidates.append(Detection(tuple(np.clip(b,0,1)),detection.confidence,detection.class_id,detection.label))
        chosen=[]
        for d in sorted(candidates,key=lambda d:d.confidence,reverse=True):
            if not any(d.class_id==old.class_id and iou(d.bbox,old.bbox)>.5 for old in chosen):chosen.append(d)
        self.last_seconds=time.perf_counter()-start;return chosen

class Selector:
    def __init__(self,base,t,detector):self.base=base;self.t=t;self.detector=detector;detector.transport=t;self.reset()
    def reset(self):self.base.reset();self.history=deque();self.info={};self.detector.reset_tracking()
    def choose(self,rgb,detections,count_step=False):
        diag=self.t.diagnostics;now=float(diag['sim_time']);fov=float(diag['fov']);candidates=list(detections);chosen=candidates;velocity=np.zeros(2)
        if self.history:
            while len(self.history)>1 and now-self.history[0]['time']>.7:self.history.popleft()
            last=self.history[-1];times=np.array([v['time'] for v in self.history]);angles=np.stack([v['bearing'] for v in self.history])
            if len(times)>=3 and times[-1]-times[0]>=.12:
                dt=times-times.mean();velocity=np.clip((dt[:,None]*(angles-angles.mean(0))).sum(0)/max(1e-12,float((dt*dt).sum())),-30.,30.)
            age=max(0.,now-last['time']);predicted=last['bearing']+velocity*age
            angular_area=float(np.median([v['angular_area'] for v in self.history]));expected=angular_area/max(1e-12,math.tan(math.radians(fov/2))**2);ranked=[]
            for d in candidates:
                a=bearing(d.center,diag);delta=a-predicted;delta[0]=(delta[0]+180)%360-180;distance=float(np.linalg.norm(delta));ratio=d.area/max(expected,1e-12)
                if .5<=ratio<=2. and distance<=max(2.5,1.+15*age):
                    score=distance/.5+abs(math.log(ratio))+.2*(1-d.confidence);ranked.append((score,d))
            chosen=[min(ranked,key=lambda x:x[0])[1]] if ranked else []
        selected=self.base.choose(rgb,chosen,count_step=count_step)
        if selected:
            angle=bearing(selected.center,diag);entry=dict(time=now,bearing=angle,angular_area=selected.area*math.tan(math.radians(fov/2))**2)
            self.history.append(entry);self.detector.track=dict(time=now,bearing=angle,velocity=velocity,area=selected.area);self.detector.lost=0
        else:self.detector.lost+=1
        self.info=dict(raw_candidates=len(candidates),association_candidates=len(chosen),lost=self.detector.lost,scope='PureRGBhistory,owncameraoptics andmotionassociation; noGT/path/speedinputs')
        return selected
