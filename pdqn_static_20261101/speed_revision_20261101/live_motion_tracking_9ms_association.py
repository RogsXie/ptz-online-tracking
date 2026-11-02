"""9m/s scene with camera-compensated detection association, no GT inputs."""
import speed_bootstrap
import json,math,numpy as np
import live_motion_tracking_9ms_mask as masked
from motion_learning15 import LOW,HIGH,SECONDS

runner=masked.runner;hierarchy=masked.hierarchy

def bearing(center,diag):
    pitch,yaw,_=map(math.radians,diag['rotation']);tan=math.tan(math.radians(diag['fov']/2));x,y=[2*v-1 for v in center]
    f=np.array([math.cos(pitch)*math.cos(yaw),math.cos(pitch)*math.sin(yaw),math.sin(pitch)])
    right=np.array([-math.sin(yaw),math.cos(yaw),0.]);up=np.array([-math.sin(pitch)*math.cos(yaw),-math.sin(pitch)*math.sin(yaw),math.cos(pitch)])
    ray=f+x*tan*right-y*tan*540/960*up
    return np.array([math.degrees(math.atan2(ray[1],ray[0])),math.degrees(math.atan2(ray[2],math.hypot(ray[0],ray[1])))])

class Selector:
    def __init__(self,base,transport):self.base=base;self.t=transport;self.reset()
    def reset(self):self.base.reset();self.last=None;self.info={}
    def choose(self,rgb,detections,count_step=False):
        diag=self.t.diagnostics;now=float(diag['sim_time']);fov=float(diag['fov']);candidates=list(detections);kept=candidates
        if self.last:
            dt=max(0.,now-self.last['time']);expected=self.last['area']*(math.tan(math.radians(self.last['fov']/2))/math.tan(math.radians(fov/2)))**2
            kept=[]
            for detection in candidates:
                ratio=detection.area/max(expected,1e-12);delta=bearing(detection.center,diag)-self.last['bearing'];delta[0]=(delta[0]+180)%360-180
                if .35<=ratio<=3. and float(np.linalg.norm(delta))<=1.5+20.*dt:kept.append(detection)
        selected=self.base.choose(rgb,kept,count_step=count_step)
        self.info=dict(raw_candidates=len(candidates),association_candidates=len(kept),rejected=len(candidates)-len(kept),scope='PreviousRGBboxarea withopticalscalecompensation andcamera-compensated angularassociation; no GT or knownvehiclevelocity')
        if selected:self.last=dict(time=now,area=selected.area,fov=fov,bearing=bearing(selected.center,diag))
        return selected

class Observation(masked.Observation):
    def __init__(self,t,detector,extractor):
        super().__init__(t,detector,extractor);self.cfff=Selector(self.cfff,t)
    def observe(self,*args,**kwargs):
        state,row=super().observe(*args,**kwargs);row['rgb_association']=self.cfff.info;return state,row

def write(path,value):
    if path.name=='configuration.json':
        value.update(detection_association='FrozenRGBdetections/ReID pluspreviousboxarearatio.35..3 afterFOVcompensation andowncamera bearingjump<=1.5+20dtdegrees. No GT,knownpathor truevelocity used. Confidence scoresunchanged.')
    return masked.write(path,value)

if __name__=='__main__':
    runner.create_agent=masked.create_agent;runner.resume=hierarchy.resume;runner.SpeedEnvironment=Observation
    runner.LOW=LOW;runner.HIGH=HIGH;runner.SECONDS_PER_RAW=SECONDS
    runner.layout=masked.fastscene.scene.layout;runner.materialize=masked.fastscene.materialize
    runner.AtomicTransport=hierarchy.Transport;runner.DisplayTransport=hierarchy.DisplayTransport
    runner.write=write;runner.main()
