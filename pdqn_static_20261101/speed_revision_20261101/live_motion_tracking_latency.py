"""Image-motion state prediction over measured sensing latency; no motor rule."""
import speed_bootstrap
import math,numpy as np
import live_motion_tracking15 as hierarchy
from motion_learning15 import LOW,HIGH,SECONDS

runner=hierarchy.runner

class Observation(hierarchy.Observation):
    def observe(self,*args,**kwargs):
        state,row=super().observe(*args,**kwargs)
        if hierarchy.STAGE=='tracking' and row['visible']:
            diag=self.t.diagnostics;pitch,yaw,_=map(math.radians,diag['rotation']);tan=math.tan(math.radians(diag['fov']/2))
            f=np.array([math.cos(pitch)*math.cos(yaw),math.cos(pitch)*math.sin(yaw),math.sin(pitch)])
            right=np.array([-math.sin(yaw),math.cos(yaw),0.]);up=np.array([-math.sin(pitch)*math.cos(yaw),-math.sin(pitch)*math.sin(yaw),math.cos(pitch)])
            ray=f+row['x']*tan*right-row['y']*tan*540/960*up
            by=math.atan2(ray[1],ray[0]);bp=math.atan2(ray[2],math.hypot(ray[0],ray[1]))
            age=min(.15,max(0.,row['detector_s']+row['reid_s']+float(diag.get('capture_readback_seconds',0.))+.003))
            vy,vp=map(math.radians,row['image_motion_bearing_rate_deg_s']);by+=vy*age;bp+=vp*age
            predicted=np.array([math.cos(bp)*math.cos(by),math.cos(bp)*math.sin(by),math.sin(bp)])
            depth=float(predicted@f)
            if depth>0:
                actual=state.copy();state=state.copy()
                state[0]=float(np.clip((predicted@right)/depth/tan,-1.5,1.5))
                state[1]=float(np.clip(-(predicted@up)/depth/(tan*540/960),-1.5,1.5))
                row.update(observed_state=actual.tolist(),latency_compensated_state=state.tolist(),measured_perception_age_s=age,
                    prediction_scope='Targetbearing extrapolated only overmeasuredcapture/detector/ReIDlatency usingRGB-derived angularvelocity andowncameraoptics. Motoractions remain neural outputs.')
        return state,row

def write(path,value):
    if path.name=='configuration.json':
        value.update(perception_latency_compensation='Onlytrackinginputxy predictedovermeasuredRGBreadback+detector+ReID+3ms interfaceage, capped150ms; trueobservedrowxy andsameframeyellowcolor unchanged.',
            no_feedforward_motor_control=True)
    return hierarchy.write(path,value)

if __name__=='__main__':
    runner.create_agent=hierarchy.create_agent;runner.resume=hierarchy.resume;runner.SpeedEnvironment=Observation
    runner.LOW=LOW;runner.HIGH=HIGH;runner.SECONDS_PER_RAW=SECONDS
    runner.layout=hierarchy.scene.layout;runner.materialize=hierarchy.scene.materialize
    runner.AtomicTransport=hierarchy.Transport;runner.DisplayTransport=hierarchy.DisplayTransport
    runner.write=write;runner.main()
