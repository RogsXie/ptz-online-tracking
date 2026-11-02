"""Speed revision: same image task; declared lower bound for angular duration."""
import bootstrap
from bootstrap import HERE
import json,math,time
import numpy as np
from hascam.cfff import CFFF
from integration.transport import ACTION_RATES

SPEC=json.loads((HERE/'protocol.json').read_text())['task']
MIN_RAW=(.3,.01,.01,.01,.01,.3,.3)

def image_cost(row):
    return row['center_error']+.3*abs(math.log(row['area']/.04)) if row['visible'] else 3.

def hold_assessment(rows):
    """Require actual policy-issued Stay for the entire most recent 5 seconds."""
    if not rows or rows[-1]['sim_time']<5:return dict(success=False)
    end=rows[-1]['sim_time'];window=[];elapsed=0.
    for row in reversed(rows[1:]):
        overlap=max(0.,min(row['duration'],row['sim_time']-(end-5.)))
        if overlap<=0:break
        window.append((row,overlap));elapsed+=overlap
    if elapsed<5.-1e-5 or any(not r['visible'] or r['action']!=0 for r,w in window):return dict(success=False)
    errors=np.array([r['center_error'] for r,w in window]);weights=np.array([w for r,w in window])
    order=np.argsort(errors);cdf=np.cumsum(weights[order])/weights.sum()
    p95=float(errors[order][min(len(errors)-1,np.searchsorted(cdf,.95))])
    mean=float(np.average(errors,weights=weights))
    log_mean=float(np.average([abs(math.log(r['area']/.04)) for r,w in window],weights=weights))
    area_ok=all(.025<=r['area']<=.055 for r,w in window)
    return dict(success=bool(mean<=.025 and p95<=.05 and log_mean<=.25 and area_ok and end-5<=45),
        hold_center_mean=mean,hold_center_p95=p95,hold_logscale_mean=log_mean,hold_start_s=end-5.,hold_duration_s=elapsed)

class Environment:
    def __init__(self,t,detector,extractor):
        self.t,self.detector,self.extractor=t,detector,extractor
        self.cfff=CFFF(extractor)

    def observe(self,action=0,duration=0.,reset=False):
        start=time.perf_counter();detections=self.detector(self.t.rgb);detector_s=time.perf_counter()-start
        start=time.perf_counter();selected=self.cfff.choose(self.t.rgb,detections,count_step=False);reid_s=time.perf_counter()-start
        visible=selected is not None
        if visible:
            x,y=[2*c-1 for c in selected.center];area=selected.area;c=selected.confidence
            tx,ty=(0.,0.) if reset or self.forward_x==-2 else (x-self.forward_x,y-self.forward_y)
            state=np.array([x,y,area*100,c,self.forward_action,self.forward_time,tx,ty],np.float32)
        else:
            x=y=-2.;area=None;c=0.;state=np.array([-2,-2,-1,0,0,0,0,0],np.float32)
        if not reset:
            self.forward_x,self.forward_y=x,y
            self.forward_action=action/6. if visible else 0.
            # Original pad_action[index] bug preserved for the source-state profile.
            self.forward_time=(float(action)/5. if action<6 else 0.) if visible else 0.
        row=dict(visible=visible,x=x if visible else None,y=y if visible else None,
            area=area,confidence=c,center_error=math.hypot(x,y) if visible else None,
            bbox=list(selected.bbox) if selected else None,action=int(action),duration=float(duration),
            sim_time=float(self.t.diagnostics['sim_time']),detector_s=detector_s,reid_s=reid_s)
        return state,row

    def reset(self,scene,seed):
        self.t.scenario(scene);self.t.reset(seed);self.cfff.reset()
        self.forward_x=self.forward_y=self.forward_action=self.forward_time=0.
        self.steps=0;self.confidence_streak=0;self.source_condition=False
        state,row=self.observe(reset=True);self.rows=[row];self.previous=row
        return state,row

    def step(self,action,raw_parameter):
        duration=float(np.clip(raw_parameter,MIN_RAW[action],5.))/5.
        p,t,z=ACTION_RATES[action];before=self.t.diagnostics['sim_time']
        start=time.perf_counter();self.t.rpc(f'vset /ptz/step {p} {t} {z} {duration:.12f}')
        communication_s=time.perf_counter()-start;start=time.perf_counter();self.t.capture();capture_s=time.perf_counter()-start
        assert abs(self.t.diagnostics['sim_time']-before-duration)<1e-5
        if self.t.diagnostics.get('demo_geometry_follow') or self.t.diagnostics.get('free_preview'):raise RuntimeError('Non-policy mode')
        state,row=self.observe(action,duration);self.steps+=1
        self.confidence_streak=self.confidence_streak+1 if row['confidence']>.9 else 0
        self.source_condition |= self.confidence_streak>=10
        cost=image_cost(row);inner=bool(row['visible'] and row['center_error']<=.02 and abs(math.log(row['area']/.04))<=.18)
        reward=20*(image_cost(self.previous)-cost)-2*cost+.75*inner-.02*(action!=0)
        row.update(reward=float(reward),raw_parameter=float(raw_parameter),execution_clipped=not MIN_RAW[action]<=raw_parameter<=5.,
            communication_s=communication_s,capture_s=capture_s,source_confidence_condition=self.source_condition)
        self.rows.append(row);self.previous=row
        assessment=hold_assessment(self.rows)
        truncated=row['sim_time']>=50 or self.steps>=512
        done=bool(assessment['success'] or truncated)
        reward+=10. if assessment['success'] and not truncated else -10. if truncated else 0.
        row.update(reward=float(reward),done=done,truncated=truncated,**assessment)
        if truncated:row['success']=False
        return state,float(reward),done,row

