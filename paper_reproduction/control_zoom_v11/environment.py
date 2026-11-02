"""Image-only scale objective and explicit seven-action physical contracts."""
import math
import numpy as np
from control_learning.environment import ControlEnvironment
from control_v6.temporal import discount,discounted_seconds

STATE_FEATURES=['4*normalized_x','4*normalized_y','log(area/.04)/4','confidence_or0','reserved0','reserved0','reserved0','reserved0']


def image_cost(row):
    return float(row['center_error'])+.2*abs(math.log(row['area_ratio']/.04)) if row['visible'] else 2.


def temporal_reward(before,after,action,duration):
    old,new=image_cost(before),image_cost(after)
    good=after['visible'] and after['center_error']<=.025 and .025<=after['area_ratio']<=.055
    rate=(1. if good else -.2)-6*new if after['visible'] else -8.
    factor=discount(duration);shaping=20*(old-factor*new)
    base=rate*discounted_seconds(duration);effort=-.02 if action else 0.
    return shaping+base+effort,dict(image_progress=shaping,image_rate=rate,duration=duration,
        discount_factor=factor,discounted_duration_s=discounted_seconds(duration),rate_reward=base,effort=effort,
        center_and_scale_cost=new,rate_quadrature='post-action observed RGB rate held over interval')


class ZoomState:
    def __init__(self):self.last_seen=np.zeros(3,np.float32)

    def build(self,detection,action=(0,0.)):
        if detection is not None:
            self.last_seen=np.array([2*detection.center[0]-1,2*detection.center[1]-1,
                                    math.log(detection.area/.04)/4],np.float32)
        x,y,area=self.last_seen
        return np.array([4*x,4*y,area,detection.confidence if detection is not None else 0.,0,0,0,0],np.float32)


class ZoomEnvironment(ControlEnvironment):
    def audit_row(self,observation,action,reward=0.,parts=None):
        return super().audit_row(observation,action,reward,parts) | {
            'detector_rgb_scale_fallback':bool(getattr(self.detector,'last_fallback',False))}

    def reset(self,scenario,seed):
        super().reset(scenario,seed)
        self.fixed_location=tuple(self.transport.diagnostics['camera_location'])
        self.transport.mode('ACTIVE_ZOOM_DOUBLE_DQN_V11')
        self.builder=ZoomState()
        self.rows[0]['zoom']=math.log(self.rows[0]['area_ratio']/.04) if self.rows[0]['visible'] else None
        return self.builder.build(self.selected)

    def step(self,action,duration):
        before=self.rows[-1]
        state,_,done,row=super().step(action,duration)
        elapsed=row['sim_time_s']-before['sim_time_s'];expected=list(before['rotation'])
        if action in (1,2):expected[0]=min(-40.,max(-80.,expected[0]+(1 if action==2 else -1)*8*duration))
        if action in (3,4):expected[1]=min(65.,max(-65.,expected[1]+(1 if action==3 else -1)*8*duration))
        expected_fov=min(66.,max(14.,before['fov']+(-1 if action==5 else 1 if action==6 else 0)*8*duration))
        rotation_error=max(abs(a-b) for a,b in zip(expected,row['rotation']))
        fov_error=abs(row['fov']-expected_fov)
        reward,parts=temporal_reward(before,row,action,duration)
        if max(rotation_error,fov_error)>1e-3 or abs(elapsed-duration)>1e-5 or abs(duration-.04)>1e-7 or not math.isfinite(reward):
            raise RuntimeError('Zoom action/time/finite-value contract failed')
        if tuple(self.transport.diagnostics['camera_location'])!=self.fixed_location or not -80.0001<=row['rotation'][0]<=-39.9999:
            raise RuntimeError('Fixed installation/downward pitch contract failed')
        unchanged=max(abs(a-b) for a,b in zip(before['rotation'],row['rotation']))<1e-5 and abs(row['fov']-before['fov'])<1e-5
        row.update(reward=float(reward),reward_parts=parts,actuator_error_deg=max(rotation_error,fov_error),
                   saturated_nonstay=bool(action!=0 and unchanged),terminated=False,truncated=bool(done),
                   bootstrap_from_actual_next_observation=True,
                   zoom=math.log(row['area_ratio']/.04) if row['visible'] else None,
                   magnification=math.tan(math.radians(31))/math.tan(math.radians(row['fov']/2)))
        return state,float(reward),done,row
