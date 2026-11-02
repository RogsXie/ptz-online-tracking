"""Explicit changed RGB objective; physical actuation and acceptance gates unchanged."""
import math
from control_zoom_v11.environment import ZoomEnvironment as PhysicalEnvironment
from control_v6.temporal import discount,discounted_seconds

def image_cost(row):
    return max(0.,float(row['center_error'])-.015)+.2*abs(math.log(row['area_ratio']/.04)) if row['visible'] else 2.

def temporal_reward(before,after,action,duration):
    old,new=image_cost(before),image_cost(after)
    good=after['visible'] and after['center_error']<=.025 and .025<=after['area_ratio']<=.055
    rate=(1. if good else -.2)-6*new if after['visible'] else -8.
    factor=discount(duration);shaping=20*(old-factor*new);base=rate*discounted_seconds(duration);effort=-.02 if action else 0.
    return shaping+base+effort,dict(image_progress=shaping,image_rate=rate,duration=duration,discount_factor=factor,discounted_duration_s=discounted_seconds(duration),rate_reward=base,effort=effort,center_and_scale_cost=new,center_inner_band=.015,objective='v14_precision_band_changed_from_v13',rate_quadrature='post-action observed RGB rate held over interval')

class ZoomEnvironment(PhysicalEnvironment):
    def step(self,action,duration):
        before=self.rows[-1];state,_,done,row=super().step(action,duration)
        reward,parts=temporal_reward(before,row,action,duration)
        if not math.isfinite(reward):raise RuntimeError('Nonfinite v14 reward')
        row.update(reward=float(reward),reward_parts=parts)
        return state,float(reward),done,row
