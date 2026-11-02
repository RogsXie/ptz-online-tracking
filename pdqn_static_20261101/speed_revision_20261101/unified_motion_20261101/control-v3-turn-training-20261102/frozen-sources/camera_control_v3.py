"""Camera-only latency-aware scale/edge budgets, from RGB and own optics."""
import math
import numpy as np
from unified_perception import project

PROFILE='precision_v3_zoom_lookahead_edge_priority'


def feedback_horizon(row,duration):
    # Previous completed observation interval and image age are own measured
    # timings. No scenario schedule, route, target GT or speed is read.
    extras=row.get('dynamic_extras',[0.]*7)
    interval=float(extras[3])*.1 if len(extras)>3 else .1
    age=float(row.get('measured_perception_age_s',0.))
    return min(.25,max(.10,min(.25,interval),duration+min(.15,age)))


def safe_acquisition_step(diag,angles,velocity,row,max_step,zoom_rate,tolerance=40.):
    def estimate(step):
        future=dict(diag,fov=float(diag['fov'])-step)
        horizon=feedback_horizon(row,step/zoom_rate)
        xy=np.asarray(project(angles+velocity*horizon,future))
        return (xy-.5)*[960.,540.]
    no_zoom=estimate(0.)
    if max(abs(no_zoom))>tolerance:return 0.,no_zoom
    full=estimate(max_step)
    if max(abs(full))<=tolerance:return float(max_step),full
    lo,hi=0.,max_step
    for _ in range(16):
        mid=(lo+hi)/2
        if max(abs(estimate(mid)))<=tolerance:lo=mid
        else:hi=mid
    return float(lo),estimate(lo)


def edge_axis(error,pixel_velocity,row,valid):
    half=np.array([480.,270.]);horizon=feedback_horizon(row,0.)
    risk=np.maximum(abs(error/half),abs((error+pixel_velocity*horizon)/half))
    axis=int(np.argmax(risk));other=1-axis;actions=[3,4] if axis==0 else [1,2]
    if risk[axis]>.65 and risk[axis]>risk[other]+.15 and any(valid[a] for a in actions):
        return axis,risk.tolist()
    return None,risk.tolist()
