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


def search_motion(env,row,diag,limits):
    """Short locked-target search from last real RGB motion, camera commands only.

    Does not create a detection or replace the locked identity. At the existing
    3s memory boundary the old estimate is abandoned. Mechanical limits and a
    six-degree pulse bound apply to every command.
    """
    import unified_core as core
    memory=getattr(env,'v3_last_visible',None)
    association=env.cfff.info
    if not memory or association.get('locked_track_id') is None:return None
    if memory['generation']!=association.get('target_generation',0):return None
    age=float(row['sim_time'])-memory['time']
    if not 0.<=age<3.:return None
    velocity=np.asarray(memory['velocity'],float)
    if np.max(abs(velocity))<2.:return None
    elapsed=min(3.,age+float(row.get('measured_perception_age_s',0.))+.10)
    goal=np.asarray(memory['bearing'],float)+velocity*elapsed
    pitch,yaw,_=diag['rotation']
    delta=np.array([(goal[0]-yaw+180.)%360.-180.,goal[1]-pitch])
    point=np.asarray(project(goal,diag));risk=abs(2*point-1)
    valid=core.valid_actions(diag,limits);actions=[3 if delta[0]>0 else 4,2 if delta[1]>0 else 1]
    axes=[i for i in range(2) if abs(delta[i])>.3 and valid[actions[i]]]
    fov=float(diag['fov']);maximum=float(limits['fov_max_deg'])
    previous=getattr(env,'v3_previous_search_action',0)
    if fov<maximum-.05 and (max(risk)<.55 or previous in (1,2,3,4) or not axes):
        action=6;duration=min(6.,maximum-fov)/float(limits['zoom_fov_deg_s']);pan_gain=None;zoom_gain=1.
    elif axes:
        axis=max(axes,key=lambda i:abs(delta[i])/(33. if i==0 else 19.))
        action=actions[axis];rate=float(limits['pan_deg_s' if axis==0 else 'tilt_deg_s'])
        duration=max(.001,min(6.,.85*abs(delta[axis]))/rate);pan_gain=1.;zoom_gain=None
    else:return None
    env.v3_previous_search_action=action
    return dict(action=action,duration=duration,pan_gain=pan_gain,zoom_gain=zoom_gain,
        last_real_rgb_age_s=age,last_rgb_angular_velocity_deg_s=velocity.tolist(),
        predicted_search_bearing_deg=goal.tolist(),search_delta_deg=delta.tolist(),search_fov_limit_deg=maximum)
