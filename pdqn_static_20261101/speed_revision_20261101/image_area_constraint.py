"""Explicit user-requested RGB area cap; pan/tilt retain frozen neural selection."""
import math
import numpy as np
import torch
import unified_core as core

LIMIT=.02
SETTLE=.0185
MAX_ZOOM_PULSE=.16
WIDE_VIEW_HOLD=2.

def field_for_area(fov,area,target=SETTLE):
    return math.degrees(2*math.atan(math.tan(math.radians(fov/2))*math.sqrt(area/target)))

def filter_action(env,agent,state,action,duration,params):
    row=env.latest_rgb_observation
    fov=float(env.t.diagnostics['fov'])
    maximum=float(env.camera_limits['fov_max_deg'])
    minimum=float(env.camera_limits['fov_min_deg'])
    zoom_rate=float(env.camera_limits['zoom_fov_deg_s'])
    env.t.zoom_gain_override=None
    boxes=list(row.get('area_guard_candidates',[]))
    if row['visible']:boxes.append(dict(area=row['area'],label=row.get('target_label')))
    largest=max(boxes,key=lambda x:x['area']) if boxes else None
    area=largest['area'] if largest else None
    info=dict(pdqn_action=int(action),pdqn_duration_s=float(duration),input_area=area,
        selected_input_area=row['area'],reliable_vehicle_count=len(row.get('area_guard_candidates',[])),
        input_label=largest['label'] if largest else None,input_fov_deg=fov,area_limit=LIMIT,settle_area=SETTLE,
        observation_sim_time=row['sim_time'],reason='neural_action',zoom_gain_override=None)
    desired=field_for_area(fov,area) if largest else maximum
    desired=min(maximum,max(minimum,desired))
    now=float(row['sim_time'])
    if now<getattr(env,'wide_view_hold_until',-1.):
        desired=max(desired,min(maximum,env.wide_view_fov_floor))
        info['wide_view_hold_until']=env.wide_view_hold_until
    if (not largest or area>LIMIT) and fov<desired-1e-4:
        action=6
        duration=max(.001,min(MAX_ZOOM_PULSE,(desired-fov)/zoom_rate))
        env.t.zoom_gain_override=1.
        env.wide_view_hold_until=now+WIDE_VIEW_HOLD
        env.wide_view_fov_floor=min(maximum,fov+zoom_rate*duration)
        info.update(reason='area_exceeded_zoom_out' if largest else 'lost_expand_view',
                    desired_fov_deg=desired,zoom_gain_override=1.,wide_view_hold_until=env.wide_view_hold_until)
    elif action==5:
        # Even a learned zoom-in action must leave at least the requested view.
        budget=max(0.,fov-desired)
        seconds=budget/(zoom_rate*core.gain(fov)[1])
        edge=bool(row['visible'] and any([row['bbox'][0]<.01,row['bbox'][1]<.01,row['bbox'][2]>.99,row['bbox'][3]>.99]))
        if not row['visible'] or seconds<.001 or edge:
            valid=core.valid_actions(env.t.diagnostics,env.camera_limits)
            valid[5]=False
            with torch.inference_mode():
                q=agent.actor_net(torch.as_tensor(state,dtype=torch.float32,device=agent.device).unsqueeze(0),
                    torch.as_tensor(params,dtype=torch.float32,device=agent.device).unsqueeze(0))[0].cpu().numpy()
            q[~valid]=-np.inf
            action=int(q.argmax())
            duration=float(np.clip(params[action],core.LOW[action],core.HIGH[action]))*core.SECONDS[action]
            info.update(reason='zoom_in_blocked_at_area_limit',desired_fov_deg=desired)
        elif duration>seconds:
            duration=seconds
            info.update(reason='zoom_in_duration_limited',desired_fov_deg=desired)
    info.update(executed_action=int(action),executed_duration_s=float(duration))
    return action,duration,info
