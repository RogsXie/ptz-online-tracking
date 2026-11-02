"""Locked-target scale/search rules and remaining-error bounds on neural motion."""
import math
import numpy as np,torch
import unified_core as core
from unified_perception import bearing,project
from image_area_constraint import field_for_area
from integration.transport import ACTION_RATES

AREA_LIMIT=.02
AREA_SETTLE=.0185
SEARCH_STEP=6.
STARTUP_AREA=.016
STARTUP_CENTER_PX=40.

def filter_action(env,agent,state,action,duration,params):
    row=env.latest_rgb_observation;diag=env.t.diagnostics
    fov=float(diag['fov']);limits=env.camera_limits
    low,high=float(limits['fov_min_deg']),float(limits['fov_max_deg'])
    zoom_rate=float(limits['zoom_fov_deg_s']);now=float(row['sim_time'])
    env.t.zoom_gain_override=None
    info=dict(pdqn_action=int(action),pdqn_duration_s=float(duration),input_area=row['area'],
        selected_input_area=row['area'],input_fov_deg=fov,input_label=row.get('target_label'),
        observation_sim_time=now,area_limit=AREA_LIMIT,zoom_gain_override=None)
    generation=env.cfff.info.get('target_generation',0)
    if getattr(env,'startup_target_generation',None)!=generation or now<getattr(env,'startup_last_time',-1.):
        env.startup_target_generation=generation
        env.startup_scale_pending=True
    env.startup_last_time=now
    info.update(startup_target_area=STARTUP_AREA,startup_center_tolerance_px=STARTUP_CENTER_PX,
        startup_scale_pending=env.startup_scale_pending)
    def finish(a,d,reason,**fields):
        info.update(reason=reason,executed_action=int(a),executed_duration_s=float(d),
            zoom_gain_override=env.t.zoom_gain_override,**fields)
        return int(a),float(d),info
    if not row['visible']:
        env.lock_searching=True
        env.lock_recovering=False
        if fov<high-1e-4:
            env.t.zoom_gain_override=1.
            step=min(SEARCH_STEP,high-fov)
            return finish(6,max(.001,step/zoom_rate),'locked_missing_progressive_zoom',phase='search',desired_fov_deg=fov+step)
        return finish(0,.025,'locked_missing_hold_camera',phase='search')
    recovered=bool(getattr(env,'lock_searching',False))
    if recovered:
        env.lock_searching=False;env.lock_recovering=True
    info['just_recovered']=recovered
    phase='recover' if getattr(env,'lock_recovering',False) else 'track'
    area=float(row['area'])
    if area>=STARTUP_AREA*.9 or fov<=low+1e-4:
        env.startup_scale_pending=False
    if phase=='track' and env.startup_scale_pending:phase='acquire'
    info['startup_scale_pending']=env.startup_scale_pending
    # Single-target mode: distractors never force more zoom after recovery.
    if area>AREA_LIMIT and fov<high-1e-4:
        wanted=min(high,field_for_area(fov,area,AREA_SETTLE))
        env.t.zoom_gain_override=1.
        return finish(6,max(.001,min(SEARCH_STEP,wanted-fov)/zoom_rate),'locked_area_zoom_out',phase=phase,desired_fov_deg=wanted)
    error=np.asarray(state[:2],dtype=float)*[480.,270.]
    center=(np.asarray(state[:2],dtype=float)+1)/2
    angles=bearing(center,diag)
    angular_velocity=np.asarray(row.get('image_motion_bearing_rate_deg_s',[0.,0.]))
    future=np.asarray(project(angles+angular_velocity*.01,diag))
    target_pixel_velocity=(future-center)*[960.,540.]/.01
    jac=[]
    for rotation_index,image_axis in [(1,0),(0,1)]:
        perturbed=dict(diag,rotation=list(diag['rotation']))
        perturbed['rotation'][rotation_index]+=.01
        jac.append((project(angles,perturbed)[image_axis]-center[image_axis])*[960.,540.][image_axis]/.01)
    gain,_=core.gain(fov)
    speeds=[float(limits['pan_deg_s'])*gain,float(limits['tilt_deg_s'])*gain]
    valid=core.valid_actions(diag,limits)
    budgets={}
    for a,axis,sign in [(3,0,1),(4,0,-1),(2,1,1),(1,1,-1)]:
        camera_rate=jac[axis]*sign*speeds[axis]
        previous_sign=getattr(env,'last_pan_sign' if axis==0 else 'last_tilt_sign',0)
        deadband=6. if previous_sign and sign!=previous_sign else 4.
        closing=max(.25*abs(camera_rate),abs(camera_rate)-np.sign(error[axis])*target_pixel_velocity[axis])
        budget=.85*abs(error[axis])/max(1e-6,closing)
        budgets[a]=budget
        if abs(error[axis])<=deadband or error[axis]*camera_rate>=0 or budget<.001:
            valid[a]=False
    wanted=max(low,min(high,field_for_area(fov,area,AREA_SETTLE)))
    inward=max(0.,fov-wanted)
    edge=any([row['bbox'][0]<.01,row['bbox'][1]<.01,row['bbox'][2]>.99,row['bbox'][3]>.99])
    valid[5]&=bool(inward/(zoom_rate*core.gain(fov)[1])>=.001 and not edge)
    valid[6]&=bool(area>AREA_SETTLE)
    center_tolerance=STARTUP_CENTER_PX if phase=='acquire' else 10.
    urgent=bool(np.max(np.abs(error))>center_tolerance and any(valid[a] for a in [1,2,3,4]))
    if urgent:
        valid[[0,5,6]]=False
    # First acquisition actively restores a useful target scale. A moderate
    # centre tolerance prevents moving targets from starving zoom indefinitely;
    # each exclusive pulse is bounded and followed by fresh RGB feedback.
    if phase=='acquire' and np.max(np.abs(error))<=STARTUP_CENTER_PX and not edge:
        desired=max(low,field_for_area(fov,area,STARTUP_AREA))
        step=min(SEARCH_STEP,max(0.,fov-desired))
        if step>=.1:
            env.t.zoom_gain_override=1.
            return finish(5,max(.001,step/zoom_rate),'locked_startup_zoom_in',phase=phase,
                predicted_error_px=error.tolist(),desired_fov_deg=desired)
    # Recover scale immediately once the locked target is centred. Never wait
    # for a wide-view timer or another vehicle's area to permit zoom-in.
    if phase=='recover' and not urgent:
        if area<.012 and fov>low+1e-4 and not edge:
            desired=max(low,field_for_area(fov,area,.016))
            step=min(SEARCH_STEP,max(0.,fov-desired))
            if step>=.1:
                env.t.zoom_gain_override=1.
                return finish(5,max(.001,step/zoom_rate),'locked_recovery_zoom_in',phase=phase,
                    predicted_error_px=error.tolist(),desired_fov_deg=desired)
        env.lock_recovering=False;phase='track'
    original=action
    if not valid[action]:
        with torch.inference_mode():
            q=agent.actor_net(torch.as_tensor(state,dtype=torch.float32,device=agent.device).unsqueeze(0),
                torch.as_tensor(params,dtype=torch.float32,device=agent.device).unsqueeze(0))[0].cpu().numpy()
        q[~valid]=-np.inf;action=int(q.argmax())
        duration=float(np.clip(params[action],core.LOW[action],core.HIGH[action]))*core.SECONDS[action]
    reason='neural_action' if action==original else 'neural_action_masked'
    if action in budgets:
        capped=min(duration,budgets[action])
        if capped<duration:reason='remaining_error_duration_cap'
        duration=capped
        if action in [3,4]:env.last_pan_sign=1 if action==3 else -1
        else:env.last_tilt_sign=1 if action==2 else -1
    elif action==5:
        budget=inward/(zoom_rate*core.gain(fov)[1])
        if duration>budget:duration=budget;reason='zoom_in_area_duration_cap'
    return finish(action,max(.001,duration),reason,phase=phase,predicted_error_px=error.tolist(),
        motion_budgets_s={str(k):float(v) for k,v in budgets.items()})
