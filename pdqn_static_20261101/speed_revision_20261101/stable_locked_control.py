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
    control_v3=bool(getattr(agent,'precision_control_v3',False))
    centering_v4=bool(getattr(agent,'precision_centering_v4',False))
    startup_center=STARTUP_CENTER_PX
    if centering_v4:
        from camera_centering_v4 import ACQUIRE_CENTER_PX
        startup_center=ACQUIRE_CENTER_PX
    env.t.zoom_gain_override=None
    env.t.pan_tilt_gain_override=None
    info=dict(pdqn_action=int(action),pdqn_duration_s=float(duration),input_area=row['area'],
        selected_input_area=row['area'],input_fov_deg=fov,input_label=row.get('target_label'),
        observation_sim_time=now,area_limit=AREA_LIMIT,zoom_gain_override=None)
    generation=env.cfff.info.get('target_generation',0)
    if getattr(env,'startup_target_generation',None)!=generation or now<getattr(env,'startup_last_time',-1.):
        env.startup_target_generation=generation
        env.startup_scale_pending=True
        env.centering_v4_last_axis=None
        env.centering_v4_axis_times={0:now,1:now}
    env.startup_last_time=now
    info.update(startup_target_area=STARTUP_AREA,startup_center_tolerance_px=startup_center,
        startup_scale_pending=env.startup_scale_pending)
    def finish(a,d,reason,**fields):
        if centering_v4 and a in (1,2,3,4):
            env.centering_v4_last_axis=0 if a in (3,4) else 1
            env.centering_v4_axis_times[env.centering_v4_last_axis]=now
        info.update(reason=reason,executed_action=int(a),executed_duration_s=float(d),
            zoom_gain_override=env.t.zoom_gain_override,pan_tilt_gain_override=env.t.pan_tilt_gain_override,**fields)
        return int(a),float(d),info
    if not row['visible']:
        env.lock_searching=True
        env.lock_recovering=False
        if control_v3:
            from camera_control_v3 import search_motion
            search=search_motion(env,row,diag,limits)
            if search is not None:
                a=search.pop('action');d=search.pop('duration')
                env.t.pan_tilt_gain_override=search.pop('pan_gain');env.t.zoom_gain_override=search.pop('zoom_gain')
                return finish(a,d,'locked_rgb_motion_search',phase='search',no_predicted_detection=True,**search)
        # Widen enough to search, but preserve a detectable image size for a
        # far target. The estimate uses only its last visible angular area.
        angular_area=getattr(env,'last_visible_angular_area',None)
        search_max=min(high,max(fov,math.degrees(2*math.atan(math.sqrt(angular_area/.0015))))) if angular_area else high
        if fov<search_max-1e-4:
            env.t.zoom_gain_override=1.
            step=min(SEARCH_STEP,search_max-fov)
            return finish(6,max(.001,step/zoom_rate),'locked_missing_progressive_zoom',phase='search',desired_fov_deg=fov+step,search_fov_limit_deg=search_max)
        return finish(0,.025,'locked_missing_hold_camera',phase='search')
    recovered=bool(getattr(env,'lock_searching',False))
    if recovered:
        env.lock_searching=False;env.lock_recovering=True
    info['just_recovered']=recovered
    phase='recover' if getattr(env,'lock_recovering',False) else 'track'
    area=float(row['area'])
    env.last_visible_angular_area=area*math.tan(math.radians(fov/2))**2
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
    # Per-command history can contain fewer than3samples at low command rates.
    # Use the actual RGB tracker's per-frame motion estimate for the motor
    # governor; keep the neural15state/per-command history unchanged.
    tracked_velocity=env.cfff.info.get('locked_bearing_velocity_deg_s')
    angular_velocity=np.asarray(tracked_velocity if tracked_velocity is not None else row.get('image_motion_bearing_rate_deg_s',[0.,0.]))
    if control_v3:
        env.v3_last_visible=dict(time=now,generation=generation,
            bearing=bearing(((row['x']+1)/2,(row['y']+1)/2),diag).tolist(),velocity=angular_velocity.tolist())
        env.v3_previous_search_action=0
    future=np.asarray(project(angles+angular_velocity*.01,diag))
    target_pixel_velocity=(future-center)*[960.,540.]/.01
    jac=[]
    for rotation_index,image_axis in [(1,0),(0,1)]:
        perturbed=dict(diag,rotation=list(diag['rotation']))
        perturbed['rotation'][rotation_index]+=.01
        jac.append((project(angles,perturbed)[image_axis]-center[image_axis])*[960.,540.][image_axis]/.01)
    gain,_=core.gain(fov)
    motion_speed=float(np.max(np.abs(angular_velocity)))
    motion_floor=.18 if motion_speed>.8 else .10 if motion_speed>.35 else 0.
    if getattr(agent,'precision_motion_budget',False) and motion_speed>.35:
        # Fresh-policy curriculum only. Compensate motor duty lost to image/
        # feedback waits while retaining remaining-error pulse bounds below.
        required=max(abs(float(angular_velocity[0]))/float(limits['pan_deg_s']),
            abs(float(angular_velocity[1]))/float(limits['tilt_deg_s']))
        motion_floor=max(motion_floor,min(1.,2.8*required+.03))
    if motion_floor>gain:
        gain=motion_floor;env.t.pan_tilt_gain_override=gain
    speeds=[float(limits['pan_deg_s'])*gain,float(limits['tilt_deg_s'])*gain]
    valid=core.valid_actions(diag,limits)
    pulse_target_error=error
    if centering_v4:
        from camera_centering_v4 import pulse_error
        from camera_control_v3 import feedback_horizon
        pulse_target_error=pulse_error(error,target_pixel_velocity,feedback_horizon(row,0.))
        info['pulse_target_error_px']=pulse_target_error.tolist()
    budgets={}
    for a,axis,sign in [(3,0,1),(4,0,-1),(2,1,1),(1,1,-1)]:
        camera_rate=jac[axis]*sign*speeds[axis]
        previous_sign=getattr(env,'last_pan_sign' if axis==0 else 'last_tilt_sign',0)
        deadband=6. if previous_sign and sign!=previous_sign else 4.
        closing=max(.25*abs(camera_rate),abs(camera_rate)-np.sign(error[axis])*target_pixel_velocity[axis])
        budget=(.90 if centering_v4 else .85)*abs(pulse_target_error[axis])/max(1e-6,closing)
        budgets[a]=budget
        if abs(error[axis])<=deadband or error[axis]*camera_rate>=0 or budget<.001:
            valid[a]=False
    wanted=max(low,min(high,field_for_area(fov,area,AREA_SETTLE)))
    inward=max(0.,fov-wanted)
    edge=any([row['bbox'][0]<.01,row['bbox'][1]<.01,row['bbox'][2]>.99,row['bbox'][3]>.99])
    valid[5]&=bool(inward/(zoom_rate*core.gain(fov)[1])>=.001 and not edge)
    valid[6]&=bool(area>AREA_SETTLE)
    safe_zoom_step=None
    if control_v3 and phase=='acquire' and not edge:
        from camera_control_v3 import safe_acquisition_step
        desired=max(low,field_for_area(fov,area,STARTUP_AREA))
        requested_step=min(SEARCH_STEP,max(0.,fov-desired))
        zoom_tolerance=40.
        if centering_v4:
            from camera_centering_v4 import ZOOM_FEEDBACK_TOLERANCE_PX
            zoom_tolerance=ZOOM_FEEDBACK_TOLERANCE_PX
        safe_zoom_step,predicted_zoom_error=safe_acquisition_step(diag,angles,angular_velocity,row,requested_step,zoom_rate,tolerance=zoom_tolerance)
        info.update(acquisition_requested_zoom_step_deg=requested_step,safe_acquisition_zoom_step_deg=safe_zoom_step,
            after_zoom_feedback_error_px=predicted_zoom_error.tolist())
        if safe_zoom_step<.1:valid[5]=False
    center_tolerance=startup_center if phase=='acquire' and (not control_v3 or safe_zoom_step is None or safe_zoom_step>=.1) else 10.
    urgent=bool(np.max(np.abs(error))>center_tolerance and any(valid[a] for a in [1,2,3,4]))
    if urgent:
        valid[[0,5,6]]=False
        # Avoid repeatedly spending feedback cycles on an almost-correct axis
        # while a fast target is escaping along the other one.
        major=int(np.argmax(np.abs(error)));minor=1-major
        if abs(error[major])>max(20.,3.*abs(error[minor])) and any(valid[a] for a in ([3,4] if major==0 else [1,2])):
            valid[[1,2] if minor==1 else [3,4]]=False
    if control_v3:
        from camera_control_v3 import edge_axis
        axis,risk=edge_axis(error,target_pixel_velocity,row,valid)
        info['predicted_edge_risk_xy']=risk
        if axis is not None:
            valid[[0,5,6]]=False
            valid[[1,2] if axis==0 else [3,4]]=False
            urgent=True
            info['edge_priority_axis']='pan' if axis==0 else 'tilt'
    if centering_v4 and urgent and 'edge_priority_axis' not in info:
        from camera_centering_v4 import preferred_axis
        priority=preferred_axis(error,valid,getattr(env,'centering_v4_last_axis',None),
            getattr(env,'centering_v4_axis_times',None),now)
        if priority is not None:
            valid[[1,2] if priority==0 else [3,4]]=False
            info['centering_priority_axis']='pan' if priority==0 else 'tilt'
    # First acquisition actively restores a useful target scale. A moderate
    # centre tolerance prevents moving targets from starving zoom indefinitely;
    # each exclusive pulse is bounded and followed by fresh RGB feedback.
    if phase=='acquire' and np.max(np.abs(error))<=startup_center and not edge:
        desired=max(low,field_for_area(fov,area,STARTUP_AREA))
        step=min(SEARCH_STEP,max(0.,fov-desired))
        if control_v3 and safe_zoom_step is not None:step=min(step,safe_zoom_step)
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
        axis=0 if action in (3,4) else 1
        if centering_v4 and phase=='acquire' and abs(error[axis])>60.:
            floor=min(.1,float(core.HIGH[action])*core.SECONDS[action],budgets[action])
            if duration<floor:
                duration=floor;reason='acquisition_bounded_motor_floor'
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
