"""Post-run timing/GT audit only. This module is never used by control."""
import json,math,statistics,argparse
from pathlib import Path

def rows(p):return [json.loads(s) for s in p.read_text(encoding='utf-8').splitlines()]
def stats(v):
    if not v:return None
    v=sorted(v);return dict(mean=statistics.mean(v),p50=statistics.median(v),p95=v[int(.95*(len(v)-1))],maximum=max(v))
def overlap(a,b):
    if not a or not b:return 0.
    x=max(0,min(a[2],b[2])-max(a[0],b[0]));y=max(0,min(a[3],b[3])-max(a[1],b[1]));z=x*y
    return z/max(1e-12,(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-z)

def analyze(root):
    r=rows(root/'decisions.jsonl');g=rows(root/'offline-diagnostics.jsonl');f=rows(root/'capture-timestamps.jsonl');cfg=json.loads((root/'configuration.json').read_text())
    timed=stats([1000*(b['command_execution']['io_ack_wall_s']-a['observation_capture_request_wall_s']) for a,b in zip(r,r[1:])])
    stop=max(t for t,m in cfg['motion']['speed_profile'] if m==0)
    def moving(t):
        speed=1.
        for boundary,mult in cfg['motion']['speed_profile']:
            if t>=boundary:speed=mult
        return speed>0
    gaps=[1000*max(0,b['command_execution']['native_sim_s']-a['command_execution']['native_sim_s']-a['duration']) for a,b in zip(r,r[1:]) if a['action'] and a['command_execution']['native_sim_s']<stop]
    duty=sum(max(0,min(stop,x['command_execution']['native_sim_s']+x['duration'])-max(0,x['command_execution']['native_sim_s'])) for x in r if x['action'])/stop
    camera=json.loads((root/'scene.json').read_text())['camera'];actual_durations=[];actual_gaps=[]
    for a,b in zip(r,r[1:]):
        if not a['action'] or a['command_execution']['native_sim_s']>=stop:continue
        x,y=a['command_execution'],b['command_execution'];p,t,z=x['executed_rates']
        if p:duration=abs(y['own_rotation'][1]-x['own_rotation'][1])/(abs(p)*camera['pan_deg_s'])
        elif t:duration=abs(y['own_rotation'][0]-x['own_rotation'][0])/(abs(t)*camera['tilt_deg_s'])
        else:duration=abs(y['own_fov']-x['own_fov'])/(abs(z)*camera['zoom_fov_deg_s'])
        duration=min(a['duration'],duration);actual_durations.append(min(duration,max(0,stop-x['native_sim_s'])))
        actual_gaps.append(1000*max(0,y['native_sim_s']-x['native_sim_s']-duration))
    phases={}
    phase_ranges=[('moving',0,stop),('stopped',25,30)]
    if 'crossroad' in cfg['motion']['scenario']:phase_ranges.extend([('turn',8.3333,10.9512),('exit',12,stop)])
    else:phase_ranges.append(('10to18s',10,18))
    for name,lo,hi in phase_ranges:
        rs=[x for x in r if lo<=x['sim_time']<hi and (name!='moving' or moving(x['sim_time']))];matched=[x for x in rs if x['visible'] and overlap(x['bbox'],g[x['observation_frame']].get('gt_bbox_normalized'))>=.3]
        px=[math.hypot(x['x']*480,x['y']*270) for x in matched]
        phases[name]=dict(samples=len(rs),retention=len(matched)/max(1,len(rs)),center_px=stats(px),yellow_fraction=sum(x['center_error']<=.025 for x in matched)/max(1,len(rs)))
    reversals=0;previous={};moves=0
    for x in r:
        a=x['action']
        if not a:continue
        moves+=1;axis=0 if a in (3,4) else 1 if a in (1,2) else 2;last=previous.get(axis)
        if last and last[0]!=a and 0<x['command_execution']['native_sim_s']-last[1]<=.5:reversals+=1
        previous[axis]=(a,x['command_execution']['native_sim_s'])
    cue=feature_response=pan_response=None
    cam=json.loads((root/'scene.json').read_text())['camera']['location_cm']
    bearings=[math.degrees(math.atan2(x['target_location'][1]-cam[1],x['target_location'][0]-cam[0])) for x in g]
    for i in range(2,len(g)-2):
        if 'crossroad' in cfg['motion']['scenario'] and 8<g[i]['sim_time']<12:
            omega=(bearings[i+2]-bearings[i-2])/(g[i+2]['sim_time']-g[i-2]['sim_time'])
            if omega<-.02:cue=g[i]['sim_time'];break
    if cue is not None:
        candidates=[x for x in r if x['sim_time']>=cue and x['image_motion_bearing_rate_deg_s'][0]<-.02]
        if candidates:feature_response=candidates[0]['sim_time']-cue
        candidates=[x for x in r if x['command_execution']['native_sim_s']>=cue and x['command_execution']['executed_rates'][0]<-1e-5]
        if candidates:pan_response=candidates[0]['command_execution']['native_sim_s']-cue
    result=dict(folder=str(root.resolve()),control_hz=len(r)/(f[-1]['wall_s']-f[0]['wall_s']),capture_fps=(len(f)-1)/(f[-1]['wall_s']-f[0]['wall_s']),
        image_request_to_next_action_ack_ms=timed,post_pulse_gap_ms=stats(gaps),nonStay_pulse_duty_before_final_stop=duty,
        actual_camera_motion_duty_before_final_stop=sum(actual_durations)/stop,post_actual_camera_motion_gap_ms=stats(actual_gaps),
        duty_scope='Actual motion duration from consecutive native command own-camera pose/FOV deltas divided by executed physical rate; accounts for mechanical clipping. Final unpaired command excluded. No targetGT used for these timings.',
        stage_ms={k:stats([1000*x[k] for x in r]) for k in ['source_select_s','command_queue_and_rpc_s','feedback_wait_s','observe_total_s','detector_s','reid_s']},
        phases=phases,opposite_corrections_within500ms=reversals,opposite_corrections_per100_moving_commands=100*reversals/max(1,moves),
        turn_world_bearing_reversal_cue_s=cue,estimated_motion_sign_response_after_cue_s=feature_response,first_reverse_pan_after_cue_s=pan_response,
        response_scope='Image request to following native actionack is a measured timing bound, not vehicle-event response. Turn cue from offlineGT bearing: sign<-0.02deg/s. First reversepan may include centering corrections, so descriptive, not causal qualification.',
        estimator_window_s=cfg['motion_estimator_window_s'],frozen=cfg['all_learning_state_frozen'],same_checkpoint=cfg['checkpoint_sha256'])
    (root/'latency-results.json').write_text(json.dumps(result,indent=2),encoding='utf-8');return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('folders',type=Path,nargs='+');a=p.parse_args()
    for folder in a.folders:
        r=analyze(folder);print(json.dumps(dict(folder=folder.name,hz=r['control_hz'],age_ms=r['image_request_to_next_action_ack_ms']['p50'],gap_ms=r['post_actual_camera_motion_gap_ms']['p50'],duty=r['actual_camera_motion_duty_before_final_stop'],turn=r['phases'].get('turn'),moving=r['phases']['moving'],estimator_response=r['estimated_motion_sign_response_after_cue_s'],pan_response=r['first_reverse_pan_after_cue_s'])))
