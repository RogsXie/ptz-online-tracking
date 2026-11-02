"""File-only release-video audit; no simulator commands and no training."""
import speed_bootstrap
import argparse,hashlib,json,math
from pathlib import Path
import cv2,numpy as np,torch
from independent_validation import weighted_quantile
from author_aligned.source import create_agent,verify_source

def lines(p):return [json.loads(s) for s in p.read_text().splitlines()]
def average(a):return sum(v*w for v,w in a)/sum(w for v,w in a) if a else None

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('folder',type=Path);args=p.parse_args();root=args.folder
    torch.set_num_threads(2);base=json.loads((root/'independent-live-results.json').read_text())
    cfg=json.loads((root/'configuration.json').read_text());initial=json.loads((root/'initial-observation.json').read_text())
    rows=lines(root/'decisions.jsonl');frames=lines(root/'capture-timestamps.jsonl');diag=lines(root/'offline-diagnostics.jsonl')
    checkpoint=Path(cfg['checkpoint']);digest=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert digest==cfg['checkpoint_sha256'];saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    a=create_agent(20261101,'cpu')
    for name in ('actor_net','param_net'):getattr(a,name).load_state_dict(saved[name])
    with torch.inference_mode():
        parameters=a.param_net(torch.tensor([r['input_state'] for r in rows],dtype=torch.float32))
        q=a.actor_net(torch.tensor([r['input_state'] for r in rows],dtype=torch.float32),parameters)
    parameter_error=float((parameters-torch.tensor([r['all_parameters'] for r in rows])).abs().max())
    action_mismatch=int((q.argmax(1)!=torch.tensor([r['action'] for r in rows])).sum())
    state_chain=rows[0]['input_state']==initial['state'] and all(x['next_state']==y['input_state'] for x,y in zip(rows,rows[1:]))
    duration_ok=all(abs(min(cfg['raw_highs'][r['action']],max(cfg['raw_lows'][r['action']],r['raw_parameter']))*cfg['seconds_per_raw'][r['action']]-r['duration'])<1e-10 and r['raw_parameter']==r['all_parameters'][r['action']] for r in rows)
    from integration.transport import ACTION_RATES
    physical_errors=[]
    for row in rows:
        pre=max((f for f in frames if f['wall_s']<=row['command_wall_s']),key=lambda f:f['wall_s'])
        post=min(frames,key=lambda f:abs(f['sim_time']-row['sim_time']))
        if post['remaining']>1e-6:continue
        pan,tilt,zoom=ACTION_RATES[row['action']];pr,tr,zr=cfg['rates'];dt=row['duration']
        expected=[np.clip(pre['rotation'][0]+tilt*tr*dt,-80.,-40.),np.clip(pre['rotation'][1]+pan*pr*dt,-65.,65.),np.clip(pre['fov']-zoom*zr*dt,14.,66.)]
        physical_errors.extend(abs(a-b) for a,b in zip(expected,[post['rotation'][0],post['rotation'][1],post['fov']]))
    physical_error=max(physical_errors,default=float('inf'))
    rates_ok=all([d['resolved_config']['camera']['pan_deg_s'],d['resolved_config']['camera']['tilt_deg_s'],d['resolved_config']['camera']['zoom_fov_deg_s']]==cfg['rates'] for d in diag)
    intervals=[];before=frames[0]['sim_time'];loss_run=max_loss=0.;reversals=rapid=0;previous={};settling=None
    before_row=initial
    for row in rows:
        dt=row['sim_time']-before;intervals.append((before,row['sim_time'],row));before=row['sim_time']
        if row['visible']:loss_run=0.
        else:loss_run+=dt;max_loss=max(max_loss,loss_run)
        action=row['action']
        if action:
            axis=0 if action in (1,2) else 1 if action in (3,4) else 2
            if axis in previous and previous[axis][0]!=action:
                reversals+=1;field='y' if axis==0 else 'x' if axis==1 else None
                if field and before_row['visible'] and row['visible']:
                    if intervals[-1][0]-previous[axis][1]<=.5 and abs(before_row[field])<=.03 and abs(row[field])-abs(before_row[field])>=.02:rapid+=1
            previous[axis]=(action,intervals[-1][0])
        before_row=row
        if settling is None and row['sim_time']-frames[0]['sim_time']>=5.:
            end=row['sim_time'];window=[(r,max(0.,min(hi,end)-max(lo,end-5.))) for lo,hi,r in intervals]
            window=[(r,w) for r,w in window if w>0]
            if sum(w for r,w in window)>=5.-1e-5 and all(r['visible'] and r['action']==0 and .025<=r['area']<=.055 for r,w in window):
                errors=[(r['center_error'],w) for r,w in window];scales=[(abs(math.log(r['area']/.04)),w) for r,w in window]
                if average(errors)<=.025 and weighted_quantile(errors,.95)<=.05 and average(scales)<=.25:settling=end-5.-frames[0]['sim_time']
    wall=frames[-1]['wall_s']-frames[0]['wall_s'];sim=frames[-1]['sim_time']-frames[0]['sim_time']
    clocks=all(b['wall_s']>x['wall_s'] and b['sim_time']>x['sim_time'] for x,b in zip(frames,frames[1:]))
    visibility=sum(bool(d['gt_in_frame']) for d in diag)/len(diag)
    checks=base['checks']|dict(initial_full_detection=bool(initial['visible'] and all(.003<v<.997 for v in initial['bbox'])),
        initial_gt_full_box=bool(diag[0]['gt_bbox_normalized'] and all(0<=v<=1 for v in diag[0]['gt_bbox_normalized'])),
        source_verified=bool(verify_source()),physical_rate_configuration=rates_ok,actual_commanded_motion=physical_error<.003,original_discrete_outputs=action_mismatch==0,
        original_continuous_outputs=parameter_error<3e-4,state_chain=state_chain,exact_selected_duration=duration_ok,
        settling_by45seconds=settling is not None and settling<=45.,maximum_unintended_loss=max_loss<=.25+1e-6,
        reversal_frequency=reversals/max(sim,1e-12)*10<=4.+1e-6,rapid_unwarranted_reversals=rapid<=2,
        raw_frame_gt_visibility=visibility>=.99,monotonic_clocks=clocks,wall_ue_clock_agreement=abs(sim-wall)<=max(.1,.01*wall))
    video=cv2.VideoCapture(str(root/'live-sensor-vfr.mp4'));encoded_frames=int(video.get(cv2.CAP_PROP_FRAME_COUNT));fps=video.get(cv2.CAP_PROP_FPS);video.release()
    encoded_duration=encoded_frames/fps if fps else 0.;checks['encoded_duration_matches_capture']=abs(encoded_duration-wall)<=.2
    timing=json.loads((root/'ue-motion-timing.json').read_text());ticks=timing['motion_tick_intervals_seconds']
    report=dict(passed=all(checks.values()),checks=checks,case=cfg['case'],checkpoint_sha256=digest,
        scope='Fresh paired development-case real-wall capture; multi-case independent qualification is separate',
        center_mean=base['center_mean'],center_p95=base['center_p95'],area_mean=base['area_mean'],settling_time_s=settling,
        matched_availability=base['matched_availability'],max_loss_s=max_loss,reversals=reversals,rapid_unwarranted_reversals=rapid,
        source_action_mismatches=action_mismatch,continuous_output_max_cpu_gpu_difference=parameter_error,maximum_physical_command_error_degrees=physical_error,
        first_zoom_wall_s=next((r['command_wall_s'] for r in rows if r['action'] in (5,6)),None),
        raw_captured_frames=len(frames),encoded_frames=encoded_frames,wall_duration_s=wall,simulation_duration_s=sim,
        encoded_duration_s=encoded_duration,raw_gt_visible_fraction=visibility,
        ue_motion_ticks=len(ticks),ue_tick_median_s=float(np.median(ticks)),ue_tick_p95_s=float(np.quantile(ticks,.95)),
        action_counts={str(i):sum(r['action']==i for r in rows) for i in range(7)})
    (root/'independent-speed-live-results.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2));assert report['passed'],'Realtime release evidence failed; preserve and inspect'

if __name__=='__main__':main()
