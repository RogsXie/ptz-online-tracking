"""Focused disjoint procedural check; frozen policy and no replay insertion.

Synchronous same-delay comparison of RGB control quality, never an FPS test.
The original reference branch uses its unchanged observation contract.
"""
import speed_bootstrap
from speed_bootstrap import configure_precision_project
import argparse,json,time,warnings
from collections import Counter
from pathlib import Path
import cv2,numpy as np,torch
from threadpoolctl import threadpool_limits
import unified_core as core
import precision_policy_v2 as precision
import live_unified_motion as demo
import live_crossroad_robust_tracker as scenario
from precision_training_scenes import scene
from target_roi_detector import Detector
from control_latency_runtime import SharedTransport
from stream_arena import StreamArena
from integration.transport import ACTION_RATES
from train import exclusive,snapshot,finish_evaluation
from calibrate import write
from train_precision_from_scratch import advance_observed
from integration.metrics import iou


def postrun_identity(records):
    """Offline scoring only, called after a case's controls have finished."""
    matches=Counter();initial_identity=None;wrong=unknown=0
    for record in records:
        box=record['bbox']
        if box is None:continue
        candidates=[(iou(box,t.get('gt_bbox_clipped_normalized')),t['id'])
            for t in record['targets'] if t.get('gt_bbox_clipped_normalized')]
        if not candidates:unknown+=1;continue
        overlap,identity=max(candidates)
        if overlap<.3:unknown+=1;continue
        if initial_identity is None:initial_identity=identity
        matches[identity]+=1;wrong+=int(identity!=initial_identity)
    return dict(initial_selected_identity=initial_identity,visible_matches=dict(matches),
        visible_matches_other_identity=wrong,unmatched_visible=unknown,usage='Post-run diagnostics only; never in policy or action filtering.')


def postrun_camera(records,identity):
    if identity is None:return dict(target=None,source_frames=0)
    errors=[];outside=undefined=center_outside=0
    for record in records[1:]:
        target=next(t for t in record['targets'] if t['id']==identity)
        outside+=int(not target['gt_in_frame']);box=target.get('gt_bbox_normalized')
        if box is None:undefined+=1;continue
        center_outside+=int(not(0.<=(box[0]+box[2])/2<=1. and 0.<=(box[1]+box[3])/2<=1.))
        errors.append(float(np.hypot((box[0]+box[2]-1)*480.,(box[1]+box[3]-1)*270.)))
    return dict(target=identity,source_frames=len(records)-1,target_out_of_view=outside,
        target_center_out_of_view=center_outside,
        undefined_projection=undefined,center_median_p95_px=np.percentile(errors,[50,95]).tolist() if errors else None,
        scoring='All completed-command native source images, including no-box/occlusion. GT post-run only.')


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--split',choices=['development','check'],default='development');p.add_argument('--steps',type=int,default=96)
    p.add_argument('--cases',type=int,nargs='+',choices=range(8),default=list(range(8)))
    p.add_argument('--control-v3',action='store_true')
    args=p.parse_args();exclusive();configure_precision_project();warnings.filterwarnings('ignore');torch.set_num_threads(4);cv2.setNumThreads(1)
    root=args.output.resolve();root.mkdir(exist_ok=False)
    profile=torch.load(args.checkpoint,map_location='cpu',weights_only=False)['metadata']['motion_profile']
    if profile==precision.PROFILE:
        a=precision.load_for_evaluation(args.checkpoint);observation=precision.observation_class(scenario.Observation)
        core.LOW,core.HIGH=precision.LOW,precision.HIGH
    else:
        a=demo.create_agent();core.resume(a,args.checkpoint);observation=scenario.Observation
    if args.control_v3:
        if profile!=precision.PROFILE:raise ValueError('Control-v3 requires precision observation contract')
        a.precision_control_v3=True
    a.epsilon=0.;frozen=snapshot(a)
    d,e=scenario.previous.previous.previous.previous.frontend();d=Detector(d.base)
    cases=[]
    with StreamArena(root,headless=True) as arena:
        t=SharedTransport(arena.port)
        try:
            env=observation(t,d,e)
            for i in args.cases:
                folder=root/f'case-{i:02d}';path,item=scene(folder,i,args.split)
                delay=[.04,.08,.12][i%3];cycle=[.12,.16,.20,.24][(i//3)%4]
                env.precision_training_age=delay;state,row=env.reset(path,item['seed'])
                distances=[];areas=[];misses=overrides=0
                diagnostic_records=[dict(bbox=row['bbox'],targets=t.diagnostics.get('targets',[]))]
                with (folder/'steps.jsonl').open('w',encoding='utf-8',buffering=1) as log:
                    for k in range(args.steps):
                        core.bind_camera(a,env)
                        action,selected,params=a.choose_action(state,train=True);proposal=int(action)
                        requested=float(np.clip(selected[0],core.LOW[action],core.HIGH[action]))*core.SECONDS[action]
                        action,duration,guard=scenario.action_filter(env,a,state,int(action),requested,params)
                        duration=float(np.clip(duration,core.LOW[action]*core.SECONDS[action],core.HIGH[action]*core.SECONDS[action]))
                        g,zg=core.gain(float(t.diagnostics['fov']))
                        if getattr(t,'pan_tilt_gain_override',None) is not None:g=float(t.pan_tilt_gain_override)
                        if getattr(t,'zoom_gain_override',None) is not None:zg=float(t.zoom_gain_override)
                        pan,tilt,zoom=ACTION_RATES[action]
                        advance_observed(env,delay)
                        advance_observed(env,duration,(pan*g,tilt*g,zoom*zg))
                        idle=max(0.,cycle-delay-duration)
                        if idle>=.001:advance_observed(env,idle)
                        t.capture();state,row=env.observe(action,duration)
                        diagnostic_records.append(dict(bbox=row['bbox'],targets=t.diagnostics.get('targets',[])))
                        overrides+=int(proposal!=action or abs(duration-requested)>1e-7)
                        if row['visible']:
                            distances.append(float(np.hypot(row['x']*480.,row['y']*270.)));areas.append(row['area'])
                        else:misses+=1
                        log.write(json.dumps(dict(step=k,proposal_action=proposal,executed_action=action,duration_s=duration,guard=guard,observation=row))+'\n')
                identity_score=postrun_identity(diagnostic_records)
                result=dict(family=item['family'],steps=args.steps,visible=args.steps-misses,missing=misses,
                    median_p95_center_px=np.percentile(distances,[50,95]).tolist() if distances else None,
                    within_8px=sum(x<=8 for x in distances),area_above_2pct=sum(x>.02 for x in areas),override_count=overrides,
                    maximum_target_generation=env.cfff.info.get('target_generation',0),postrun_identity=identity_score,
                    camera_control=postrun_camera(diagnostic_records,identity_score['initial_selected_identity']))
                write(folder/'offline-identity-diagnostics.json',diagnostic_records)
                cases.append(result);write(root/'progress.json',dict(cases=cases));print('FROZEN CHECK',result,flush=True)
        finally:t.close()
    finish_evaluation(a,frozen)
    write(root/'summary.json',dict(checkpoint=str(args.checkpoint),motion_profile=profile,split=args.split,cases=cases,
        camera_control_profile='precision_v3_zoom_lookahead_edge_priority' if getattr(a,'precision_control_v3',False) else 'precision_v2',
        learning_state_unchanged=True,scope='Eight held-out procedural layouts, not broad qualification; fixed-clock data collection is not an online FPS benchmark.'))


if __name__=='__main__':
    with threadpool_limits(limits=1,user_api='blas'):main()
