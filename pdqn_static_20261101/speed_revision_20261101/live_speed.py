"""Fresh real-wall control at selected calibrated units; no trajectory replay."""
import speed_bootstrap
from speed_bootstrap import REV,configure_precision_project
import argparse,hashlib,json,time,warnings
from pathlib import Path
import cv2,numpy as np,torch
from speed_runner_v4 import create_agent,resume,frontend
from author_aligned.source import create_agent as old_agent
from author_aligned.environment import AtomicTransport
from environment import Environment as OldEnvironment
from speed_environment_v3 import Environment as SpeedEnvironment,LOW,HIGH,SECONDS_PER_RAW
from integration.runtime import StandaloneArena
from integration.transport import ACTION_RATES
from speed_scenes import layout,materialize
from train import resume as old_resume,exclusive,snapshot,finish_evaluation
from calibrate import write
from live_demo import encode
from live_demo_visible import DisplayTransport,display_command

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--index',type=int,default=7);p.add_argument('--old-baseline',action='store_true')
    p.add_argument('--visible',action='store_true');p.add_argument('--seconds',type=float,default=50.)
    args=p.parse_args();exclusive();configure_precision_project();warnings.filterwarnings('ignore');torch.set_num_threads(4)
    root=args.output.resolve();root.mkdir(exist_ok=False);(root/'frames').mkdir()
    detector,extractor=frontend()
    if args.old_baseline:a=old_agent(20261101,'cuda');old_resume(a,args.checkpoint);cls=OldEnvironment
    else:a=create_agent();resume(a,args.checkpoint);cls=SpeedEnvironment
    frozen=snapshot(a);a.epsilon=0.;frames=[];item=layout('development',args.index)
    if args.old_baseline:item['execution_profile']='released'
    scene=materialize(root,item)
    config=dict(checkpoint=str(args.checkpoint.resolve()),checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        case=item,detector=detector.provenance,reid=extractor.metadata(),actual_realtime_Ue_motion=True,
        policy='Direct source P-DQN four networks, epsilon0, learning frozen',no_pose_replay=True,
        all_learning_state_frozen=False,execution_profile='released' if args.old_baseline else 'paper25_calibrated_units_stability_margin_v4',
        raw_lows=[.3]*7 if args.old_baseline else list(LOW),raw_highs=[5.]*7 if args.old_baseline else list(HIGH),
        seconds_per_raw=[.2]*7 if args.old_baseline else list(SECONDS_PER_RAW),rates=[4.,2.,5.] if args.old_baseline else [25.,25.,5.],
        video='Actual RGB captures, original wall timestamps, no frame replication or acceleration',requested_wall_s=args.seconds)
    write(root/'configuration.json',config)
    with StandaloneArena(root,headless=not args.visible) as arena:
        t=(DisplayTransport if args.visible else AtomicTransport)(arena.port,hud=False)
        try:
            env=cls(t,detector,extractor);state,initial=env.reset(scene,item['seed'])
            write(root/'initial-observation.json',dict(state=state.tolist(),**initial))
            if args.visible:t.rpc(display_command(initial))
            t.rpc('vset /ptz/realtime 1');t.mode('PTZ_AUTOMATIC_CENTER_AND_ZOOM');started=time.perf_counter()
            with (root/'capture-timestamps.jsonl').open('w',encoding='utf-8',buffering=1) as frame_log, \
                 (root/'offline-diagnostics.jsonl').open('w',encoding='utf-8',buffering=1) as gt_log, \
                 (root/'decisions.jsonl').open('w',encoding='utf-8',buffering=1) as decision_log:
                def capture():
                    t.capture();wall=time.perf_counter()-started
                    assert t.diagnostics['realtime_control'] and not t.diagnostics['demo_geometry_follow'] and not t.diagnostics['free_preview']
                    index=len(frames)
                    cv2.imwrite(str(root/f'frames/{index:05d}.jpg'),cv2.cvtColor(t.rgb,cv2.COLOR_RGB2BGR),[cv2.IMWRITE_JPEG_QUALITY,95])
                    r=dict(frame=index,wall_s=wall,sim_time=t.diagnostics['sim_time'],rotation=t.diagnostics['rotation'],fov=t.diagnostics['fov'],remaining=t.diagnostics['action_remaining_seconds'])
                    frames.append(r);frame_log.write(json.dumps(r)+'\n');gt_log.write(json.dumps(t.diagnostics)+'\n')
                capture()
                while time.perf_counter()-started<args.seconds:
                    if (root/'STOP_REQUEST').exists():raise InterruptedError('Requested action-boundary live stop')
                    input_state=state.copy();select_start=time.perf_counter()
                    act,param,all_params=a.choose_action(state,train=True);act=int(act)
                    raw=float(param[0]);duration=float(np.clip(raw,config['raw_lows'][act],config['raw_highs'][act]))*config['seconds_per_raw'][act]
                    pan,tilt,zoom=ACTION_RATES[act];command_wall=time.perf_counter()-started
                    t.rpc(f'vset /ptz/play {pan} {tilt} {zoom} {duration:.12f}')
                    while True:
                        capture()
                        if t.diagnostics['action_remaining_seconds']<=1e-6 or time.perf_counter()-started>=args.seconds:break
                    state,row=env.observe(act,duration)
                    if args.visible:t.rpc(display_command(row))
                    row.update(action=act,raw_parameter=raw,duration=duration,command_wall_s=command_wall,
                        observation_wall_s=frames[-1]['wall_s'],decision_done_wall_s=time.perf_counter()-started,
                        selection_and_execution_wall_s=time.perf_counter()-select_start,all_parameters=all_params.tolist(),
                        input_state=input_state.tolist(),next_state=state.tolist())
                    decision_log.write(json.dumps(row)+'\n')
                write(root/'ue-motion-timing.json',t.rpc('vget /ptz/timing'));t.rpc('vset /ptz/realtime 0')
        finally:t.close()
    finish_evaluation(a,frozen);config.update(all_learning_state_frozen=True,frame_count=len(frames),wall_duration_s=frames[-1]['wall_s']-frames[0]['wall_s'])
    write(root/'configuration.json',config);encode(root,frames);print('FRESH REALTIME',root,flush=True)

if __name__=='__main__':main()
