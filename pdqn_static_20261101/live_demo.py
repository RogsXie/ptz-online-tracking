"""Capture a new real-time UE rollout. No pose replay or reconstructed motion."""
import bootstrap
from bootstrap import HERE,REPRO
import argparse,hashlib,json,subprocess,sys,time,warnings
from pathlib import Path
import cv2,numpy as np,torch
from author_aligned.environment import AtomicTransport
from author_aligned.source import create_agent
from control_model_v13.detector import create_detector
from environment import Environment
from integration.experiment import make_extractor
from integration.runtime import StandaloneArena
from integration.transport import ACTION_RATES
from scenes import materialize
from evaluation_scenes import layout
from train import resume,exclusive,snapshot,finish_evaluation
from calibrate import write

def encode(folder,frames):
    # VFR durations come from actual capture wall timestamps, not action duration replication.
    tools=REPRO/'verification/value-redesign-v10-20261031/video-tools'
    sys.path.insert(0,str(tools))
    from imageio_ffmpeg import get_ffmpeg_exe
    manifest=folder/'frames.ffconcat'
    lines=['ffconcat version 1.0']
    for i,row in enumerate(frames):
        lines.append(f"file 'frames/{i:05d}.jpg'")
        if i+1<len(frames):lines.append(f"duration {frames[i+1]['wall_s']-row['wall_s']:.9f}")
    manifest.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    startup=subprocess.STARTUPINFO();startup.dwFlags|=subprocess.STARTF_USESHOWWINDOW;startup.wShowWindow=0
    result=subprocess.run([get_ffmpeg_exe(),'-y','-loglevel','error','-safe','1','-f','concat','-i',str(manifest),
        '-vsync','vfr','-c:v','libx264','-crf','18','-preset','fast','-threads','2','-pix_fmt','yuv420p',
        '-movflags','+faststart',str(folder/'live-sensor-vfr.mp4')],startupinfo=startup,capture_output=True)
    if result.returncode:raise RuntimeError(result.stderr.decode(errors='replace'))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--index',type=int,default=7)
    p.add_argument('--split',choices=['development','validation'],default='development')
    p.add_argument('--visible',action='store_true',help='Show the actual UE camera window')
    p.add_argument('--seconds',type=float,default=50.);args=p.parse_args()
    exclusive();warnings.filterwarnings('ignore');torch.set_num_threads(4)
    root=args.output.resolve();root.mkdir(exist_ok=False);(root/'frames').mkdir()
    detector=create_detector(checkpoint=REPRO/'weights/yolo26s_ue_overhead_v1.pt',device='cuda',threshold=.15,imgsz=960)
    detector.warmup();extractor=make_extractor('cuda');agent=create_agent(20261101,'cuda');resume(agent,args.checkpoint)
    frozen=snapshot(agent);agent.epsilon=0.
    frames=[];decisions=[];item=layout(args.split,args.index);scene=materialize(root,item)
    with StandaloneArena(root,headless=not args.visible) as arena:
        t=AtomicTransport(arena.port,hud=False)
        try:
            env=Environment(t,detector,extractor);state,initial=env.reset(scene,item['seed'])
            write(root/'initial-observation.json',dict(state=state.tolist(),**initial))
            t.rpc('vset /ptz/realtime 1');t.mode('PTZ_AUTOMATIC_CENTER_AND_ZOOM');started=time.perf_counter()
            def capture():
                t.capture();wall=time.perf_counter()-started
                assert t.diagnostics['realtime_control'] and not t.diagnostics['demo_geometry_follow']
                index=len(frames)
                cv2.imwrite(str(root/f'frames/{index:05d}.jpg'),cv2.cvtColor(t.rgb,cv2.COLOR_RGB2BGR),[cv2.IMWRITE_JPEG_QUALITY,95])
                row=dict(frame=index,wall_s=wall,sim_time=t.diagnostics['sim_time'],
                    rotation=t.diagnostics['rotation'],fov=t.diagnostics['fov'],remaining=t.diagnostics['action_remaining_seconds'])
                frames.append(row);frame_log.write(json.dumps(row)+'\n')
                # Strictly separate offline diagnostics. No branch or selection reads this file.
                gt_log.write(json.dumps(t.diagnostics)+'\n')
            with (root/'capture-timestamps.jsonl').open('w',buffering=1) as frame_log, \
                 (root/'offline-diagnostics.jsonl').open('w',buffering=1) as gt_log, \
                 (root/'decisions.jsonl').open('w',buffering=1) as decision_log:
                capture()
                while time.perf_counter()-started<args.seconds:
                    if (root/'STOP_REQUEST').exists():break
                    input_state=state.copy()
                    select_start=time.perf_counter();act,param,all_params=agent.choose_action(state,train=True)
                    duration=float(np.clip(param[0],.3,5.))/5.;pan,tilt,zoom=ACTION_RATES[int(act)]
                    reply=t.rpc(f'vset /ptz/play {pan} {tilt} {zoom} {duration:.12f}')
                    while True:
                        capture()
                        if t.diagnostics['action_remaining_seconds']<=1e-6:break
                        if time.perf_counter()-started>=args.seconds:break
                    state,row=env.observe(int(act),duration)
                    if args.visible:
                        box=row['bbox'] or [0,0,0,0]
                        t.rpc('vset /ptz/detection '+' '.join(f'{v:.12f}' for v in box)+f" {int(row['visible'])}")
                    row.update(action=int(act),raw_parameter=float(param[0]),duration=duration,
                        selection_and_execution_wall_s=time.perf_counter()-select_start,all_parameters=all_params.tolist(),
                        input_state=input_state.tolist(),next_state=state.tolist())
                    decisions.append(row);decision_log.write(json.dumps(row)+'\n')
                timing=t.rpc('vget /ptz/timing');t.rpc('vset /ptz/realtime 0')
                write(root/'ue-motion-timing.json',timing)
        finally:t.close()
    finish_evaluation(agent,frozen)
    write(root/'configuration.json',dict(checkpoint=str(args.checkpoint),case=item,
        checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        detector=detector.provenance,reid=extractor.metadata(),
        actual_realtime_Ue_motion=True,policy='original author P-DQN networks, epsilon=0; geometric task training',
        no_pose_replay=True,all_learning_state_frozen=True,
        video='Actual captured RGB frames, variable frame durations derived from wall timestamps',
        frame_count=len(frames),wall_duration_s=frames[-1]['wall_s']-frames[0]['wall_s']))
    encode(root,frames)
    print('REALTIME CAPTURE',root,flush=True)

if __name__=='__main__':main()
