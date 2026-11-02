"""Independent 30fps actual RGB recording and frozen image-based P-DQN."""
import speed_bootstrap
from speed_bootstrap import REV,configure_precision_project
import argparse,json,time,hashlib,warnings,math
from pathlib import Path
import numpy as np,torch
import live_variable_speed_requested as scenario
import live_unified_motion as demo
import unified_core as core
from unified_perception import bearing,project
from async_frame_runtime import FrameRuntime,SnapshotTransport
from integration.runtime import StandaloneArena
from integration.transport import ACTION_RATES
from train import exclusive,snapshot,finish_evaluation
from calibrate import write
from encode_precise_vfr import encode

class Observation(core.Observation):
    def observe(self,*args,**kwargs):
        state,row=super().observe(*args,**kwargs)
        if demo.AGENT is not None:
            demo.AGENT.camera_diagnostics=self.t.diagnostics;demo.AGENT.camera_limits=self.camera_limits
        if row['visible']:
            diag=self.t.diagnostics;angles=bearing(((row['x']+1)/2,(row['y']+1)/2),diag)
            sample_age=time.perf_counter()-self.t.frame.received if self.t.frame else row['detector_s']+row['reid_s']
            age=min(.2,max(0.,sample_age+float(diag.get('capture_readback_seconds',0.))+.003))
            xy=project(angles+np.asarray(row['image_motion_bearing_rate_deg_s'])*age,diag)
            actual=state.copy();state=state.copy();state[:2]=np.clip(2*np.asarray(xy)-1,-1.5,1.5)
            row.update(observed_state=actual.tolist(),latency_compensated_state=state.tolist(),measured_perception_age_s=age)
        return state,row

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--seconds',type=float,default=30.);p.add_argument('--fps',type=float,default=30.)
    p.add_argument('--checkpoint',type=Path,default=core.ROOT/'pdqn_training_v1/last.pt');args=p.parse_args()
    exclusive();configure_precision_project();warnings.filterwarnings('ignore');torch.set_num_threads(4)
    root=args.output.resolve();root.mkdir(exist_ok=False);(root/'frames').mkdir()
    detector,extractor=core.frontend();a=demo.create_agent();core.resume(a,args.checkpoint);frozen=snapshot(a);a.epsilon=0.
    demo.CASE='known9';item=demo.layout('development',7);scene=scenario.materialize(root,item)
    config=dict(checkpoint=str(args.checkpoint.resolve()),checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),case=item,detector=detector.provenance,reid=extractor.metadata(),
        motion=demo.MOTION,architecture='Single frozen15input original P-DQN, four networks; same checkpoint as synchronous demonstration.',execution_profile=core.PROFILE,
        target_capture_fps=args.fps,async_capture=True,single_UnrealCV_connection=True,latest_frame_only_for_control=True,writer_thread=True,
        policy_input='Pinned immutable RGB/own-camera snapshot; same motion features and optical gain; sample-age prediction capped200ms.',
        actual_realtime_Ue_motion=True,no_pose_replay=True,no_video_acceleration=True,all_learning_state_frozen=False,requested_wall_s=args.seconds)
    write(root/'configuration.json',config);runtime=None
    with StandaloneArena(root,headless=True) as arena:
        t=SnapshotTransport(arena.port)
        try:
            env=Observation(t,detector,extractor);state,initial=env.reset(scene,item['seed']);write(root/'initial-observation.json',dict(state=state.tolist(),**initial))
            t.rpc('vset /ptz/realtime 1');runtime=FrameRuntime(t.raw,root,args.fps);t.runtime=runtime;runtime.start(args.seconds)
            with (root/'decisions.jsonl').open('w',encoding='utf-8',buffering=1) as log:
                while time.perf_counter()-runtime.started<args.seconds:
                    runtime.check()
                    if (root/'STOP_REQUEST').exists():break
                    if args.seconds-(time.perf_counter()-runtime.started)<.25:break
                    inputs=state.copy();select_start=time.perf_counter();act,param,params=a.choose_action(state,train=True);act=int(act)
                    raw=float(param[0]);duration=float(np.clip(raw,core.LOW[act],core.HIGH[act]))*core.SECONDS[act];p,tlt,z=ACTION_RATES[act]
                    command_wall=time.perf_counter()-runtime.started;t.rpc(f'vset /ptz/play {p} {tlt} {z} {duration:.12f}')
                    after=runtime.latest.sequence if runtime.latest else -1
                    frame=runtime.wait_frame(after,command_done=True);t.pin(frame)
                    state,row=env.observe(act,duration)
                    row.update(action=act,raw_parameter=raw,duration=duration,command_wall_s=command_wall,observation_wall_s=frame.wall_s,
                        decision_done_wall_s=time.perf_counter()-runtime.started,selection_and_execution_wall_s=time.perf_counter()-select_start,
                        all_parameters=params.tolist(),input_state=inputs.tolist(),next_state=state.tolist(),observation_frame=frame.sequence,
                        sample_buffer_age_at_inference_end_s=time.perf_counter()-frame.received)
                    log.write(json.dumps(row)+'\n')
            # Let the independent recorder finish its requested wall window.
            # Do not begin a final action whose feedback would be beyond it.
            while time.perf_counter()-runtime.started<args.seconds:
                runtime.check();time.sleep(.005)
            runtime.stop();t.runtime=None;write(root/'ue-motion-timing.json',t.rpc('vget /ptz/timing'));t.rpc('vset /ptz/realtime 0')
        finally:
            if runtime and runtime.io_thread.is_alive():runtime.stop()
            t.close()
    finish_evaluation(a,frozen);config.update(all_learning_state_frozen=True,frame_count=len(runtime.frames),wall_duration_s=runtime.frames[-1]['wall_s']-runtime.frames[0]['wall_s'],missed_capture_deadlines=runtime.missed_deadlines)
    write(root/'configuration.json',config);encode(root,runtime.frames,name='live-ue-realtime-vfr.mp4');print('ASYNC REALTIME COMPLETE',root,flush=True)
if __name__=='__main__':main()
