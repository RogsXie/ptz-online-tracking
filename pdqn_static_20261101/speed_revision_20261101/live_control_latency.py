"""Paired actual end-to-end latency experiment; action completion unchanged."""
import speed_bootstrap
from speed_bootstrap import configure_precision_project,REV
import argparse,json,time,hashlib,warnings,shutil
from pathlib import Path
import numpy as np,torch
import unified_core as core
import live_unified_motion as demo
import live_crossroad_turn as cross
import live_variable_speed_requested as variable
from live_async_variable_speed import Observation
from control_latency_runtime import Runtime,Transport,SharedTransport
from fast_rgb_perception import Detector,PeriodicReID
from integration.runtime import StandaloneArena
from integration.transport import ACTION_RATES
from train import exclusive,snapshot,finish_evaluation
from calibrate import write
from encode_precise_vfr import encode

ACTION_FILTER=None

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--seconds',type=float,default=30.);p.add_argument('--fps',type=float,default=30.)
    p.add_argument('--fast',action='store_true');p.add_argument('--scene',choices=['crossroad','variable'],default='crossroad')
    p.add_argument('--shm',action='store_true')
    p.add_argument('--motion-window',type=float,default=.6);p.add_argument('--reuse-completed',action='store_true')
    p.add_argument('--gpu-stream',action='store_true');p.add_argument('--poll-fps',type=float,default=120.)
    p.add_argument('--checkpoint',type=Path,default=core.ROOT/'pdqn_training_v1/last.pt');args=p.parse_args()
    exclusive();configure_precision_project();warnings.filterwarnings('ignore');torch.set_num_threads(4)
    root=args.output.resolve();root.mkdir(exist_ok=False);(root/'frames').mkdir();sources=root/'frozen-sources';sources.mkdir()
    for n in ['live_control_latency.py','control_latency_runtime.py','fast_rgb_perception.py','short_motion_features.py','async_frame_runtime.py','live_async_variable_speed.py','live_crossroad_turn.py','live_variable_speed_requested.py']:shutil.copy2(REV/n,sources/n)
    if args.gpu_stream:
        shutil.copy2(REV/'gpu_stream_runtime.py',sources/'gpu_stream_runtime.py')
        for n in ['PTZArena.cpp','PTZArena.h','PTZSim.Build.cs']:shutil.copy2(REV/'UE/PTZSim/Source/PTZSim'/n,sources/n)
    detector,extractor=core.frontend()
    if args.fast:detector=Detector(detector.base);extractor=PeriodicReID(extractor)
    a=demo.create_agent();core.resume(a,args.checkpoint);frozen=snapshot(a);a.epsilon=0.
    demo.CASE='known9';item=demo.layout('development',7);scene=(cross if args.scene=='crossroad' else variable).materialize(root,item)
    config=dict(checkpoint=str(args.checkpoint.resolve()),checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),case=item,
        detector=detector.provenance,reid=extractor.metadata(),motion=demo.MOTION,architecture='One frozen original P-DQN,15input,7exclusiveactions,4networks.',
        target_capture_fps=args.fps,async_capture=True,shared_memory_images=args.shm or args.gpu_stream,native_gpu_stream=args.gpu_stream,fast_perception=args.fast,action_semantics='Each neural pulse completes fully; no preemption, extension, simultaneousaction or repeatedoldaction. Same neural rawduration mapping, physicalrates and optics gain.',
        motion_estimator_window_s=args.motion_window,reuse_completed_sample=args.reuse_completed,all_learning_state_frozen=False,no_GT_control=True,no_pose_replay=True,no_video_acceleration=True)
    write(root/'configuration.json',config);runtime=None
    if args.gpu_stream:
        from stream_arena import StreamArena
        arena_type=StreamArena;shutil.copy2(REV/'stream_arena.py',sources/'stream_arena.py')
    else:arena_type=StandaloneArena
    with arena_type(root,headless=True) as arena:
        if args.gpu_stream:
            from gpu_stream_runtime import StreamTransport,StreamRuntime
            t=StreamTransport(arena.port)
        else:t=(SharedTransport if args.shm else Transport)(arena.port)
        if args.fast:extractor.transport=t
        try:
            env=Observation(t,detector,extractor)
            if args.fast:extractor.association=env.cfff;extractor.detector=detector
            state,initial=env.reset(scene,item['seed']);write(root/'initial-observation.json',dict(state=state.tolist(),**initial))
            if args.motion_window<.6:
                from short_motion_features import ShortFeatures
                previous=env.motion_features;env.motion_features=ShortFeatures(args.motion_window)
                env.motion_features.samples=previous.samples;env.motion_features.previous=previous.previous
            t.rpc('vset /ptz/realtime 1')
            if args.gpu_stream:
                t.raw.start_stream(args.fps);runtime=StreamRuntime(t.raw,root,args.poll_fps)
            else:runtime=Runtime(t.raw,root,args.fps)
            t.runtime=runtime;runtime.start(args.seconds)
            with (root/'decisions.jsonl').open('w',encoding='utf-8',buffering=1) as log:
                while time.perf_counter()-runtime.started<args.seconds-.25:
                    runtime.check()
                    if (root/'STOP_REQUEST').exists():break
                    inputs=state.copy();start=time.perf_counter();act,param,params=a.choose_action(state,train=True);act=int(act);chosen=time.perf_counter()
                    raw=float(param[0]);duration=float(np.clip(raw,core.LOW[act],core.HIGH[act]))*core.SECONDS[act]
                    action_filter_info=None
                    if ACTION_FILTER is not None:
                        act,duration,action_filter_info=ACTION_FILTER(env,a,inputs,act,duration,params)
                        chosen=time.perf_counter()
                    p,tlt,z=ACTION_RATES[act]
                    submit=time.perf_counter();t.rpc(f'vset /ptz/play {p} {tlt} {z} {duration:.12f}');ack=time.perf_counter();event=t.raw.play_events[-1].copy()
                    after=runtime.latest.sequence if runtime.latest else -1
                    frame=runtime.wait_completed(event) if args.reuse_completed else runtime.wait_frame(after,command_done=True)
                    ready=time.perf_counter();t.pin(frame)
                    state,row=env.observe(act,duration);end=time.perf_counter()
                    row.update(action=act,raw_parameter=raw,duration=duration,command_wall_s=submit-runtime.started,command_ack_wall_s=ack-runtime.started,
                        source_select_s=chosen-start,command_queue_and_rpc_s=ack-submit,feedback_wait_s=ready-ack,observe_total_s=end-ready,
                        command_execution=event,observation_capture_request_wall_s=frame.diagnostics['capture_request_wall_s'],
                        observation_wall_s=frame.wall_s,decision_done_wall_s=end-runtime.started,selection_and_execution_wall_s=end-start,
                        all_parameters=params.tolist(),input_state=inputs.tolist(),next_state=state.tolist(),observation_frame=frame.sequence)
                    if action_filter_info is not None:row['area_constraint']=action_filter_info
                    log.write(json.dumps(row)+'\n')
            while time.perf_counter()-runtime.started<args.seconds:runtime.check();time.sleep(.005)
            runtime.stop();t.runtime=None
            if args.gpu_stream:t.raw.stop_stream()
            write(root/'ue-motion-timing.json',t.rpc('vget /ptz/timing'));t.rpc('vset /ptz/realtime 0')
            write(root/'native-command-events.json',t.raw.play_events)
        finally:
            if runtime and runtime.io_thread.is_alive():runtime.stop()
            t.close()
    finish_evaluation(a,frozen);config.update(all_learning_state_frozen=True,frame_count=len(runtime.frames),wall_duration_s=runtime.frames[-1]['wall_s']-runtime.frames[0]['wall_s'])
    if args.fast:config.update(detector_fast_calls=detector.fast_calls,detector_recovery_calls=detector.recovery_calls,reid_real_calls=extractor.real_calls,reid_cached_calls=extractor.cached_calls)
    if args.gpu_stream:config.update(native_duplicate_polls_discarded=t.raw.duplicate_polls,native_source_sequence_gaps=t.raw.sequence_gaps,native_torn_read_retries=t.raw.torn_retries,
        native_ring_slots=4,independent_memory_reader_and_command_threads=True,compact_motor_acknowledgement=True,
        video_timestamps='Actual native GPU acquisition platformclock mapped to Python wall by startup midpoint anchor; acquisition cadence, not polling cadence. No repeated source sequences.',
        capture_clock_anchor_uncertainty_s=t.raw.anchor_uncertainty)
    write(root/'configuration.json',config)
    if args.gpu_stream:
        from encode_60fps_vfr import encode as source_encode
        source_encode(root,runtime.frames,name='live-ue-realtime-vfr.mp4')
    else:encode(root,runtime.frames,name='live-ue-realtime-vfr.mp4')
    print('TIMED CONTROL COMPLETE',root,flush=True)
if __name__=='__main__':main()
