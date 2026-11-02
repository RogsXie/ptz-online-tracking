"""True online frames from continuous perception; motor commands remain exclusive."""
import speed_bootstrap
from speed_bootstrap import REV,configure_precision_project
import argparse,copy,hashlib,json,shutil,time,warnings
from pathlib import Path
import cv2,numpy as np,torch
from threadpoolctl import threadpool_limits
import unified_core as core
import live_unified_motion as demo
import live_crossroad_robust_tracker as scenario
from target_roi_detector import Detector
from fast_rgb_perception import PeriodicReID
from continuous_perception_runtime import Worker,policy_view
from online_result_recorder import Recorder
from stream_arena import StreamArena
from gpu_stream_runtime import StreamTransport,StreamRuntime
from integration.transport import ACTION_RATES
from train import exclusive,snapshot,finish_evaluation
from calibrate import write

DEFAULT_CENTERING_V4=True

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seconds',type=float,default=30.);p.add_argument('--fps',type=float,default=30.)
    p.add_argument('--perception-fps',type=float,default=30.)
    p.add_argument('--latest-frame',action='store_true',help='Skip unread older source frames; report sequence gaps.')
    p.add_argument('--control-v3',action='store_true',help='Camera-only zoom lookahead and edge-axis priority.')
    p.add_argument('--centering-v4',action=argparse.BooleanOptionalAction,default=DEFAULT_CENTERING_V4,
        help='Balanced RGB-error axis scheduling and tighter acquisition centering.')
    p.add_argument('--scene',choices=['crossroad','viaduct20','viaduct30','precision-turn'],default='crossroad')
    p.add_argument('--checkpoint',type=Path,default=core.ROOT/'control-v3-turn-training-20261102/deploy.pt');args=p.parse_args()
    exclusive();configure_precision_project();warnings.filterwarnings('ignore')
    torch.set_num_threads(4);cv2.setNumThreads(1)
    root=args.output.resolve();root.mkdir(exist_ok=False);(root/'frames').mkdir();(root/'frozen-sources').mkdir()
    for n in ['live_fast_online_results.py','continuous_perception_runtime.py','online_result_recorder.py','encode_60fps_vfr.py','gpu_stream_runtime.py','async_frame_runtime.py','control_latency_runtime.py','fast_rgb_perception.py','stream_arena.py','batched_vehicle_reid.py','robust_locked_tracker.py','verified_visual_bridge.py','stable_locked_control.py','camera_control_v3.py','camera_centering_v4.py']:
        shutil.copy2(REV/n,root/'frozen-sources'/n)
    for n in ['PTZArena.cpp','PTZArena.h','PTZSim.Build.cs']:shutil.copy2(REV/'UE/PTZSim/Source/PTZSim'/n,root/'frozen-sources'/n)
    frontend=scenario.previous.previous.previous.previous.frontend
    detector,extractor=frontend();detector=Detector(detector.base);extractor=PeriodicReID(extractor)
    saved_profile=torch.load(args.checkpoint,map_location='cpu',weights_only=False)['metadata']['motion_profile']
    precision=saved_profile!='unified15_mixed_motion_v1'
    observation_type=scenario.Observation
    if precision:
        import precision_policy_v2 as precision_policy
        agent=precision_policy.load_for_evaluation(args.checkpoint)
        core.LOW,core.HIGH=precision_policy.LOW,precision_policy.HIGH
        observation_type=precision_policy.observation_class(observation_type)
        shutil.copy2(REV/'precision_policy_v2.py',root/'frozen-sources/precision_policy_v2.py')
    else:
        agent=demo.create_agent();core.resume(agent,args.checkpoint)
    if args.control_v3:
        if not precision:raise ValueError('Control-v3 requires precision observation contract')
        agent.precision_control_v3=True
    if args.centering_v4 and precision and getattr(agent,'precision_control_v3',False):
        agent.precision_centering_v4=True
    frozen=snapshot(agent);agent.epsilon=0.
    demo.CASE='known9';item=demo.layout('development',7)
    materialize=scenario.materialize
    if args.scene in {'viaduct20','viaduct30'}:
        from live_viaduct20_scene import materialize
    if args.scene=='precision-turn':
        from precision_training_scenes import scene as procedural_scene
        scene,item=procedural_scene(root,4,'development')
        demo.MOTION=dict(scenario='precision_heldout_fast_turn',requested_speed_m_s=item['speed_m_s'],
            usage='Scene metadata only, never supplied to control.')
    else:scene=materialize(root,item)
    config=dict(checkpoint=str(args.checkpoint),checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        case=item,scene=args.scene,detector=detector.provenance,reid=extractor.metadata(),motion=demo.MOTION,
        architecture='Same frozen P-DQN; continuous single-owner RGBperception independent ofexclusive motorwaiting. Per-command15statehistory retained by cachehandoff.',
        target_capture_fps=args.fps,target_perception_fps=args.perception_fps,
        true_online_output_recording=True,whole_vehicle_detector=True,target_selection='locked_identity_timeout_v1',
        lost_memory_timeout_s=3.,brief_miss_grace_s=.25,area_constraint_scope='locked_target_only',
        presented_video='online-output/true-online-output.mp4',
        online_overlay='Exact fresh processedRGBframe+actualonline box, rendered duringrun. No interpolation, boxreplay or postrun inference.',
        action_semantics='Every motorpulse fullycompletes; nextcommand consumes an actual perception result withsource time>=nativeactionend andremaining=0. Perception alsoprocesses midpulse frames.',
        threads=dict(torch=4,opencv=1,blas=1),all_learning_state_frozen=False,no_GT_control=True)
    config.update(startup_scale_target_area=.016,startup_center_tolerance_px=40.,
        motion_profile=saved_profile,latest_source_frame_only=args.latest_frame,
        perception_profile='Original FP32 full detection each processed frame, fresh FP32 ReID every candidate; multi-frame local turn reassociation with angular ownership and duplicate-fragment protection.',
        camera_control_profile='precision_v3_zoom_lookahead_edge_priority' if getattr(agent,'precision_control_v3',False) else 'precision_v2',
        startup_zoom_max_fov_step_deg=6.,startup_scale='First lock of each target generation: roughly centre, then actively zoom toward1.6% area. Finish at>=1.44%;2% upper guard unchanged. Fresh RGB feedback between exclusive pulses.',
        native_stream_metadata='Dynamic same-frame state only; full immutable configuration saved in initial-native-state.json and scene.json.',
        identity_association='Joint assignment; angular/established ownership; nominal locked ReID>=.70; unique local turn recovery: 2 strong-gallery frames or 3 high-confidence adjacent-crop confirmations; gallery>=.60 plus adjacent-crop ReID>=.90 for view change, tight geometry and age<=.15s for normal short-term links; age<=.35s for turn confirmation; short-term-only/recovery links cannot update appearance bank; merged/duplicate-fragment protection.',
        motion_gain='Declared engineering bound: per-frame RGB/Kalman bearing speed>.8deg/s permits minimum pan/tilt opticsgain.18;>.35 permits.10. Neural15state/per-command history unchanged. Remaining-image-error duration cap still applies.',
        search_scale='Last visible angular size limits widening to keep predicted box area>=.0015; no predicted boxes are drawn.')
    config['reid'].update(runtime_refresh_period_s=0.,fresh_batch_every_candidate=True,
        feature_cache_enabled=False,encoding_dtype='float32',
        normalized_feature_dtype='float32')
    if getattr(agent,'precision_centering_v4',False):
        config.update(camera_control_profile='precision_v4_balanced_centering',startup_center_tolerance_px=12.,
            zoom_feedback_tolerance_px=20.,centering_governor='Balance axes; prioritize significant axes waiting >=.28s; bounded half-feedback motion lead; acquire large-error duration floor <=100ms; mechanical/area/exclusive-pulse limits retained.')
    if precision:
        config.update(architecture='Freshly trained precision15 P-DQN, explicit RGB engineering mask/duration/scale bounds, continuous single-owner perception.',
            observation_contract=precision_policy.STATE_FIELDS)
    write(root/'configuration.json',config)
    runtime=worker=recorder=None;ok=False
    try:
        with StreamArena(root,headless=True) as arena:
            transport=StreamTransport(arena.port);extractor.transport=transport
            try:
                transport.raw.latest_only=args.latest_frame
                env=observation_type(transport,detector,extractor);extractor.association=env.cfff;extractor.detector=detector
                state,initial=env.reset(scene,item['seed']);write(root/'initial-observation.json',dict(state=state.tolist(),**initial))
                write(root/'initial-native-state.json',transport.diagnostics)
                from short_motion_features import ShortFeatures
                before=env.motion_features;env.motion_features=ShortFeatures(.25)
                env.motion_features.samples=before.samples;env.motion_features.previous=before.previous
                policy,cache=policy_view(env,transport)
                recorder=Recorder(root);recorder.publish(transport.rgb,initial)
                transport.rpc('vset /ptz/realtime 1');transport.raw.start_stream(args.fps)
                runtime=StreamRuntime(transport.raw,root,1000.);transport.runtime=runtime;runtime.start(args.seconds)
                worker=Worker(env,transport,runtime,recorder,root,args.perception_fps);worker.start();consumed=-1
                with (root/'decisions.jsonl').open('w',encoding='utf-8',buffering=1) as log:
                    while time.perf_counter()-runtime.started<args.seconds-.25:
                        worker.check()
                        if (root/'STOP_REQUEST').exists():break
                        agent.camera_diagnostics=transport.diagnostics;agent.camera_limits=policy.camera_limits
                        inputs=state.copy();start=time.perf_counter();act,param,params=agent.choose_action(state,train=True)
                        act=int(act);raw=float(param[0]);duration=float(np.clip(raw,core.LOW[act],core.HIGH[act]))*core.SECONDS[act]
                        act,duration,guard=scenario.action_filter(policy,agent,inputs,act,duration,params)
                        chosen=time.perf_counter();pan,tilt,zoom=ACTION_RATES[act]
                        submit=time.perf_counter();transport.rpc(f'vset /ptz/play {pan} {tilt} {zoom} {duration:.12f}');ack=time.perf_counter()
                        event=transport.raw.play_events[-1].copy();result=worker.wait_completed(event,consumed);ready=time.perf_counter()
                        consumed=result.index;transport.pin(result.frame);cache.install(result)
                        state,row=policy.observe(act,duration);end=time.perf_counter()
                        row.update(action=act,raw_parameter=raw,duration=duration,command_wall_s=submit-runtime.started,
                            command_ack_wall_s=ack-runtime.started,source_select_s=chosen-start,
                            command_queue_and_rpc_s=ack-submit,feedback_wait_s=ready-ack,observe_total_s=end-ready,
                            detector_s=result.row['detector_s'],reid_s=result.row['reid_s'],perception_s=result.row['perception_s'],
                            command_execution=event,observation_capture_request_wall_s=result.frame.diagnostics['capture_request_wall_s'],
                            observation_wall_s=result.frame.wall_s,decision_done_wall_s=end-runtime.started,
                            selection_and_execution_wall_s=end-start,all_parameters=params.tolist(),input_state=inputs.tolist(),next_state=state.tolist(),
                            observation_frame=result.frame.sequence,perception_result_index=result.index,area_constraint=guard,
                            policy_cached_actual_result=True)
                        log.write(json.dumps(row)+'\n')
                while time.perf_counter()-runtime.started<args.seconds:worker.check();time.sleep(.005)
                worker.stop();worker=None;runtime.stop();transport.runtime=None;transport.raw.stop_stream()
                recorder.flush()
                write(root/'ue-motion-timing.json',transport.rpc('vget /ptz/timing'));transport.rpc('vset /ptz/realtime 0');recorder.mark_live_stopped()
                write(root/'native-command-events.json',transport.raw.play_events)
            finally:
                try:
                    if worker is not None:worker.stop()
                finally:
                    try:
                        if runtime is not None and runtime.io_thread.is_alive():runtime.stop()
                    finally:transport.close()
        finish_evaluation(agent,frozen)
        config.update(all_learning_state_frozen=True,frame_count=len(runtime.frames),wall_duration_s=runtime.frames[-1]['wall_s']-runtime.frames[0]['wall_s'],
            reid_real_calls=extractor.real_calls,reid_cached_calls=extractor.cached_calls,roi_calls=detector.roi_calls,
            native_source_sequence_gaps=transport.raw.sequence_gaps)
        write(root/'configuration.json',config)
        from encode_60fps_vfr import encode
        encode(root,runtime.frames,name='live-ue-realtime-vfr.mp4');ok=True
    finally:
        if recorder is not None:recorder.close(encode=ok)
    print('CONTINUOUS TRUE ONLINE COMPLETE',root,flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=1,user_api='blas'):main()
