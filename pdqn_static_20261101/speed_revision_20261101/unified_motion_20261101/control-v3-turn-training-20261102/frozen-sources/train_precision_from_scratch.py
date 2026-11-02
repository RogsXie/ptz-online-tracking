"""Collect real synchronous UE RGB transitions and train fresh original P-DQN.

Training simulation clock pauses during learning. Delays/cycles are simulated
explicitly; speed/latency claims require separate genuine wall-clock online runs.
"""
import speed_bootstrap
from speed_bootstrap import REV,configure_precision_project
import argparse,hashlib,json,math,time,warnings,shutil,traceback
from collections import Counter
from pathlib import Path
import cv2,numpy as np,torch
from threadpoolctl import threadpool_limits
import train as legacy
from author_aligned.source import recorded_update,verify_source
from author_aligned.run import NETWORKS,OPTIMIZERS,restore_rng
from integration.transport import ACTION_RATES
from control_latency_runtime import SharedTransport
from stream_arena import StreamArena
import live_crossroad_robust_tracker as scenario
from target_roi_detector import Detector
import precision_policy_v2 as policy
from precision_training_scenes import scene,FAMILIES
from calibrate import write
import unified_core as core


def cost(row):
    return math.hypot(row['x']*480.,row['y']*270.)/480. if row['visible'] else 1.5


def reward(previous,row,action,cycle):
    if not row['visible']:return -3.
    pos=cost(row);scale=min(4.,abs(math.log(max(row['area'],1e-7)/.016)))
    result=12.*(cost(previous)-pos)+(-4.*pos-.2*scale+.25*(pos<=8/480 and .012<=row['area']<=.02))*(cycle/.12)
    if row['area']>.02:result-=min(2.,20.*(row['area']-.02))
    if previous['visible'] and action in (1,2,3,4):
        axis='x' if action in (3,4) else 'y';pixels=480 if axis=='x' else 270
        if previous[axis]*row[axis]<0 and abs(row[axis])*pixels>6:result-=.25
    return float(result)


def advance_observed(env,seconds,rates=(0.,0.,0.)):
    """Mirror continuous live tracking; policy state is built once per command.

    The final image is left for observe(), so no image is associated twice.
    Intermediate RGB results update only the detector/tracker, never rewards or
    state history. Native time pauses during inference, preserving 30Hz cadence.
    """
    remaining=float(seconds)
    while remaining>1e-8:
        dt=min(1./30.,remaining);remaining-=dt
        env.t.rpc(f'vset /ptz/step {rates[0]:.12f} {rates[1]:.12f} {rates[2]:.12f} {dt:.12f}')
        if remaining>1e-8:
            env.t.capture()
            detections=env.detector(env.t.rgb)
            env.cfff.choose(env.t.rgb,detections,count_step=False)


def rollout(a,env,folder,index,split,length,epsilon,learn,counts,root):
    path,item=scene(folder,index,split)
    delay=[.04,.08,.12][index%3];cycle=[.12,.16,.20,.24][(index//3)%4]
    env.precision_training_age=delay
    state,row=env.reset(path,item['seed']);a.epsilon=epsilon
    cv2.imwrite(str(folder/'initial.jpg'),cv2.cvtColor(env.t.rgb,cv2.COLOR_RGB2BGR))
    visible=centered=overrides=0;returns=[]
    with (folder/'steps.jsonl').open('w',encoding='utf-8',buffering=1) as log,(folder/'updates.jsonl').open('w',encoding='utf-8',buffering=1) as updates:
        for step in range(length):
            if (root/'STOP_REQUEST').exists():raise InterruptedError('Stop at completed transition boundary')
            # Move the world while keeping the last RGB/camera observation pinned.
            # Neither the native reply nor any scene GT enters the policy input.
            core.bind_camera(a,env);act,raw,params=a.choose_action(state,train=True)
            proposed=act;requested=float(raw[0])*policy.SECONDS[act]
            act,duration,guard=scenario.action_filter(env,a,state,act,requested,params)
            duration=float(np.clip(duration,policy.LOW[act]*policy.SECONDS[act],policy.HIGH[act]*policy.SECONDS[act]))
            g,zg=core.gain(float(env.t.diagnostics['fov']))
            if getattr(env.t,'pan_tilt_gain_override',None) is not None:g=float(env.t.pan_tilt_gain_override)
            if getattr(env.t,'zoom_gain_override',None) is not None:zg=float(env.t.zoom_gain_override)
            pan,tilt,zoom=ACTION_RATES[act]
            if delay:advance_observed(env,delay)
            advance_observed(env,duration,(pan*g,tilt*g,zoom*zg))
            idle=max(0.,cycle-delay-duration)
            if idle>=.001:advance_observed(env,idle)
            env.t.capture();env.precision_training_age=delay
            nxt,new=env.observe(act,duration);value=reward(row,new,act,max(cycle,delay+duration));done=step+1>=length
            executed_params=np.asarray(params,dtype=np.float32).copy();executed_params[act]=duration/policy.SECONDS[act]
            a.memory.push(state,np.r_[act,executed_params],value,nxt,done)
            counts[item['family']]+=1;visible+=int(new['visible']);centered+=int(new['visible'] and cost(new)<=8/480)
            overrides+=int(act!=proposed or abs(duration-requested)>1e-7);returns.append(value)
            log.write(json.dumps(dict(step=step,input_state=state.tolist(),next_state=nxt.tolist(),reward=value,done=done,
                proposal_action=proposed,proposal_duration_s=requested,executed_action=act,executed_duration_s=duration,
                replay_parameters=executed_params.tolist(),guard=guard,simulated_image_age_s=delay,
                cycle_sim_s=max(cycle,delay+duration),observation=new))+'\n')
            if learn and step%2==0:
                result=recorded_update(a)
                if result:updates.write(json.dumps(result)+'\n')
            row=new;state=nxt
    return dict(steps=length,visible=visible,centered=centered,override_count=overrides,return_sum=sum(returns),scene=item)


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--warmup-scenes',type=int,default=16);p.add_argument('--train-scenes',type=int,default=96)
    p.add_argument('--warmup-steps',type=int,default=64);p.add_argument('--train-steps',type=int,default=96)
    p.add_argument('--resume-fresh',type=Path,help='Continue this attempt only; rejects old pretrained PTZ checkpoints.')
    args=p.parse_args();legacy.exclusive();configure_precision_project();warnings.filterwarnings('ignore')
    torch.set_num_threads(4);cv2.setNumThreads(1);root=args.output.resolve();root.mkdir(exist_ok=bool(args.resume_fresh))
    source=root/('frozen-sources-stage2' if args.resume_fresh else 'frozen-sources');source.mkdir()
    for name in ['precision_policy_v2.py','precision_training_scenes.py','train_precision_from_scratch.py','stable_locked_control.py']:
        shutil.copy2(REV/name,source/name)
    # Set the declared new physical parameter ranges only in this fresh process.
    core.LOW,core.HIGH=policy.LOW,policy.HIGH
    detector,extractor=scenario.previous.previous.previous.previous.frontend();detector=Detector(detector.base)
    a=policy.create_agent();start_scene=0;counts=Counter();prior_wall=0.
    initial={n:{k:v.clone() for k,v in getattr(a,n).state_dict().items()} for n in NETWORKS}
    if args.resume_fresh:
        saved=torch.load(args.resume_fresh,map_location=a.device,weights_only=False)
        assert saved['metadata']['from_scratch'] and saved['metadata']['parent_checkpoint'] is None
        assert saved['metadata']['motion_profile']==policy.PROFILE and saved['metadata']['seed']==policy.SEED
        for name in (*NETWORKS,*OPTIMIZERS):getattr(a,name).load_state_dict(saved[name])
        for key,value in saved['replay'].items():setattr(a.memory,key,value)
        for key in ('update_count','actions_count','epsilon'):setattr(a,key,saved[key])
        restore_rng(a,saved['rng']);start_scene=int(saved['completed_episodes'])
        # Recount only completed checkpoint segments; failed partial files are
        # preserved but are not inserted into this resumed replay a second time.
        for i in range(args.warmup_scenes):
            counts[FAMILIES[i%8]]+=args.warmup_steps
        for i in range(start_scene):counts[FAMILIES[i%8]]+=args.train_steps
        initial_saved=torch.load(root/'initial-random.pt',map_location=a.device,weights_only=False)
        initial={n:initial_saved[n] for n in NETWORKS}
        prior_wall=float(json.loads((root/'status.json').read_text()).get('wall_s',0.))
        (root/'STOP_REQUEST').unlink(missing_ok=True)
    metadata=dict(motion_profile=policy.PROFILE,state_fields=policy.STATE_FIELDS,seed=policy.SEED,
        from_scratch=True,parent_checkpoint=None,reused_replay_transitions=0,source_sha256=verify_source(),
        initialization='Fresh random four-network construction, target copies, declared short-duration output biases, empty replay and both fresh Adam optimizers. No old PTZ checkpoint read.',
        training='One persistent native UE arena; fresh RGB transitions, intermediate actual RGB tracker observations at <=33.3ms slices during pulses and holds. 16x64 random-support steps and 96x96 balanced eight-family training steps. Original source P-DQN update equations.',
        observation='Correct actual action/duration; real RGB angular-Kalman velocity; measured own optics; image-age prediction. No target GT, IDs, speeds, paths or family labels.',
        reward='12*image_position_progress + cycle/.12*(-4*pixel_position/480 -.2*min(4,abslog(area/.016)) +.25*8px_jointinner) ; >2%area and >6px crossing penalties; missing -3.',
        execution='RGB engineering action mask/duration bounds retained and logged; replay stores actual executed discrete action and selected duration, not rejected proposal.',
        delay='World advances explicitly 40/80/120ms while last image remains pinned; cycles120/160/200/240ms. Synchronous collection is not online FPS evidence.',
        families=list(FAMILIES),detector=detector.provenance,reid=extractor.metadata(),gpu=torch.cuda.get_device_name(0))
    metadata.update(parameter_low=list(policy.LOW),parameter_high=list(policy.HIGH),seconds_per_raw=list(policy.SECONDS),
        precision_motion_budget='RGB velocity determines opticsgain floor up to 1; remaining-error guard retained; pan/tilt up to100ms, minimum1ms.',
        fresh_curriculum_parent=str(args.resume_fresh) if args.resume_fresh else None,
        continued_current_attempt_replay_transitions=len(a.memory) if args.resume_fresh else 0,
        fresh_curriculum_parent_sha256=hashlib.sha256(args.resume_fresh.read_bytes()).hexdigest() if args.resume_fresh else None)
    write(root/('configuration-stage2.json' if args.resume_fresh else 'configuration.json'),metadata)
    if not args.resume_fresh:legacy.checkpoint(a,root/'initial-random.pt',0,0,metadata)
    completed=start_scene;start=time.perf_counter();status='running'
    write(root/'status.json',dict(status='running',completed_scenes=completed,real_steps=sum(counts.values())))
    try:
        with StreamArena(root,headless=True) as arena:
            transport=SharedTransport(arena.port)
            try:
                env=policy.observation_class(scenario.Observation)(transport,detector,extractor)
                for i in range(args.warmup_scenes if not args.resume_fresh else 0):
                    result=rollout(a,env,root/f'warmup/{i:03d}',i,'warmup',args.warmup_steps,1.,False,counts,root)
                    write(root/'progress.json',dict(phase='warmup',completed_scenes=i+1,real_steps=sum(counts.values()),updates=a.update_count,last=result,wall_s=time.perf_counter()-start))
                    print('FRESH WARMUP',i+1,sum(counts.values()),result['visible'],flush=True)
                if not args.resume_fresh:
                    assert all(torch.equal(v,getattr(a,n).state_dict()[k]) for n,d in initial.items() for k,v in d.items())
                    write(root/'fresh-initialization-proof.json',dict(real_new_steps=sum(counts.values()),all_four_networks_unchanged_during_support=True,loaded_policy_checkpoint=None,reused_replay=0,optimizers_empty=True))
                for i in range(start_scene,args.train_scenes):
                    epsilon=.65 if i<24 else .35 if i<56 else .12
                    result=rollout(a,env,root/f'train/{i:03d}',i,'train',args.train_steps,epsilon,True,counts,root);completed=i+1
                    progress=dict(phase='training',completed_scenes=completed,real_steps=sum(counts.values()),updates=a.update_count,group_steps=dict(counts),last=result,wall_s=prior_wall+time.perf_counter()-start)
                    write(root/'progress.json',progress);print('FRESH TRAIN',completed,progress['real_steps'],a.update_count,result['scene']['family'],result['visible'],result['centered'],flush=True)
                    if completed%16==0:legacy.checkpoint(a,root/f'checkpoint-{completed:03d}.pt',completed,sum(counts.values()),metadata)
                legacy.checkpoint(a,root/'last.pt',completed,sum(counts.values()),metadata);status='complete'
            finally:transport.close()
    except BaseException:
        status='interrupted_or_failed';traceback.print_exc();raise
    finally:
        proof={n:dict(changed_tensors=sum(not torch.equal(v,getattr(a,n).state_dict()[k]) for k,v in initial[n].items()),parameter_device=sorted(set(str(v.device) for v in getattr(a,n).parameters()))) for n in NETWORKS}
        write(root/'status.json',dict(status=status,completed_scenes=completed,real_steps=sum(counts.values()),updates=a.update_count,group_steps=dict(counts),network_proof=proof,wall_s=prior_wall+time.perf_counter()-start,qualified=False))


if __name__=='__main__':
    with threadpool_limits(limits=1,user_api='blas'):main()
