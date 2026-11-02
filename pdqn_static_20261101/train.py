"""PyCharm entry: existing 50mamba interpreter, actual UE and original P-DQN."""
import bootstrap
from bootstrap import HERE,REPRO
import argparse,copy,hashlib,json,random,time,warnings,traceback
from pathlib import Path
import cv2,numpy as np,torch
from scenes import layout,materialize
from environment import Environment
from calibrate import write
from author_aligned.source import create_agent,recorded_update,verify_source,load_core
from author_aligned.environment import AtomicTransport
from author_aligned.run import save,restore_rng,rng_state,NETWORKS,OPTIMIZERS
from integration.runtime import StandaloneArena
from integration.experiment import make_extractor
from control_model_v13.detector import create_detector

def exclusive():
    import psutil,os
    for p in psutil.process_iter(['pid','name','cmdline']):
        if p.info['pid']==os.getpid():continue
        if (p.info['name'] or '').lower()=='unrealeditor.exe':raise RuntimeError(f'Existing UE PID {p.pid}; no concurrent arena permitted')

def checkpoint(agent,path,completed,steps,meta):
    save(agent,path,meta,completed,steps)

def resume(agent,path):
    s=torch.load(path,map_location=agent.device,weights_only=False)
    if s['metadata'].get('experiment')!='stationary_author_core_pdqn_20261101':raise ValueError('Unrelated checkpoint')
    for name in (*NETWORKS,*OPTIMIZERS):getattr(agent,name).load_state_dict(s[name])
    for key,value in s['replay'].items():setattr(agent.memory,key,value)
    for key in ('update_count','actions_count','epsilon'):setattr(agent,key,s[key])
    restore_rng(agent,s['rng']);return s['completed_episodes'],s['environment_steps']

def snapshot(agent):
    return dict(rng=rng_state(agent),actions=agent.actions_count,epsilon=agent.epsilon,
                updates=agent.update_count,nets={n:{k:v.clone() for k,v in getattr(agent,n).state_dict().items()} for n in NETWORKS},
                replay=(len(agent.memory),agent.memory.position),
                optimizers={n:copy.deepcopy(getattr(agent,n).state_dict()) for n in OPTIMIZERS})

def finish_evaluation(agent,s):
    assert all(torch.equal(v,getattr(agent,n).state_dict()[k]) for n,d in s['nets'].items() for k,v in d.items())
    assert agent.update_count==s['updates'] and (len(agent.memory),agent.memory.position)==s['replay']
    for name,saved in s['optimizers'].items():
        current=getattr(agent,name).state_dict()
        assert saved['param_groups']==current['param_groups']
        for key,values in saved['state'].items():
            for k,v in values.items():
                assert torch.equal(v,current['state'][key][k]) if torch.is_tensor(v) else v==current['state'][key][k]
    agent.actions_count=s['actions'];agent.epsilon=s['epsilon'];restore_rng(agent,s['rng'])

def episode(agent,env,item,folder,training,interval,root,video=False):
    folder.mkdir(parents=True,exist_ok=False);scene=materialize(folder,item)
    state,initial=env.reset(scene,item['seed']);cv2.imwrite(str(folder/'initial.jpg'),cv2.cvtColor(env.t.rgb,cv2.COLOR_RGB2BGR))
    write(folder/'initial.json',initial|item)
    # No case rejection based on slight center mismatch or detection failure.
    returns=[];updates=[];start=time.perf_counter();counts=[0]*7;reversals=0;last_axis={};lost_s=0.;max_lost_s=0.;loss_run=0.
    writer=cv2.VideoWriter(str(folder/'actual-action-end-frames.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),10,(960,540)) if video else None
    if writer:writer.write(cv2.cvtColor(env.t.rgb,cv2.COLOR_RGB2BGR))
    with (folder/'steps.jsonl').open('w',encoding='utf-8',buffering=1) as log, (folder/'updates.jsonl').open('w',encoding='utf-8',buffering=1) as ul:
        log.write(json.dumps(dict(initial=True,state=state.tolist(),**initial))+'\n')
        try:
            for step in range(512):
                if (root/'STOP_REQUEST').exists():raise InterruptedError('STOP_REQUEST at action boundary')
                wall=time.perf_counter();action,param,parameters=agent.choose_action(state,train=True);selection_s=time.perf_counter()-wall
                next_state,reward,done,row=env.step(int(action),float(param[0]));counts[action]+=1
                if action:
                    axis={1:0,2:0,3:1,4:1,5:2,6:2}[int(action)]
                    if axis in last_axis and last_axis[axis]!=int(action):reversals+=1
                    last_axis[axis]=int(action)
                if not row['visible']:lost_s+=row['duration'];loss_run+=row['duration'];max_lost_s=max(max_lost_s,loss_run)
                else:loss_run=0.
                row.update(input_state=state.tolist(),next_state=next_state.tolist(),all_parameters=parameters.tolist(),selection_s=selection_s)
                log.write(json.dumps(row,allow_nan=False)+'\n')
                if writer:
                    im=cv2.cvtColor(env.t.rgb,cv2.COLOR_RGB2BGR)
                    for _ in range(max(1,round(row['duration']*10))):writer.write(im)
                if training:
                    agent.memory.push(state,np.concatenate(([action],parameters)).ravel(),float(reward),next_state,done)
                    if step%interval==0:
                        wall=time.perf_counter();update=recorded_update(agent)
                        if update:
                            update['wall_s']=time.perf_counter()-wall;ul.write(json.dumps(update)+'\n');updates.append(update)
                            if not (root/'gpu-update-proof.json').exists():
                                proof=dict(update_count=agent.update_count,replay_size=len(agent.memory),batch=agent.batch_size,
                                    networks={n:sorted(set(str(v.device) for v in getattr(agent,n).parameters())) for n in NETWORKS},
                                    gradients={n:sorted(set(str(v.grad.device) for v in getattr(agent,n).parameters() if v.grad is not None)) for n in ('actor_net','param_net')},
                                    adam={n:sorted(set(str(v.device) for d in getattr(agent,n).state.values() for k,v in d.items() if k in ('exp_avg','exp_avg_sq'))) for n in OPTIMIZERS},
                                    real_interaction_update=update,
                                    source_cpu_boundary='Original _invert_gradients copies parameter gradients/values to CPU; retained, measured, not hidden')
                                assert all(v==['cuda:0'] for group in ('networks','gradients','adam') for v in proof[group].values())
                                write(root/'gpu-update-proof.json',proof)
                returns.append(reward);state=next_state
                if done:break
        finally:
            if writer:writer.release()
    cv2.imwrite(str(folder/'final.jpg'),cv2.cvtColor(env.t.rgb,cv2.COLOR_RGB2BGR))
    elapsed=env.rows[-1]['sim_time'];result=dict(**item,success=bool(row.get('success')),source_condition=env.source_condition,
        steps=len(returns),sim_time=elapsed,reward=sum(returns),final_center=row['center_error'],final_area=row['area'],
        best_center=min([r['center_error'] for r in env.rows if r['visible']],default=None),
        availability=1-lost_s/max(elapsed,1e-9),longest_loss_s=max_lost_s,reversals=reversals,
        reversals_per10s=reversals/max(elapsed,1e-9)*10,action_counts=counts,
        updates=len(updates),updates_total=agent.update_count,replay_size=len(agent.memory),
        epsilon=agent.epsilon,wall_s=time.perf_counter()-start,
        q_loss=float(np.mean([u['q_loss'] for u in updates])) if updates else None,
        update_mean_s=float(np.mean([u['wall_s'] for u in updates])) if updates else None,
        parameter_grad_mean=float(np.mean([u['parameter_grad_norm_after_clip'] for u in updates])) if updates else None)
    for k in ('hold_center_mean','hold_center_p95','hold_logscale_mean','hold_start_s'):result[k]=row.get(k)
    write(folder/'summary.json',result);return result

def evaluate(agent,env,root,completed,count=20,split='development'):
    folder=root/f'{split}-{completed:04d}';folder.mkdir(exist_ok=False)
    s=snapshot(agent);agent.epsilon=0.;results=[]
    # Balanced quadrant x scale; rotate the offset layer in successive probes.
    indices=[i for i in range(80) if (lambda x:x['layer']==(x['quadrant']+x['scale_group'])%5)(layout(split,i))]
    if count>=80:indices=list(range(count))
    else:indices=indices[:count]
    try:
        for i in indices:
            item=layout(split,i);r=episode(agent,env,item,folder/f'case-{i:03d}',False,32,root,video=True)
            results.append(r);print('EVAL',completed,i,r['success'],r['final_center'],r['final_area'],r['action_counts'],flush=True)
            write(folder/'results.json',results)
    finally:finish_evaluation(agent,s)
    write(folder/'frozen-verification.json',dict(unchanged_weights_targets_replay_optimizers=True,updates=agent.update_count))
    return results

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=HERE/'training-source-cadence')
    p.add_argument('--episodes',type=int,default=320);p.add_argument('--checkpoint',type=Path)
    p.add_argument('--update-interval',type=int,default=32);p.add_argument('--evaluate-only',action='store_true')
    p.add_argument('--eval-count',type=int,default=16);p.add_argument('--split',default='development',choices=['development','validation','final_test'])
    args=p.parse_args();exclusive();warnings.filterwarnings('ignore');torch.set_num_threads(4)
    root=args.output.resolve();root.mkdir(exist_ok=False)
    detector=create_detector(checkpoint=REPRO/'weights/yolo26s_ue_overhead_v1.pt',device='cuda',threshold=.15,imgsz=960)
    assert detector.base.provenance['sha256']=='1dd0de55c1791dce289cc7dc1d8eb6c50e488b26d38941f4c20845a6cadb5ae2'
    detector.warmup();extractor=make_extractor('cuda');agent=create_agent(20261101,'cuda');completed=steps=0
    if args.checkpoint:completed,steps=resume(agent,args.checkpoint)
    meta=dict(experiment='stationary_author_core_pdqn_20261101',source=verify_source(),
        protocol=json.loads((HERE/'protocol.json').read_text()),update_interval=args.update_interval,
        learning_schedule=dict(effective_update_interval=args.update_interval,source_main_interval=32,
            difference='Only update frequency changes; original uniform replay and update equations preserved' if args.update_interval!=32 else 'Source update frequency',
            evidence='80-episode frozen contrast: 0/16 before vs 3/16 after 2048 extra original updates on identical real replay'),
        detector=detector.provenance,reid=extractor.metadata(),cuda_devices={n:str(next(getattr(agent,n).parameters()).device) for n in NETWORKS},
        torch=torch.__version__,gpu=torch.cuda.get_device_name(),parent=str(args.checkpoint) if args.checkpoint else None,
        sources={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in HERE.glob('*.py')})
    write(root/'configuration.json',meta);checkpoint(agent,root/'initial.pt',completed,steps,meta)
    _,_,epsilon=load_core()
    try:
        with StandaloneArena(root,headless=True) as arena:
            t=AtomicTransport(arena.port,hud=False)
            try:
                env=Environment(t,detector,extractor)
                if args.evaluate_only:
                    evaluate(agent,env,root,completed,args.eval_count,args.split);return
                with (root/'episodes.jsonl').open('w',encoding='utf-8',buffering=1) as log:
                    for _ in range(args.episodes):
                        number=completed+1;agent.epsilon=epsilon(number)
                        item=layout('train',completed)
                        r=episode(agent,env,item,root/f'training/episode-{number:04d}',True,args.update_interval,root,video=number in (1,20,80,160,320))
                        completed=number;steps+=r['steps'];log.write(json.dumps(r)+'\n')
                        print('TRAIN',completed,'success',r['success'],'return',round(r['reward'],2),'final',r['final_center'],r['final_area'],'updates',agent.update_count,'steps',r['steps'],'eps',round(agent.epsilon,3),flush=True)
                        write(root/'progress.json',dict(completed=completed,steps=steps,updates=agent.update_count,last=r,status='training'))
                        if completed%10==0:checkpoint(agent,root/'last.pt',completed,steps,meta)
                        if completed%80==0:
                            checkpoint(agent,root/f'checkpoint-{completed:04d}.pt',completed,steps,meta)
                            results=evaluate(agent,env,root,completed)
                            write(root/f'development-{completed:04d}-summary.json',dict(passed=sum(r['success'] for r in results),count=len(results)))
                    checkpoint(agent,root/'last.pt',completed,steps,meta)
            finally:t.close()
    except BaseException as e:
        checkpoint(agent,root/'interrupted.pt',completed,steps,meta)
        write(root/'failure.json',dict(error=repr(e),traceback=traceback.format_exc(),completed=completed,steps=steps));raise
    finally:
        write(root/'status.json',dict(completed=completed,steps=steps,updates=agent.update_count,qualified=False))

if __name__=='__main__':main()
