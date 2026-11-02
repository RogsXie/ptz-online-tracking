"""Real fine-pulse support, image-only relabeling and source GPU fine-tuning."""
import speed_bootstrap
from speed_bootstrap import REV,LEGACY,configure_precision_project
import json,random,time,hashlib,warnings
import cv2,numpy as np,torch
import train as legacy
from author_aligned.source import recorded_update,load_core
from author_aligned.environment import AtomicTransport
from author_aligned.run import NETWORKS
from integration.runtime import StandaloneArena
from speed_agent_v3 import create_agent,initialize
from speed_runner_v3 import frontend,AuditedSpeed
from speed_runner_v4 import campaign
from speed_scenes import layout,materialize
from stability_objective import StabilityEnvironment,objective
from calibrate import write

PROFILE=json.loads((REV/'training_profile_v4.json').read_text(encoding='utf-8'))

def relabel(a,out):
    counts=dict(real_episodes_read=0,real_transitions=0,artificial_old_completion_bootstraps=0,changed_rewards=0)
    with (out/'relabel-audit.jsonl').open('w',encoding='utf-8',buffering=1) as audit:
        for path in sorted((REV/'training-v3/training').glob('episode-*/steps.jsonl')):
            rows=[json.loads(s) for s in path.read_text().splitlines()]
            if len(rows)<2:continue
            previous={};counts['real_episodes_read']+=1
            for index,(before,row) in enumerate(zip(rows,rows[1:]),1):
                reward,done,detail=objective(before,row,rows[:index+1],previous)
                if row['done'] and not done:counts['artificial_old_completion_bootstraps']+=1
                counts['changed_rewards']+=abs(reward-row['reward'])>1e-8
                a.memory.push(np.array(row['input_state'],np.float32),np.array([row['action'],*row['all_parameters']]),reward,np.array(row['next_state'],np.float32),done)
                counts['real_transitions']+=1
            audit.write(json.dumps(dict(source=str(path),source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),steps=len(rows)-1))+'\n')
    assert counts['real_episodes_read']==160 and counts['real_transitions']==7014
    write(out/'relabel-summary.json',counts);return counts

def collect(a,t,detector,extractor,out):
    frozen={n:{k:v.clone() for k,v in getattr(a,n).state_dict().items()} for n in NETWORKS}
    low=a.action_parameter_min_numpy.copy();high=a.action_parameter_max_numpy.copy();eps=a.epsilon
    a.action_parameter_max_numpy=np.array([5.,1.,1.,1.,1.,5.,5.],np.float32);a.epsilon=1.
    env=StabilityEnvironment(t,detector,extractor);steps=0;counts=[0]*7
    try:
        for i in range(128):
            rng=random.Random(38200000+i);item=layout('train',200000+i);q=i%4;sx,sy=[(-1,-1),(1,-1),(-1,1),(1,1)][q]
            item.update(index=i,seed=38200000+i,quadrant=q,layer=0,scale_group=2,
                planned_center=[.5+sx*rng.uniform(.004,.065),.5+sy*rng.uniform(.004,.065)],fov_deg=rng.uniform(20,30),role='Random fine-support training fragment, not complete control episode')
            folder=out/f'fine-support/fragment-{i:04d}';folder.mkdir(parents=True)
            state,initial=env.reset(materialize(folder,item),item['seed']);write(folder/'initial.json',initial|item)
            cv2.imwrite(str(folder/'initial.jpg'),cv2.cvtColor(t.rgb,cv2.COLOR_RGB2BGR))
            with (folder/'steps.jsonl').open('w',encoding='utf-8',buffering=1) as f:
                f.write(json.dumps(dict(initial=True,state=state.tolist(),**initial))+'\n')
                for _ in range(8):
                    act,param,params=a.choose_action(state,train=True);next_state,reward,done,row=env.step(int(act),float(param[0]))
                    row.update(input_state=state.tolist(),next_state=next_state.tolist(),all_parameters=params.tolist(),behavior_epsilon=1.)
                    f.write(json.dumps(row)+'\n');a.memory.push(state,np.concatenate(([act],params)).ravel(),reward,next_state,done)
                    steps+=1;counts[int(act)]+=1;state=next_state
                    if done:break
            if (i+1)%16==0:print('REAL FINE SUPPORT',i+1,steps,flush=True)
    finally:a.action_parameter_min_numpy=low;a.action_parameter_max_numpy=high;a.epsilon=eps
    assert a.update_count==0 and all(torch.equal(v,getattr(a,n).state_dict()[k]) for n,d in frozen.items() for k,v in d.items())
    write(out/'fine-support-summary.json',dict(fragments=128,new_real_transitions=steps,completed_control_episodes_claimed=0,
        original_random_branch=True,weights_and_targets_unchanged=True,action_counts=counts))
    return steps

def main():
    legacy.exclusive();configure_precision_project();warnings.filterwarnings('ignore');torch.set_num_threads(4)
    out=REV/'training-v4';out.mkdir(exist_ok=False);detector,extractor=frontend();a=create_agent();parent=initialize(a,out)
    relabeled=relabel(a,out);meta=dict(experiment='stationary_author_core_pdqn_speed_revision',speed_profile=PROFILE['profile'],profile=PROFILE,
        parent=parent,relabel_summary=relabeled,detector=detector.provenance,reid=extractor.metadata())
    steps=7014;completed=0;_,_,epsilon=load_core()
    try:
        with StandaloneArena(out,headless=True) as arena:
            t=AtomicTransport(arena.port,hud=False)
            try:
                steps+=collect(a,t,detector,extractor,out);write(out/'configuration.json',meta)
                frozen={k:v.clone() for k,v in a.param_net.state_dict().items()}
                for p in a.param_net_optimiser.param_groups:p['lr']=0.
                with (out/'critic-warmup.jsonl').open('w',encoding='utf-8',buffering=1) as f:
                    for _ in range(512):f.write(json.dumps(recorded_update(a))+'\n')
                assert all(torch.equal(v,a.param_net.state_dict()[k]) for k,v in frozen.items())
                a.param_net_optimiser=torch.optim.Adam(a.param_net.parameters(),lr=.00005)
                write(out/'critic-warmup-proof.json',dict(updates=512,online_parameter_weights_unchanged=True,real_replay=len(a.memory)))
                legacy.checkpoint(a,out/'after-warmup.pt',0,steps,meta)
                env=StabilityEnvironment(t,detector,extractor);evalenv=AuditedSpeed(t,detector,extractor);started=time.perf_counter()
                with (out/'episodes.jsonl').open('w',encoding='utf-8',buffering=1) as f:
                    for n in range(1,81):
                        a.epsilon=epsilon(480+n);r=legacy.episode(a,env,layout('train',60000+n-1),out/f'training/episode-{n:04d}',True,4,out,video=False)
                        completed=n;steps+=r['steps'];f.write(json.dumps(r)+'\n')
                        delta=sum(float((v-a.param_net.state_dict()[k]).abs().sum()) for k,v in frozen.items())
                        write(out/'progress.json',dict(completed=n,steps=steps,updates=a.update_count,postwarm_updates=a.update_count-512,
                            parameter_weight_abs_change=delta,last=r,wall_s=time.perf_counter()-started))
                        if n%10==0:print('V4 REAL TRAIN',n,steps,a.update_count,r['success'],flush=True)
                        if n in (20,40,80):
                            path=out/f'checkpoint-{n:04d}.pt';legacy.checkpoint(a,path,n,steps,meta)
                            campaign(a,evalenv,out/f'development-{n:04d}',path,count=80)
                        if n%10==0:legacy.checkpoint(a,out/'last.pt',n,steps,meta)
                legacy.checkpoint(a,out/'last.pt',completed,steps,meta)
            finally:t.close()
    except BaseException as exc:
        legacy.checkpoint(a,out/'interrupted.pt',completed,steps,meta);write(out/'failure.json',dict(error=repr(exc)));raise
    finally:write(out/'status.json',dict(completed=completed,steps=steps,updates=a.update_count,qualified=False))

if __name__=='__main__':main()
