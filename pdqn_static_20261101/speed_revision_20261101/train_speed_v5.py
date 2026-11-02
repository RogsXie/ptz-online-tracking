"""Conservative GPU fine-tuning with original raw bounds at calibrated25rate."""
import speed_bootstrap
from speed_bootstrap import REV,configure_precision_project
import json,random,time,warnings
import cv2,numpy as np,torch
import train as legacy
from author_aligned.source import load_core
from author_aligned.environment import AtomicTransport
from author_aligned.run import NETWORKS
from integration.runtime import StandaloneArena
from speed_agent_v5 import PROFILE,create_agent,initialize
from speed_runner_v5 import frontend,AuditedSpeed,campaign
from speed_environment_v5 import Environment
from speed_scenes import layout,materialize
from calibrate import write

def collect(a,t,detector,extractor,out):
    frozen={n:{k:v.clone() for k,v in getattr(a,n).state_dict().items()} for n in NETWORKS};eps=a.epsilon;a.epsilon=1.
    env=Environment(t,detector,extractor);steps=0;counts=[0]*7
    try:
        for i in range(64):
            if (out/'STOP_REQUEST').exists():raise InterruptedError('Requested collection-boundary stop')
            rng=random.Random(38300000+i);item=layout('train',300000+i);q=i%4;sx,sy=[(-1,-1),(1,-1),(-1,1),(1,1)][q]
            item.update(index=i,seed=38300000+i,quadrant=q,layer=0,scale_group=2,planned_center=[.5+sx*rng.uniform(.004,.065),.5+sy*rng.uniform(.004,.065)],fov_deg=rng.uniform(20,30),role='Uniform random real training fragment; not complete control episode')
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
            if (i+1)%16==0:print('V5 REAL SUPPORT',i+1,steps,flush=True)
    finally:a.epsilon=eps
    assert a.update_count==0 and all(torch.equal(v,getattr(a,n).state_dict()[k]) for n,d in frozen.items() for k,v in d.items())
    write(out/'fine-support-summary.json',dict(fragments=64,new_real_transitions=steps,completed_control_episodes_claimed=0,original_random_branch=True,weights_and_targets_unchanged=True,action_counts=counts))
    return steps

def main():
    legacy.exclusive();configure_precision_project();warnings.filterwarnings('ignore');torch.set_num_threads(4)
    out=REV/'training-v5';out.mkdir(exist_ok=False);detector,extractor=frontend();a=create_agent();parent=initialize(a,out)
    meta=dict(experiment='stationary_author_core_pdqn_speed_revision',speed_profile=PROFILE['profile'],profile=PROFILE,parent=parent,detector=detector.provenance,reid=extractor.metadata())
    write(out/'configuration.json',meta);steps=completed=0;_,_,epsilon=load_core()
    frozen={n:{k:v.clone() for k,v in getattr(a,n).state_dict().items()} for n in ('actor_net','param_net')}
    try:
        with StandaloneArena(out,headless=True) as arena:
            t=AtomicTransport(arena.port,hud=False)
            try:
                steps=collect(a,t,detector,extractor,out);env=Environment(t,detector,extractor);evalenv=AuditedSpeed(t,detector,extractor);started=time.perf_counter()
                with (out/'episodes.jsonl').open('w',encoding='utf-8',buffering=1) as f:
                    for n in range(1,41):
                        a.epsilon=epsilon(480+n);r=legacy.episode(a,env,layout('train',80000+n-1),out/f'training/episode-{n:04d}',True,4,out,video=False)
                        completed=n;steps+=r['steps'];f.write(json.dumps(r)+'\n')
                        delta={name:sum(float((v-getattr(a,name).state_dict()[k]).abs().sum()) for k,v in values.items()) for name,values in frozen.items()}
                        write(out/'progress.json',dict(completed=n,steps=steps,updates=a.update_count,network_weight_abs_change=delta,last=r,wall_s=time.perf_counter()-started))
                        if n%10==0:
                            assert all(d>0 for d in delta.values()) and a.update_count>0
                            print('V5 REAL TRAIN',n,steps,a.update_count,r['success'],flush=True)
                            path=out/f'checkpoint-{n:04d}.pt';legacy.checkpoint(a,path,n,steps,meta)
                            if n in (10,20,40):campaign(a,evalenv,out/f'development-{n:04d}',path,count=80)
                            legacy.checkpoint(a,out/'last.pt',n,steps,meta)
            finally:t.close()
    except BaseException as exc:
        legacy.checkpoint(a,out/'interrupted.pt',completed,steps,meta);write(out/'failure.json',dict(error=repr(exc)));raise
    finally:write(out/'status.json',dict(completed=completed,steps=steps,updates=a.update_count,qualified=False))

if __name__=='__main__':main()
