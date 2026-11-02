"""Real UE RGB dynamic interactions and unchanged pinned-source GPU updates."""
import speed_bootstrap
from speed_bootstrap import REV,configure_precision_project
import json,time,warnings,argparse
from pathlib import Path
import numpy as np,torch
import train as legacy
from author_aligned.source import recorded_update,verify_source
from author_aligned.run import NETWORKS
from author_aligned.environment import AtomicTransport
from integration.runtime import StandaloneArena
from speed_runner_v5 import frontend
from motion_learning15 import create_agent,transfer,TrainingEnvironment,training_scene,PROFILE
from calibrate import write

def episode(agent,env,out,index,epsilon,updates):
    scene,item=training_scene(out,index);state,row=env.reset(scene,item['seed']);agent.epsilon=epsilon;count=0;center=[]
    with (out/'steps.jsonl').open('w',encoding='utf-8',buffering=1) as f,(out/'updates.jsonl').open('w',encoding='utf-8',buffering=1) as u:
        f.write(json.dumps(dict(initial=True,state=state.tolist(),**row))+'\n')
        for step in range(80 if updates else 64):
            if (out.parents[1]/'STOP_REQUEST').exists():raise InterruptedError('Training stop request')
            act,param,params=agent.choose_action(state,train=True);nxt,reward,done,row=env.step(int(act),float(param[0]));count+=1
            row.update(input_state=state.tolist(),next_state=nxt.tolist(),all_parameters=params.tolist());f.write(json.dumps(row)+'\n')
            agent.memory.push(state,np.r_[act,params],reward,nxt,done)
            if updates and step%2==0:
                record=recorded_update(agent)
                if record:u.write(json.dumps(record)+'\n')
            if row['visible']:center.append(row['center_error'])
            state=nxt
            if done:break
    return dict(real_steps=count,center_mean=float(np.mean(center)) if center else None,visible=len(center),updates=agent.update_count)

def main():
    p=argparse.ArgumentParser();p.add_argument('--episodes',type=int,default=16);p.add_argument('--output',type=Path,default=REV/'dynamic15-training-v1');args=p.parse_args()
    legacy.exclusive();configure_precision_project();warnings.filterwarnings('ignore');torch.set_num_threads(4);out=args.output;out.mkdir(exist_ok=False)
    detector,extractor=frontend();agent=create_agent();parent=transfer(agent);meta=dict(motion_profile=PROFILE,parent=parent,source_sha256=verify_source(),
        observation='Old8 sourcefeatures plus image/camera-derived bearingvelocity2,FOV,feedbackdt,owncamera increments3. No GT or pathvelocity.',
        action='7exclusive sourceP-DQN actions; angularrawbounds.25..10,others.3..5; original source update math; Qlr1e-4,Paramlr1e-5',
        training_clock='Real UE atomic actions plus explicitidle to100mscycle, RGB at cycleend; scene advances during idle. This is training clock, not realtime performance evidence.',
        reward='20imagecostimprovement-2imagecost+1.5inner-.02nonStay; continuousdynamicmaintenance,8secondfragments; engineeredobjective',
        detector=detector.provenance,reid=extractor.metadata())
    write(out/'configuration.json',meta);steps=completed=0;started=time.perf_counter()
    with StandaloneArena(out,headless=True) as arena:
        t=AtomicTransport(arena.port,hud=False)
        try:
            env=TrainingEnvironment(t,detector,extractor);frozen={n:{k:v.clone() for k,v in getattr(agent,n).state_dict().items()} for n in NETWORKS}
            for i in range(8):
                result=episode(agent,env,out/f'prefill/fragment-{i:03d}',i,.5,False);steps+=result['real_steps'];print('REAL DYNAMIC PREFILL',i+1,steps,flush=True)
            assert all(torch.equal(v,getattr(agent,n).state_dict()[k]) for n,d in frozen.items() for k,v in d.items())
            write(out/'prefill-proof.json',dict(real_fragments=8,real_transitions=steps,completed_episodes_claimed=0,networks_unchanged=True))
            for group in agent.param_net_optimiser.param_groups:group['lr']=0.
            with (out/'warmup-updates.jsonl').open('w',encoding='utf-8',buffering=1) as f:
                for _ in range(256):f.write(json.dumps(recorded_update(agent))+'\n')
            agent.param_net_optimiser=torch.optim.Adam(agent.param_net.parameters(),lr=.00001)
            for n in range(1,args.episodes+1):
                result=episode(agent,env,out/f'training/episode-{n:03d}',100+n,.2 if n<=8 else .1,True);steps+=result['real_steps'];completed=n
                write(out/'progress.json',dict(completed=n,real_steps=steps,updates=agent.update_count,last=result,wall_s=time.perf_counter()-started));print('REAL DYNAMIC TRAIN',n,steps,agent.update_count,result['center_mean'],flush=True)
                if n in (4,8,16,24,32) or n==args.episodes:legacy.checkpoint(agent,out/f'checkpoint-{n:03d}.pt',completed,steps,meta)
                if n==1:
                    proof=dict(networks={name:sorted(set(str(v.device) for v in getattr(agent,name).parameters())) for name in NETWORKS},gradients={name:sorted(set(str(v.grad.device) for v in getattr(agent,name).parameters() if v.grad is not None)) for name in ('actor_net','param_net')},real_steps=steps,updates=agent.update_count)
                    assert all(v==['cuda:0'] for x in ('networks','gradients') for v in proof[x].values());write(out/'gpu-proof.json',proof)
            legacy.checkpoint(agent,out/'last.pt',completed,steps,meta)
        finally:t.close();write(out/'status.json',dict(completed=completed,real_steps=steps,updates=agent.update_count,qualified=False))
if __name__=='__main__':main()
