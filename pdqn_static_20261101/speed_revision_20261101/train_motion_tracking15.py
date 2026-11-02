"""Fresh fine dynamic replay; no expert action, controller or groundtruth reward."""
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
from motion_learning15 import transfer
from motion_tracking15 import Environment,scene,create_agent,PROFILE
from calibrate import write

def rollout(a,env,folder,index,length,epsilon,learn):
    path,item=scene(folder,index);state,row=env.reset(path,item['seed']);a.epsilon=epsilon
    real=0;inner=0;visible=0
    with (folder/'steps.jsonl').open('w',encoding='utf-8',buffering=1) as f,(folder/'updates.jsonl').open('w',encoding='utf-8',buffering=1) as u:
        f.write(json.dumps(dict(initial=True,state=state.tolist(),**row))+'\n')
        for step in range(length):
            if (folder.parents[1]/'STOP_REQUEST').exists():raise InterruptedError('Stop atrealactionboundary')
            original_high=a.action_parameter_max_numpy.copy();a.action_parameter_max_numpy=np.array([5.,2.,2.,2.,2.,5.,5.],np.float32)
            try:act,param,params=a.choose_action(state,train=True)
            finally:a.action_parameter_max_numpy=original_high
            nxt,reward,done,row=env.step(int(act),float(param[0]));real+=1;visible+=row['visible'];inner+=bool(row['visible'] and row['center_error']<=.025)
            row.update(input_state=state.tolist(),next_state=nxt.tolist(),all_parameters=params.tolist());f.write(json.dumps(row)+'\n')
            a.memory.push(state,np.r_[act,params],reward,nxt,done)
            if learn and step%2==0:
                rec=recorded_update(a)
                if rec:u.write(json.dumps(rec)+'\n')
            state=nxt
            if done:break
    return dict(real_steps=real,visible=visible,centered=inner,updates=a.update_count)

def main():
    p=argparse.ArgumentParser();p.add_argument('--episodes',type=int,default=32);p.add_argument('--output',type=Path,default=REV/'dynamic15-tracking-v3');args=p.parse_args()
    legacy.exclusive();configure_precision_project();warnings.filterwarnings('ignore');torch.set_num_threads(4);out=args.output;out.mkdir(exist_ok=False)
    detector,extractor=frontend();a=create_agent();parent=transfer(a);meta=dict(motion_profile=PROFILE,parent=parent,source_sha256=verify_source(),
        objective='5costprogress-30center-2abslogscale+1jointcentered, lost-50; RGB-only. Originalsourceupdates Qlr.001 Paramlr.0001',
        scope='Newactualnearcenter dynamicdata at100msfixedcycle;microexplorationangular.25..2,learnerbounds.25..10. Notteacheractions. No olddifferentgainreplay.',
        phase='Trackingpolicy; originalstaticpolicy retained separately foracquisition',actuator='Squaredopticsgain4,zoomproportionalFOV,shortStay',detector=detector.provenance,reid=extractor.metadata())
    write(out/'configuration.json',meta);steps=completed=0;start=time.perf_counter()
    with StandaloneArena(out,headless=True) as arena:
        t=AtomicTransport(arena.port,hud=False)
        try:
            env=Environment(t,detector,extractor);frozen={n:{k:v.clone() for k,v in getattr(a,n).state_dict().items()} for n in NETWORKS}
            for i in range(64):
                result=rollout(a,env,out/f'fine-support/fragment-{i:03d}',i,8,1.,False);steps+=result['real_steps']
                if (i+1)%16==0:print('REAL DYNAMIC FINE',i+1,steps,flush=True)
            assert all(torch.equal(v,getattr(a,n).state_dict()[k]) for n,d in frozen.items() for k,v in d.items())
            write(out/'fine-support-proof.json',dict(real_fragments=64,real_steps=steps,complete_episodes_claimed=0,four_networks_unchanged=True))
            for group in a.param_net_optimiser.param_groups:group['lr']=0.
            for _ in range(256):recorded_update(a)
            a.param_net_optimiser=torch.optim.Adam(a.param_net.parameters(),lr=.0001)
            for n in range(1,args.episodes+1):
                result=rollout(a,env,out/f'training/episode-{n:03d}',100+n,80,.3 if n<=8 else .15 if n<=16 else .1,True);steps+=result['real_steps'];completed=n
                write(out/'progress.json',dict(completed=n,real_steps=steps,updates=a.update_count,last=result,wall_s=time.perf_counter()-start));print('REAL DYNAMIC TRACK TRAIN',n,steps,a.update_count,result['centered'],flush=True)
                if n in (8,16,24,32) or n==args.episodes:legacy.checkpoint(a,out/f'checkpoint-{n:03d}.pt',completed,steps,meta)
            legacy.checkpoint(a,out/'last.pt',completed,steps,meta)
        finally:t.close();write(out/'status.json',dict(completed=completed,real_steps=steps,updates=a.update_count,qualified=False))
if __name__=='__main__':main()
