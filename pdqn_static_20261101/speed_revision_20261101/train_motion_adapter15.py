"""Unmodified source updates on actual dynamic replay; train only added columns."""
import speed_bootstrap
from speed_bootstrap import REV
import json,time,argparse
from pathlib import Path
import torch,numpy as np
import train as legacy
from author_aligned.source import recorded_update,verify_source
from author_aligned.run import NETWORKS
from motion_learning15 import transfer
from motion_adapter15 import create_agent,transform,PROFILE
from calibrate import write

def main():
    p=argparse.ArgumentParser();p.add_argument('--updates',type=int,default=2048);p.add_argument('--output',type=Path,default=REV/'dynamic15-adapter-v2');args=p.parse_args()
    legacy.exclusive();torch.set_num_threads(4);out=args.output;out.mkdir(exist_ok=False);a=create_agent();parent=transfer(a)
    path=REV/'dynamic15-training-v1/last.pt';s=torch.load(path,map_location='cpu',weights_only=False)
    for state,action,reward,nxt,done in s['replay']['buffer']:a.memory.push(transform(state),action,reward,transform(nxt),done)
    frozen={n:{k:v.clone() for k,v in getattr(a,n).state_dict().items()} for n in ('actor_net','param_net')}
    meta=dict(motion_profile=PROFILE,parent=parent,replay_source=str(path),real_dynamic_transitions_reused=len(a.memory),new_environment_steps=0,
        scope='Offline originalsourceRLupdates onpreviousrealUEtransitions, no actionteacher or trajectoryreplaycontrol. Trainonly7new inputcolumns,freezeall8oldstatecolumns,7parametercolumns,biases,anddeeplayers.',
        source_sha256=verify_source(),observation='RGB+owncamera angularvelocity2;zeroother5extra features;Gaussianproximityblend feedsnewfeaturesonly, no manualactionselection.',
        updates=args.updates,q_lr=.001,param_lr=.0001,critic_warmup=256)
    write(out/'configuration.json',meta)
    for group in a.param_net_optimiser.param_groups:group['lr']=0.
    start=time.perf_counter()
    with (out/'updates.jsonl').open('w',encoding='utf-8',buffering=1) as f:
        for i in range(1,args.updates+1):
            if (out/'STOP_REQUEST').exists():raise InterruptedError('Userstop atupdateboundary')
            if i==257:a.param_net_optimiser=torch.optim.Adam(a.param_net.parameters(),lr=.0001)
            record=recorded_update(a);f.write(json.dumps(record)+'\n')
            if i%128==0:write(out/'progress.json',dict(updates=i,real_transitions=len(a.memory),wall_s=time.perf_counter()-start));print('REAL REPLAY MOTION ADAPTER',i,len(a.memory),flush=True)
            if i in (512,1024,2048) or i==args.updates:legacy.checkpoint(a,out/f'updates-{i:04d}.pt',0,len(a.memory),meta)
    changed={}
    for name,values in frozen.items():
        current=getattr(a,name).state_dict()
        for key,before in values.items():
            after=current[key]
            if key=='layers.0.weight':
                assert torch.equal(before[:,:8],after[:,:8])
                if name=='actor_net':assert torch.equal(before[:,15:],after[:,15:])
                changed[name]=float((after[:,8:15]-before[:,8:15]).abs().sum())
            else:assert torch.equal(before,after),(name,key)
    assert all(v>0 for v in changed.values())
    write(out/'frozen-static-proof.json',dict(old_online_state_parameter_columns_and_all_deeplayers_bitwise_unchanged=True,new_motion_columns_abs_change=changed,updates=a.update_count,reused_real_transitions=len(a.memory),new_episodes_claimed=0))
    write(out/'status.json',dict(completed=True,updates=a.update_count,qualified=False));print('DYNAMIC ADAPTER READY',out,flush=True)
if __name__=='__main__':main()
