"""Tighter RGB training margin on the same actual dynamic replay."""
import speed_bootstrap
from speed_bootstrap import REV
import argparse,json,math,time,hashlib
from pathlib import Path
import torch
import train as legacy
from motion_tracking15 import create_agent,resume,PROFILE
from author_aligned.source import recorded_update
from calibrate import write

def cost(s):return math.hypot(float(s[0]),float(s[1]))+.3*abs(math.log(float(s[2])/4.)) if s[3]>0 else 3.

def main():
    p=argparse.ArgumentParser();p.add_argument('--updates',type=int,default=1024);args=p.parse_args();legacy.exclusive();torch.set_num_threads(4)
    out=REV/'dynamic15-tracking-v4-precision';out.mkdir(exist_ok=False);parent=REV/'dynamic15-tracking-v3/checkpoint-032.pt';a=create_agent();saved=resume(a,parent)
    a.memory.buffer=[];a.memory.position=0
    for state,action,old_reward,nxt,done in saved['replay']['buffer']:
        if nxt[3]>0:
            center=math.hypot(float(nxt[0]),float(nxt[1]));scale=abs(math.log(float(nxt[2])/4.));inner=center<=.012 and scale<=.25
            reward=5*(cost(state)-cost(nxt))-60*center-2*scale+1.5*inner
        else:reward=-50.
        a.memory.push(state,action,float(reward),nxt,done)
    for group in a.actor_optimiser.param_groups:group['lr']=.0001
    for group in a.param_net_optimiser.param_groups:group['lr']=.00001
    meta=dict(saved['metadata']);meta.update(reward_revision='PrecisionRGBmargin.012,centerpenalty60; acceptance/colorremain.025. Reuseactualgain4transitions, no newdirections or expertlabels.',parent=str(parent),parent_sha256=hashlib.sha256(parent.read_bytes()).hexdigest(),extra_original_updates=args.updates,new_real_steps=0)
    write(out/'configuration.json',meta);start=time.perf_counter()
    with (out/'updates.jsonl').open('w',encoding='utf-8',buffering=1) as f:
        for i in range(1,args.updates+1):
            if (out/'STOP_REQUEST').exists():raise InterruptedError('Requestedupdateboundarystop')
            f.write(json.dumps(recorded_update(a))+'\n')
            if i%128==0:print('PRECISION RGB RL UPDATES',i,a.update_count,flush=True)
            if i in (256,512,1024) or i==args.updates:legacy.checkpoint(a,out/f'extra-{i:04d}.pt',saved['completed_episodes'],saved['environment_steps'],meta)
    write(out/'status.json',dict(completed=True,extra_updates=args.updates,total_updates=a.update_count,real_replay=len(a.memory),new_episodes=0,wall_s=time.perf_counter()-start))
if __name__=='__main__':main()
