"""RGB-only precision reward on the same balanced real mixed replay."""
import speed_bootstrap
import json,math,time,hashlib,warnings
import torch
import train as legacy
from unified_core import ROOT,create_agent,resume
from author_aligned.source import recorded_update
from calibrate import write

def cost(s):return math.hypot(float(s[0]),float(s[1]))+.3*abs(math.log(float(s[2])/4.)) if s[3]>0 else 3.

def main():
    legacy.exclusive();warnings.filterwarnings('ignore');torch.set_num_threads(4)
    out=ROOT/'pdqn_precision_v2';out.mkdir(exist_ok=False);parent=ROOT/'pdqn_training_v1/last.pt';a=create_agent();saved=resume(a,parent)
    a.memory.buffer=[];a.memory.position=0
    for s,action,old_reward,nxt,done in saved['replay']['buffer']:
        if nxt[3]>0:
            center=math.hypot(float(nxt[0]),float(nxt[1]));scale=abs(math.log(float(nxt[2])/4.));inner=center<=.012 and scale<=.25
            reward=20*(cost(s)-cost(nxt))-60*center-3*scale+2.*inner-.005*(action[0]!=0)
        else:reward=-150.
        a.memory.push(s,action,float(reward),nxt,done)
    for g in a.actor_optimiser.param_groups:g['lr']=.0001
    for g in a.param_net_optimiser.param_groups:g['lr']=.00001
    meta=dict(saved['metadata']);meta.update(reward_revision='20imagecostprogress-60center-3abslogscale+2jointinner<=.012, lost-150. RGB only; yellow and acceptance remain.025.',
        parent=str(parent),parent_sha256=hashlib.sha256(parent.read_bytes()).hexdigest(),extra_original_updates=1536,new_real_steps=0,selection_scope='Precision refinement motivated by known9development result. No evaluation trajectories or heldout schedules added to replay.')
    write(out/'configuration.json',meta);start=time.perf_counter()
    with (out/'updates.jsonl').open('w',encoding='utf-8',buffering=1) as f:
        for i in range(1,1537):
            f.write(json.dumps(recorded_update(a))+'\n')
            if i%256==0:print('UNIFIED PRECISION',i,a.update_count,flush=True)
            if i in (512,1024,1536):legacy.checkpoint(a,out/f'extra-{i:04d}.pt',saved['completed_episodes'],saved['environment_steps'],meta)
    write(out/'status.json',dict(completed=True,extra_updates=1536,total_updates=a.update_count,real_replay=len(a.memory),new_real_steps=0,new_episodes=0,wall_s=time.perf_counter()-start))
if __name__=='__main__':main()
