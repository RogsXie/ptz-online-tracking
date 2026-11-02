"""Balanced real UE replay and unchanged source P-DQN updates, one policy."""
import speed_bootstrap
from speed_bootstrap import REV,configure_precision_project
import json,time,warnings,hashlib,shutil
from pathlib import Path
from collections import Counter
import numpy as np,torch
import train as legacy
from author_aligned.source import recorded_update,verify_source
from author_aligned.run import NETWORKS
from author_aligned.environment import AtomicTransport
from integration.runtime import StandaloneArena
from motion_learning15 import transfer
from unified_core import ROOT,PROFILE,create_agent,frontend,Environment,bind_camera
from unified_scenes import scene
from calibrate import write

def rollout(a,env,folder,index,length,eps,learn,counts):
    path,item=scene(folder,index);env.length=length;env.cycle=[.1,.13,.16][(index//3)%3]
    state,row=env.reset(path,item['seed']);a.epsilon=eps;visible=centered=0
    with (folder/'steps.jsonl').open('w',encoding='utf-8',buffering=1) as f,(folder/'updates.jsonl').open('w',encoding='utf-8',buffering=1) as u:
        f.write(json.dumps(dict(initial=True,state=state.tolist(),**row))+'\n')
        for step in range(length):
            if (ROOT/'STOP_REQUEST').exists():raise InterruptedError('Requested action-boundary stop')
            bind_camera(a,env);act,param,params=a.choose_action(state,train=True)
            nxt,reward,done,row=env.step(int(act),float(param[0]));visible+=row['visible'];centered+=bool(row['visible'] and row['center_error']<=.025)
            row.update(input_state=state.tolist(),next_state=nxt.tolist(),all_parameters=params.tolist(),training_group=item['group'])
            f.write(json.dumps(row)+'\n');a.memory.push(state,np.r_[act,params],reward,nxt,done);counts[item['group']]+=1
            if learn and step%2==0:
                rec=recorded_update(a)
                if rec:u.write(json.dumps(rec)+'\n')
            state=nxt
            if done:break
    return dict(real_steps=step+1,visible=visible,centered=centered,scene=item,updates=a.update_count)

def main():
    legacy.exclusive();configure_precision_project();warnings.filterwarnings('ignore');torch.set_num_threads(4)
    out=ROOT/'pdqn_training_v1';out.mkdir(exist_ok=False);sources=out/'frozen_sources';sources.mkdir()
    names=['unified_core.py','unified_scenes.py','unified_perception.py','motion_learning15.py','train_unified_motion.py']
    for n in names:shutil.copy2(REV/n,sources/n)
    detector,extractor=frontend();a=create_agent();parent=transfer(a)
    meta=dict(motion_profile=PROFILE,parent=parent,source_sha256=verify_source(),source_files={n:hashlib.sha256((REV/n).read_bytes()).hexdigest() for n in names},
        architecture='Single15input P-DQN, seven exclusive actions and seven parameter outputs, four networks. No hierarchy or policy switch.',
        reward='20 imagecost progress -30 center -2 abslog(area/.04) +1.5 jointinner(center<=.015,logscale<=.25) -.005 nonStay, lost-50. RGB only.',
        curriculum='24x24 exploration fragments then48x120 fixed control segments; round-robin static/near_dynamic/far_dynamic balanced replay. Fragments/segments are not full30s qualified episodes.',
        scope='Speeds0/1.5/4.5/9m/s, multidirectional, constant/step ramps/stops/restarts, offsets/scales/FOV/heights. No GT/speed/path/schedule in states, actions or reward. Fixed training clock100/130/160ms; live evaluation real wall.',
        extensions='15 RGB/camera observations; own-optics gain4; short durations; physical-limit-only Q mask; frozen adapted YOLO with RGB-history ROI and association. Original update math unchanged.',
        detector=detector.provenance,reid=extractor.metadata(),gpu=torch.cuda.get_device_name(0))
    write(out/'configuration.json',meta);counts=Counter();completed=0;start=time.perf_counter()
    with StandaloneArena(out,headless=True) as arena:
        t=AtomicTransport(arena.port,hud=False)
        try:
            env=Environment(t,detector,extractor);frozen={n:{k:v.clone() for k,v in getattr(a,n).state_dict().items()} for n in NETWORKS}
            for i in range(24):
                result=rollout(a,env,out/f'support/fragment-{i:03d}',i,24,1.,False,counts)
                if (i+1)%3==0:print('UNIFIED REAL SUPPORT',i+1,dict(counts),flush=True)
            assert all(torch.equal(v,getattr(a,n).state_dict()[k]) for n,d in frozen.items() for k,v in d.items())
            write(out/'support-proof.json',dict(real_steps=sum(counts.values()),group_steps=dict(counts),four_networks_unchanged=True,full_episodes=0))
            for g in a.param_net_optimiser.param_groups:g['lr']=0.
            for _ in range(256):recorded_update(a)
            a.param_net_optimiser=torch.optim.Adam(a.param_net.parameters(),lr=.00003)
            for n in range(1,49):
                result=rollout(a,env,out/f'training/segment-{n:03d}',300+n-1,120,.35 if n<=12 else .2 if n<=24 else .1,True,counts);completed=n
                progress=dict(completed_segments=n,real_steps=sum(counts.values()),updates=a.update_count,group_steps=dict(counts),last=result,wall_s=time.perf_counter()-start)
                write(out/'progress.json',progress);print('UNIFIED TRAIN',n,progress['real_steps'],a.update_count,result['scene']['group'],result['visible'],result['centered'],flush=True)
                if n%12==0:legacy.checkpoint(a,out/f'checkpoint-{n:03d}.pt',n,sum(counts.values()),meta)
            legacy.checkpoint(a,out/'last.pt',completed,sum(counts.values()),meta)
        finally:
            t.close();write(out/'status.json',dict(completed_segments=completed,real_steps=sum(counts.values()),updates=a.update_count,group_steps=dict(counts),qualified=False))
if __name__=='__main__':main()
