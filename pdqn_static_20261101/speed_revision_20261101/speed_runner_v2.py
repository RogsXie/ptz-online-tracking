"""Stable adaptation orchestration; all update operations remain author source."""
import speed_bootstrap
from speed_bootstrap import REV
import json,hashlib,math
from pathlib import Path
import torch
import train as legacy
from speed_runner import frontend,initialize,AuditedSpeed,AuditedOld
from speed_scenes import layout
from speed_independent_validation import case
from evaluate_audited import aggregate
from speed_agent import create_agent as original_config_agent
from author_aligned.run import NETWORKS,OPTIMIZERS,restore_rng
from calibrate import write

PROFILE=json.loads((REV/'training_profile_v2.json').read_text(encoding='utf-8'))

def create_agent():
    a=original_config_agent();a.actor_lr=.0001;a.param_net_lr=.00001
    for p in a.actor_optimiser.param_groups:p['lr']=a.actor_lr
    for p in a.param_net_optimiser.param_groups:p['lr']=a.param_net_lr
    return a

def resume(agent,path):
    s=torch.load(path,map_location=agent.device,weights_only=False)
    assert s['metadata']['speed_profile']==PROFILE['profile']
    for n in (*NETWORKS,*OPTIMIZERS):getattr(agent,n).load_state_dict(s[n])
    for k,v in s['replay'].items():setattr(agent.memory,k,v)
    for k in ('update_count','actions_count','epsilon'):setattr(agent,k,s[k])
    restore_rng(agent,s['rng']);return s

def campaign(agent,env,output,checkpoint,split='development',count=80,old=False,layout_factory=None):
    output.mkdir(exist_ok=False);snap=legacy.snapshot(agent);agent.epsilon=0.;results=[]
    cfg=dict(checkpoint=str(checkpoint),checkpoint_sha256=hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),split=split,count=count,
        training=False,execution_profile='released' if old else PROFILE['profile'],clock_scope='UE simulation action time; processing wall cost separate')
    write(output/'configuration.json',cfg);factory=layout_factory or layout
    try:
        for index in range(count):
            item=factory(split,index)
            if old:item['execution_profile']='released'
            folder=output/f'case-{index:03d}';temp=output/f'audit-{index:03d}.jsonl'
            with temp.open('w',encoding='utf-8',buffering=1) as audit:
                env.audit_stream=audit;ep=legacy.episode(agent,env,item,folder,False,4,output,video=False)
            del env.audit_stream;temp.replace(folder/'offline-diagnostics.jsonl')
            metric=case(folder);rows=[json.loads(s) for s in (folder/'steps.jsonl').read_text().splitlines()]
            zoom=[b['sim_time'] for b,r in zip(rows,rows[1:]) if r['action'] in (5,6)]
            metric.update(first_zoom_s=zoom[0] if zoom else None,synchronous_processing_wall_s=ep['wall_s'],
                needs_zoom=bool(metric['initial_area'] and abs(math.log(metric['initial_area']/.04))>.25))
            results.append(metric);report=dict(configuration=cfg,episodes=results,qualification=aggregate(results,count))
            write(output/'independent-results.json',report)
            if index%10==0 or not metric['passed']:print('FROZEN V2',split,index,metric['passed'],metric['failures'],'settle',round(metric['settling_time'],3),flush=True)
            if split=='development' and sum(not r['passed'] for r in results)>.05*count:
                report['futility_stop']=dict(completed=len(results),failed=sum(not r['passed'] for r in results),expected=count,qualified=False)
                write(output/'independent-results.json',report);break
    finally:
        legacy.finish_evaluation(agent,snap);write(output/'frozen-verification.json',dict(networks_targets_optimizers_replay_rng_unchanged=True,updates=agent.update_count))
    return report
