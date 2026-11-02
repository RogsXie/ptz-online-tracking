"""Shared serial execution of real source-P-DQN episodes and frozen audits."""
import speed_bootstrap
from speed_bootstrap import REV,LEGACY,REPRO,configure_precision_project
import hashlib,json,math,time
from pathlib import Path
import torch,numpy as np
import train as legacy
from author_aligned.source import create_agent as original_agent,load_core,verify_source
from author_aligned.run import NETWORKS,OPTIMIZERS,restore_rng
from speed_agent import create_agent,transfer,PROFILE
from speed_environment import Environment as SpeedEnvironment
from environment import Environment as OldEnvironment
from speed_scenes import layout,materialize
from speed_independent_validation import case
from evaluate_audited import aggregate
from control_model_v13.detector import create_detector
from integration.experiment import make_extractor
from calibrate import write

legacy.materialize=materialize

class AuditMixin:
    def observe(self,*args,**kwargs):
        result=super().observe(*args,**kwargs)
        if hasattr(self,'audit_stream'):
            d=self.t.diagnostics
            self.audit_stream.write(json.dumps({k:d.get(k) for k in ('sim_time','gt_bbox_normalized','gt_in_frame','rotation','fov')})+'\n')
        return result

class AuditedSpeed(AuditMixin,SpeedEnvironment):pass
class AuditedOld(AuditMixin,OldEnvironment):pass

def frontend():
    detector=create_detector(checkpoint=REPRO/'weights/yolo26s_ue_overhead_v1.pt',device='cuda',threshold=.15,imgsz=960)
    assert detector.provenance['sha256']=='1dd0de55c1791dce289cc7dc1d8eb6c50e488b26d38941f4c20845a6cadb5ae2'
    detector.warmup();extractor=make_extractor('cuda')
    assert extractor.sha256=='57fb9c17d88911ea64390bf5427f43511435e7f88f6eed9dbc969d4b611e53cd'
    return detector,extractor

def resume(agent,path):
    saved=torch.load(path,map_location=agent.device,weights_only=False)
    assert saved['metadata'].get('speed_profile')==PROFILE['profile']
    for name in (*NETWORKS,*OPTIMIZERS):getattr(agent,name).load_state_dict(saved[name])
    for key,value in saved['replay'].items():setattr(agent.memory,key,value)
    for key in ('update_count','actions_count','epsilon'):setattr(agent,key,saved[key])
    restore_rng(agent,saved['rng']);return saved

def initialize(agent,out):
    parent=LEGACY/'exploration-support-original-updates/updates-1024.pt';meta=transfer(agent,parent)
    saved=torch.load(parent,map_location=agent.device,weights_only=False)
    frozen=legacy.snapshot(agent);old=original_agent(20261101,str(agent.device))
    for name in NETWORKS:getattr(old,name).load_state_dict(saved[name])
    states=torch.tensor(np.array([r[0] for r in saved['replay']['buffer'][::47]][:1024]),device=agent.device)
    with torch.inference_mode():
        p0=old.param_net(states);p1=agent.param_net(states);q0=old.actor_net(states,p0);q1=agent.actor_net(states,p1)
        errors=float((q0-q1).abs().max());mismatches=int((q0.argmax(1)!=q1.argmax(1)).sum())
        assert errors<1e-4 and mismatches==0
    legacy.finish_evaluation(agent,frozen)
    write(out/'initial-transfer-equivalence.json',dict(passed=True,read_only_old_states=len(states),max_q_error=errors,
        discrete_action_mismatches=mismatches,old_replay_added=0,scope='Initialization unit-transfer algebra only, no control-success claim'))
    return meta

def campaign(agent,env,output,checkpoint,split='development',count=80,old=False):
    output.mkdir(exist_ok=False);snap=legacy.snapshot(agent);agent.epsilon=0.;results=[]
    cfg=dict(checkpoint=str(checkpoint),checkpoint_sha256=hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
        split=split,count=count,training=False,execution_profile='released' if old else PROFILE['profile'])
    write(output/'configuration.json',cfg)
    try:
        for index in range(count):
            item=layout(split,index)
            if old:item['execution_profile']='released'
            folder=output/f'case-{index:03d}';temp=output/f'audit-{index:03d}.jsonl'
            with temp.open('w',encoding='utf-8',buffering=1) as audit:
                env.audit_stream=audit
                ep=legacy.episode(agent,env,item,folder,False,4,output,video=False)
            del env.audit_stream;temp.replace(folder/'offline-diagnostics.jsonl')
            metric=case(folder);rows=[json.loads(s) for s in (folder/'steps.jsonl').read_text().splitlines()]
            zoom=[b['sim_time'] for b,r in zip(rows,rows[1:]) if r['action'] in (5,6)]
            metric.update(first_zoom_s=zoom[0] if zoom else None,synchronous_processing_wall_s=ep['wall_s'],
                needs_zoom=bool(metric['initial_area'] and abs(math.log(metric['initial_area']/.04))>.25))
            results.append(metric);report=dict(configuration=cfg,episodes=results,qualification=aggregate(results,count))
            write(output/'independent-results.json',report)
            if index%10==0 or not metric['passed']:print('FROZEN',split,index,metric['passed'],metric['failures'],'settle',round(metric['settling_time'],3),flush=True)
    finally:
        legacy.finish_evaluation(agent,snap)
        write(output/'frozen-verification.json',dict(networks_targets_optimizers_replay_rng_unchanged=True,updates=agent.update_count))
    return report
