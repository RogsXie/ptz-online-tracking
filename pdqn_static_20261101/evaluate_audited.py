"""Fresh frozen UE evaluation with separate, matched offline GT evidence."""
import bootstrap
from bootstrap import HERE,REPRO
import argparse,json,warnings,hashlib
from pathlib import Path
import torch
from author_aligned.environment import AtomicTransport
from author_aligned.source import create_agent
from control_model_v13.detector import create_detector
from integration.runtime import StandaloneArena
from integration.experiment import make_extractor
from environment import Environment
from evaluation_scenes import layout
from train import exclusive,resume,snapshot,finish_evaluation,episode
from independent_validation import case
from calibrate import write

class AuditedEnvironment(Environment):
    def observe(self,*args,**kwargs):
        result=super().observe(*args,**kwargs)
        # Write-only side channel: this diagnostic data never returns to policy.
        self.audit_stream.write(json.dumps(dict(sim_time=self.t.diagnostics['sim_time'],
            gt_bbox_normalized=self.t.diagnostics.get('gt_bbox_normalized'),
            gt_in_frame=self.t.diagnostics.get('gt_in_frame'),
            rotation=self.t.diagnostics['rotation'],fov=self.t.diagnostics['fov']))+'\n')
        return result

def aggregate(results,expected):
    groups={}
    checks={'complete':len(results)==expected,'unique_seeds':len(set(r['seed'] for r in results))==len(results),
        'all_indices':set(r['index'] for r in results)==set(range(expected)),
        'overall':sum(r['passed'] for r in results)>=.95*expected}
    for name,minimum in [('quadrant',.95),('layer',.90),('scale_group',.95)]:
        groups[name]={}
        for value in sorted(set(r[name] for r in results)):
            members=[r for r in results if r[name]==value];passed=sum(r['passed'] for r in members)
            groups[name][value]=dict(passed=passed,count=len(members))
            checks[f'{name}-{value}']=passed>=minimum*len(members)
    cells={}
    for r in results:
        key=(r['quadrant'],r['layer'],r['scale_group']);cells.setdefault(key,[]).append(r)
    if expected>=160:checks['all_cells']=len(cells)==80 and all(sum(r['passed'] for r in v)>=1 for v in cells.values())
    return dict(qualified=all(checks.values()),checks=checks,groups=groups,passed=sum(r['passed'] for r in results),count=len(results))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path);p.add_argument('--count',type=int,default=160)
    p.add_argument('--split',choices=['development','validation','final_test'],default='development')
    p.add_argument('--validation-report',type=Path);args=p.parse_args()
    checkpoint_hash=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    if args.split=='final_test':
        if not args.validation_report:raise ValueError('Independent validation report required before final test')
        report=json.loads(args.validation_report.read_text())
        if not report['qualification']['qualified'] or report['qualification']['count']!=160:raise ValueError('Validation did not qualify')
        if report.get('configuration',{}).get('split')!='validation' or report['configuration'].get('checkpoint_sha256')!=checkpoint_hash:
            raise ValueError('Validation report is not for this exact frozen checkpoint')
        frozen_report=args.validation_report.parent/'frozen-verification.json'
        if not frozen_report.exists() or not json.loads(frozen_report.read_text()).get('networks_targets_optimizers_replay_rng_unchanged'):
            raise ValueError('Validation state freeze is not verified')
    exclusive();warnings.filterwarnings('ignore');torch.set_num_threads(4);root=args.output.resolve();root.mkdir(exist_ok=False)
    detector=create_detector(checkpoint=REPRO/'weights/yolo26s_ue_overhead_v1.pt',device='cuda',threshold=.15,imgsz=960)
    detector.warmup();extractor=make_extractor('cuda');agent=create_agent(20261101,'cuda');completed,steps=resume(agent,args.checkpoint)
    frozen=snapshot(agent);agent.epsilon=0.;results=[]
    configuration=dict(checkpoint=str(args.checkpoint),checkpoint_sha256=checkpoint_hash,completed=completed,
        split=args.split,count=args.count,training=False,
        coverage_and_motion_contract=json.loads((HERE/'heldout_coverage_declaration.json').read_text()))
    write(root/'configuration.json',configuration)
    try:
        with StandaloneArena(root,headless=True) as arena:
            t=AtomicTransport(arena.port,hud=False)
            try:
                env=AuditedEnvironment(t,detector,extractor)
                for index in range(args.count):
                    folder=root/f'case-{index:03d}';folder.mkdir()
                    # episode owns folder creation; diagnostic stream lives outside then is moved in.
                    folder.rmdir()
                    temp=root/f'audit-{index:03d}.jsonl'
                    with temp.open('w',buffering=1) as audit:
                        env.audit_stream=audit
                        episode(agent,env,layout(args.split,index),folder,False,32,root,video=True)
                    temp.replace(folder/'offline-diagnostics.jsonl')
                    result=case(folder);results.append(result)
                    print('INDEPENDENT',args.split,index,result['passed'],result['failures'],result['terminal_center_mean'],result['terminal_logscale_mean'],flush=True)
                    write(root/'independent-results.json',dict(configuration=configuration,episodes=results,qualification=aggregate(results,args.count)))
            finally:t.close()
    finally:
        finish_evaluation(agent,frozen)
        assert hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()==checkpoint_hash
        write(root/'frozen-verification.json',dict(networks_targets_optimizers_replay_rng_unchanged=True,updates=agent.update_count))

if __name__=='__main__':main()
