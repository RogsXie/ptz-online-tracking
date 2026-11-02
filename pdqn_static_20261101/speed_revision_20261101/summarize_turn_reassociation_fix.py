"""Post-run comparisons: distinguish ID switches from authorized 3s forgetting."""
import argparse,json
from collections import Counter
from pathlib import Path
from audit_continuous_online import iou


def rows(path):return [json.loads(x) for x in path.read_text().splitlines() if x]


def missing_runs(path):
    samples=rows(path);runs=[];start=None
    for r in samples:
        if not r['visible'] and start is None:start=r['sim_time']
        if r['visible'] and start is not None:
            runs.append(dict(start_s=start,recovered_s=r['sim_time'],duration_s=r['sim_time']-start));start=None
    if start is not None:runs.append(dict(start_s=start,recovered_s=None,duration_s=samples[-1]['sim_time']-start))
    return runs


def generations(folder):
    diagnostics=json.loads((folder/'offline-identity-diagnostics.json').read_text())
    steps=rows(folder/'steps.jsonl');groups={};unknown=[]
    for step,diagnostic in zip(steps,diagnostics[1:]):
        r=step['observation'];gen=r['rgb_association']['target_generation']
        if not r['visible']:continue
        overlap,identity=max((iou(r['bbox'],t.get('gt_bbox_clipped_normalized')),t['id']) for t in diagnostic['targets'])
        if overlap<.3:unknown.append(r['sim_time']);continue
        groups.setdefault(gen,Counter())[identity]+=1
    return dict(matches_by_generation={str(k):dict(v) for k,v in groups.items()},
        within_generation_multiple_identities={str(k):dict(v) for k,v in groups.items() if len(v)>1},
        unmatched_visible_times=unknown,
        meaning='New highest-confidence identity after 3s continuous loss is authorized; different actors within one generation are an ID switch.')


def main():
    p=argparse.ArgumentParser();p.add_argument('--candidate',type=Path,required=True);args=p.parse_args()
    root=args.candidate.resolve();result=json.loads((root/'result.json').read_text())
    before=root.parent.parent/'control-v3-validation-20261102/CONTROL_V3_RESULT.json'
    # Locate the previous selected v3 result from the common experiment root.
    baseline=json.loads(before.read_text())
    output=dict(selected_validation=str(root),same_networks=json.loads((root.parent/'network-unchanged-proof.json').read_text()),
        source_unchanged_during_validation=result['source_unchanged_during_validation'],
        baseline={},candidate={},unseen_by_generation={})
    for label,old_label in [('online-turn','online-fast-turn'),('online-viaduct30','online-viaduct30')]:
        output['baseline'][label]=baseline['online_runs'][old_label]
        output['candidate'][label]=result['online'][label]
        output['candidate'][label]['missing_runs']=missing_runs(root/label/'perception-online.jsonl')
    for folder in sorted((root/'unseen-check').glob('case-*')):
        output['unseen_by_generation'][folder.name]=generations(folder)
    output['unseen_summary']=result['unseen_check']
    output['selected_as_local_default']=True;output['private_repository_modified']=False
    (root/'comparison.json').write_text(json.dumps(output,indent=2),encoding='utf-8')
    for label,r in output['candidate'].items():
        print(label,r['online']['visible_identity_matches'],'missing',r['online']['missing_outputs'],
            'generation',r['online']['maximum_target_generation'],'longest_missing_s',max((x['duration_s'] for x in r['missing_runs']),default=0.))
    print('GENERATION CHECK',json.dumps(output['unseen_by_generation']))


if __name__=='__main__':main()
