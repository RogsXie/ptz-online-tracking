"""Read-only scoring of a fresh-training attempt and genuine online outputs."""
import argparse,hashlib,json
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
from audit_continuous_online import summarize,read


def training(root):
    groups=defaultdict(lambda:dict(transitions=0,visible=0,within_8px=0,overrides=0,errors=[],rewards=[]))
    same_state=0;files=0
    for p in sorted(root.glob('*/*/steps.jsonl')):
        rows=read(p)
        if not rows:continue
        item=json.loads((p.parent/'scene.json').read_text())['experiment_coverage'];g=groups[item['family']]
        for previous,current in zip(rows,rows[1:]):
            if previous['next_state']!=current['input_state']:raise AssertionError(f'Replay state continuity mismatch: {p}')
            same_state+=1
        for row in rows:
            g['transitions']+=1;g['rewards'].append(row['reward'])
            g['overrides']+=int(row['proposal_action']!=row['executed_action'] or abs(row['proposal_duration_s']-row['executed_duration_s'])>1e-7)
            obs=row['observation'];g['visible']+=int(obs['visible'])
            if obs['visible']:
                px=float(np.hypot(obs['x']*480,obs['y']*270));g['errors'].append(px);g['within_8px']+=int(px<=8)
        files+=1
    summary={}
    for family,g in groups.items():
        errors=g.pop('errors');rewards=g.pop('rewards')
        summary[family]=dict(**g,center_median_p95_px=np.percentile(errors,[50,95]).tolist() if errors else None,mean_reward=float(np.mean(rewards)))
    return dict(completed_layouts=files,replay_state_continuity_pairs_verified=same_state,groups=summary,
        status=json.loads((root/'status.json').read_text()),initialization_proof=json.loads((root/'fresh-initialization-proof.json').read_text()),
        initial_sha256=hashlib.sha256((root/'initial-random.pt').read_bytes()).hexdigest(),
        candidate_sha256=hashlib.sha256((root/'last.pt').read_bytes()).hexdigest(),
        qualification='Training return/centering is not a held-out precision result. Visible errors exclude misses, reported separately.')


def online(root):
    result=summarize(root);outputs=read(root/'online-output/online-results.jsonl')[1:]
    proposals=[]
    for row in read(root/'decisions.jsonl'):
        guard=row['area_constraint']
        if guard.get('phase')!='track' or not row['visible'] or 'pdqn_action' not in guard:continue
        proposals.append(dict(unchanged=guard['pdqn_action']==row['action'] and abs(guard['pdqn_duration_s']-row['duration'])<1e-7,
            duration_delta_ms=abs(guard['pdqn_duration_s']-row['duration'])*1000.))
    result['normal_tracking_neural_proposals']=dict(count=len(proposals),
        unchanged_by_engineering=sum(x['unchanged'] for x in proposals),
        mean_duration_change_ms=float(np.mean([x['duration_delta_ms'] for x in proposals])) if proposals else None,
        interpretation='Read together with center error and misses; scale acquisition/search are excluded. Bounds are declared engineering, not neural precision evidence.')
    segments={}
    for label,start,end in [('motion_2_to_end',2.,1e9),('turn_9_to_12',9.,12.),('fast_12_to_end',12.,1e9),('parallel_24_to_end',24.,1e9)]:
        selected=[x for x in outputs if start<=x['result']['sim_time']<end]
        distances=[float(np.hypot(((x['result']['bbox'][0]+x['result']['bbox'][2])/2-.5)*960.,
            ((x['result']['bbox'][1]+x['result']['bbox'][3])/2-.5)*540.)) for x in selected if x['result']['visible']]
        segments[label]=dict(outputs=len(selected),missing=sum(not x['result']['visible'] for x in selected),
            center_median_p95_px=np.percentile(distances,[50,95]).tolist() if distances else None,
            within_8px=sum(d<=8 for d in distances))
    result['focused_segments']=segments
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--training',type=Path,required=True);p.add_argument('--online',type=Path,nargs='*',default=[])
    p.add_argument('--heldout',type=Path,nargs='*',default=[]);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    result=dict(training=training(args.training),online={x.name:online(x) for x in args.online},
        heldout={x.name:json.loads((x/'summary.json').read_text()) for x in args.heldout},
        limitation='Focused procedural and 30m/s viaduct checks only. No general reliability claim; runtime gains and neural retraining effects must be compared separately.')
    args.output.write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
