"""Independent file-only checks of real fragment logs and saved replay data.

No learner, simulator transport, environment or action-selection code is imported.
"""
import argparse,hashlib,json,math
from collections import Counter
from pathlib import Path
import numpy as np
import torch

def equal(a,b):
    if torch.is_tensor(a):return torch.equal(a,b)
    if isinstance(a,np.ndarray):return np.array_equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent',type=Path,required=True);parser.add_argument('--data',type=Path,required=True)
    args=parser.parse_args();torch.set_num_threads(2)
    root=args.data.parent;parent=torch.load(args.parent,map_location='cpu',weights_only=False)
    data=torch.load(args.data,map_location='cpu',weights_only=False)
    rows=[];metadata=[];full=visible=0
    for folder in sorted((root/'fragments').glob('fragment-*')):
        meta=json.loads((folder/'initial.json').read_text());metadata.append(meta)
        fragment=[json.loads(s) for s in (folder/'steps.jsonl').read_text().splitlines()]
        assert len(fragment)==9 and fragment[0]['initial'] and fragment[0]['sim_time']==0.
        full+=bool(fragment[0]['bbox'] and all(.003<v<.997 for v in fragment[0]['bbox']))
        visible+=fragment[0]['visible']
        for before,step in zip(fragment,fragment[1:]):
            assert step['behavior_epsilon']==1. and len(step['all_parameters'])==7
            assert all(.3<=p<=5. for p in step['all_parameters'])
            assert step['all_parameters'][step['action']]==step['raw_parameter']
            assert abs(step['duration']-step['raw_parameter']/5.)<1e-12
            assert abs(step['sim_time']-before['sim_time']-step['duration'])<1e-5
            previous=before['state'] if before.get('initial') else before['next_state']
            assert equal(step['input_state'],previous)
            assert not step['done'] and not step['truncated']
            assert math.isfinite(step['reward'])
            rows.append(step)
    assert len(metadata)==640 and len(rows)==5120
    assert {m['seed'] for m in metadata}==set(range(32120000,32120640))
    assert Counter(m['quadrant'] for m in metadata)==Counter({i:160 for i in range(4)})
    networks=('actor_net','actor_target','param_net','param_net_target')
    optimizers=('actor_optimiser','param_net_optimiser')
    assert all(equal(parent[k],data[k]) for k in (*networks,*optimizers))
    assert data['update_count']==parent['update_count']==11238
    assert data['completed_episodes']==parent['completed_episodes']==480
    assert data['environment_steps']-parent['environment_steps']==5120
    assert data['actions_count']-parent['actions_count']==5120
    old=parent['replay'];new=data['replay'];assert len(old['buffer'])==old['position']
    assert len(new['buffer'])==new['position']==len(old['buffer'])+5120<new['capacity']
    assert equal(old['buffer'],new['buffer'][:len(old['buffer'])])
    for row,saved in zip(rows,new['buffer'][len(old['buffer']):]):
        state,action,reward,next_state,done=saved
        assert np.array_equal(state,np.array(row['input_state'],dtype=np.float32))
        assert np.array_equal(action,np.array([row['action'],*row['all_parameters']]))
        assert reward==row['reward'] and done==row['done']
        assert np.array_equal(next_state,np.array(row['next_state'],dtype=np.float32))
    counts=Counter(r['action'] for r in rows)
    report=dict(passed=True,scope='CPU file-only audit; no UE actions and no learned-controller imports',
        parent_sha256=hashlib.sha256(args.parent.read_bytes()).hexdigest(),
        data_sha256=hashlib.sha256(args.data.read_bytes()).hexdigest(),
        unique_actual_start_fragments=640,new_real_transitions=5120,completed_control_episodes_claimed=0,
        parent_completed_episodes=480,updates_unchanged=11238,
        all_four_networks_targets_and_both_optimizers_bitwise_unchanged=True,
        old_replay_prefix_bitwise_unchanged=True,every_new_replay_transition_matches_raw_log=True,
        uniform_exploration_action_counts=dict(sorted(counts.items())),initial_visible=visible,
        initial_full_detected_boxes=full,quadrants=dict(Counter(m['quadrant'] for m in metadata)),
        note='All actual fragments retained, including movement away from target; sampling frequencies do not establish independent control success')
    (root/'independent-data-audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
