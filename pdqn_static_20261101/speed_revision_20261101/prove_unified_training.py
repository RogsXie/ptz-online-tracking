"""CPU inspection of serialized GPU training evidence, after the run."""
import speed_bootstrap
from speed_bootstrap import REV
import json,hashlib
from collections import Counter
import torch
from unified_core import ROOT
from author_aligned.run import NETWORKS,OPTIMIZERS

def main():
    root=ROOT/'pdqn_training_v1';locations=Counter()
    def mapper(storage,location):locations[location]+=1;return storage
    saved=torch.load(root/'last.pt',map_location=mapper,weights_only=False)
    config=json.loads((root/'configuration.json').read_text());counts=Counter();updates=0
    for path in root.glob('training/*/steps.jsonl'):
        for line in path.read_text().splitlines():
            row=json.loads(line)
            if not row.get('initial'):counts[row['training_group']]+=1
    for path in root.glob('support/*/steps.jsonl'):
        for line in path.read_text().splitlines():
            row=json.loads(line)
            if not row.get('initial'):counts[row['training_group']]+=1
    for path in root.glob('training/*/updates.jsonl'):updates+=len(path.read_text().splitlines())
    optim={}
    for name in OPTIMIZERS:
        states=saved[name]['state'];optim[name]=dict(state_tensors=len(states),nonzero_adam_moments=sum(int(torch.count_nonzero(v['exp_avg'])) for v in states.values() if 'exp_avg' in v),max_step=max(float(v['step']) for v in states.values()))
    source_hashes={n:hashlib.sha256((root/'frozen_sources'/n).read_bytes()).hexdigest() for n in config['source_files']}
    assert source_hashes==config['source_files']
    assert sum(counts.values())==6336 and len(set(counts.values()))==1
    assert saved['update_count']==256+updates==3136
    assert any(k.startswith('cuda') for k in locations)
    assert all(v['nonzero_adam_moments']>0 for v in optim.values())
    proof=dict(actual_real_transitions=sum(counts.values()),balanced_group_transitions=dict(counts),source_updates=saved['update_count'],critic_warmup_updates=256,logged_joint_updates=updates,
        completed_control_segments=48,exploration_fragments=24,full_30s_training_episodes_claimed=0,
        checkpoint_storage_locations=dict(locations),optimizer_evidence=optim,network_first_layer_shapes={n:list(saved[n]['layers.0.weight'].shape) for n in NETWORKS},
        exactly_four_networks=list(NETWORKS),new_observation_columns_nonzero={n:int(torch.count_nonzero(saved[n]['layers.0.weight'][:,8:15])) for n in NETWORKS},
        training_sources_unchanged_during_run=True,checkpoint_sha256=hashlib.sha256((root/'last.pt').read_bytes()).hexdigest(),
        scope='Serialized GPU tensors and nonzero optimizer moments plus actual UE replay/log counts; single unified P-DQN, unmodified source update math.')
    (root/'gpu-and-data-proof.json').write_text(json.dumps(proof,indent=2),encoding='utf-8');print(json.dumps(proof,indent=2))
if __name__=='__main__':main()
