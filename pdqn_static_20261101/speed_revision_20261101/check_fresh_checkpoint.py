"""CPU-only provenance/numerical check before spending GPU time on evaluation."""
import speed_bootstrap
from speed_bootstrap import REV
import argparse,hashlib,json
from pathlib import Path
import numpy as np,torch


def main():
    p=argparse.ArgumentParser();p.add_argument('--training',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();torch.set_num_threads(1);root=args.training
    initial=torch.load(root/'initial-random.pt',map_location='cpu',weights_only=False)
    final=torch.load(root/'last.pt',map_location='cpu',weights_only=False)
    assert final['metadata']['from_scratch'] and final['metadata']['parent_checkpoint'] is None
    assert not initial['replay']['buffer'] and initial['update_count']==0
    assert all(not initial[name]['state'] for name in ['actor_optimiser','param_net_optimiser'])
    proof={}
    for name in ['actor_net','actor_target','param_net','param_net_target']:
        assert all(torch.isfinite(v).all().item() for v in final[name].values())
        count=sum(not torch.equal(v,final[name][k]) for k,v in initial[name].items())
        assert count==10;proof[name]=dict(changed_tensors=count,all_finite=True)
    optimizer_steps={name:sorted({int(x['step']) for x in final[name]['state'].values()}) for name in ['actor_optimiser','param_net_optimiser']}
    assert all(x==[final['update_count']] for x in optimizer_steps.values())
    buffer=final['replay']['buffer'];assert len(buffer)==final['environment_steps']==10240
    for state,action,reward,next_state,done in buffer:
        assert state.shape==(15,) and next_state.shape==(15,) and action.shape==(8,)
        assert np.isfinite(state).all() and np.isfinite(next_state).all() and np.isfinite(action).all() and np.isfinite(reward)
    stable=REV/'unified_motion_20261101/pdqn_training_v1/last.pt'
    stable_hash=hashlib.sha256(stable.read_bytes()).hexdigest()
    assert stable_hash=='db8043ec93f916c35b7ab31b4ccf298489ceca94a4ec7290075d2e7e6c5b2434'
    result=dict(initial_empty_replay_and_optimizers=True,parent_pretrained_policy=None,
        final_real_transitions=final['environment_steps'],updates=final['update_count'],networks=proof,
        optimizer_steps=optimizer_steps,all_replay_vectors_finite=True,stable_checkpoint_unchanged_sha256=stable_hash,
        fresh_checkpoint_sha256=hashlib.sha256((root/'last.pt').read_bytes()).hexdigest(),
        fresh_curriculum_parent=final['metadata'].get('fresh_curriculum_parent'),
        meaning='Numerical/provenance check only; not a precision or online performance claim.')
    args.output.write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
