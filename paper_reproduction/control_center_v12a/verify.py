"""Prelaunch contracts using recorded data and synthetic interactions, no UE."""
from pathlib import Path
import json,tempfile,copy
from unittest.mock import patch
import numpy as np
import torch
from .agent import CenterAgent,SOURCE,SOURCE_SHA
from .scenes import training_cases,probe_cases,regression_cases,cases,STAGE
from .environment import CenterEnvironment
from .gates import assess_episode
from .rollout import frozen,evaluate,PolicyReviewRequired
from control_value_v10.agent import potential_offset
from control_v6.temporal import temporal_reward,DECAY_PER_SECOND
from control_coverage_v8.run import network_hashes,sha256,write_json

HERE=Path(__file__).parent
REPRO=HERE.parent


def complete(path):return [json.loads(x) for x in path.read_bytes().split(b'\n')[:-1] if x.strip()]


def main():
    torch.set_num_threads(4)
    cp=torch.load(SOURCE,map_location='cpu',weights_only=False)
    raw=[]
    for p in sorted((SOURCE.parent/'training').glob('*/steps.jsonl')):
        raw.extend(r for r in complete(p) if 'input_state' in r)
    assert len(raw)==30060==len(cp['replay']['buffer'])
    for row,item in zip(raw,cp['replay']['buffer']):
        s,a,p,r,n,d=item
        assert np.array_equal(s,np.array(row['input_state'],np.float32)) and np.array_equal(n,np.array(row['next_state'],np.float32))
        assert a==row['action_type'] and r==row['reward'] and d is False
        assert np.array_equal(p,np.array(row['policy_selection']['parameters'],np.float32))
    checked=[]
    for device in ('cpu','cuda'):
        a=CenterAgent(device=device);a.initialize_from_v10()
        assert len(a.memory)==a.update_count==a.environment_steps==0 and len(a.memory.history)==30060
        assert not a.actor_optimiser.state
        before=network_hashes(a);state=np.array([.8,-.4,-.2,.9,0,0,0,0],np.float32);updates=[];actions=set()
        for i in range(400):
            action=a.select_action(state,True);actions.add(action[0]);a.observe(state,action,.01,state,False)
            u=a.update()
            if u:
                updates.append(u);mix=u['rehearsal']
                assert mix['new_count']==mix['historical_count']==64
                assert len(set(mix['new_indices']))==len(set(mix['historical_indices']))==64
                assert max(mix['new_indices'])<i+1 and max(mix['historical_indices'])<30060
        assert len(updates)==37 and actions==set(range(5)) and updates[0]['new_environment_steps']==256
        assert not a.param_net_optimiser.state
        assert all(network_hashes(a)[n]==before[n] for n in ('param_net','param_net_target'))
        receipt={}
        with frozen(a,receipt):
            for _ in range(10):a.select_action(state,False)
        assert all(receipt[k] for k in ('networks_unchanged','counters_unchanged','sampling_rng_unchanged','historical_replay_unchanged','four_networks_eval','four_networks_requires_grad_false'))
        st=torch.tensor([[.8,-.4,-.5,.9,0,0,0,0],[0,0,0,.8,0,0,0,0],[0,0,.5,0,0,0,0,0]],device=device)
        ns=st.roll(1,0);rew=torch.tensor([1.,-.3,-2.],device=device);done=torch.tensor([False,False,True],device=device);dur=torch.full((3,),.04,device=device)
        with torch.no_grad():
            params=a._project(a.param_net(ns));chosen=a.actor_net(ns,params)[:,:5].argmax(1)
            ordinary=rew+(~done).float()*torch.exp(-DECAY_PER_SECOND*dur)*a.actor_target(ns,params).gather(1,chosen[:,None]).squeeze(1)
        assert torch.allclose(a.residual_target(st,rew,ns,done,dur)+potential_offset(st),ordinary,atol=2e-5)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'cp.pt';a.save(path);b=CenterAgent.load(path,device=device)
            assert network_hashes(a)==network_hashes(b) and len(b.memory)==400 and len(b.memory.history)==30060
            for _ in range(4):
                x=a.select_action(state,True);assert x==b.select_action(state,True)
                for ag in (a,b):ag.observe(state,x,.01,state,False)
                assert a.update()==b.update()
            assert network_hashes(a)==network_hashes(b)
        checked.append(dict(device=device,new_warmup256=True,real_update_interval4=True,mixed64_64=True,
            historical_separate=True,checkpoint_replay_rng_exact=True,center_bellman_equivalence=True,frozen_receipt=True,parameter_networks_frozen=True))
    # At fixed center, size must not change the phase-A reward or pass/fail gate.
    for area in (.005,.04,.09):
        rows=[dict(sim_time_s=i*.04,visible=True,matched=True,center_error=.01,x=.01,y=0.,area_ratio=area,
            fov=17.,action_type=0,duration_s=.04 if i else 0.,initial_zone_confirmed=True,zoom=float(np.log(area/.04))) for i in range(502)]
        assert assess_episode(STAGE,rows)['passed']
        rows[-1]['fov']=18.;assert not assess_episode(STAGE,rows)['passed']
    before=dict(visible=True,center_error=.1,area_ratio=.005)
    after=dict(visible=True,center_error=.02,area_ratio=.09)
    assert temporal_reward(before,after,3,.04)==temporal_reward(before|{'area_ratio':.04},after|{'area_ratio':.04},3,.04)
    assert len(training_cases())==len(probe_cases())==20 and len(probe_cases(True))==len(regression_cases())==60
    assert len({c['seed'] for c in training_cases()})==20
    for split in ('final','reserved_final_test'):
        try:cases(STAGE,split)
        except ValueError:pass
        else:raise AssertionError('Sealed split accessed')
    a=CenterAgent(device='cpu')
    def failed(*args,**kwargs):
        return dict(seed=12200000,passed=False,failures=['terminal_stability'],initial_task_ready=True,
            terminal_center_mean=.03,terminal_scale_log_error=1.,reward_sum=0.,direction_reversals=0,
            longest_loss_run_seconds=0.,stay_actions=501,saturated_nonstay=0,steps=501,max_q_bound_excess=-1.)
    with tempfile.TemporaryDirectory() as tmp:
        path=Path(tmp)/'probe'
        with patch('control_center_v12a.rollout.episode',side_effect=failed):
            try:evaluate([dict(seed=12200000,split='probe')],path,None,None,None,a,Path(tmp))
            except PolicyReviewRequired:pass
            else:raise AssertionError('Centered failure did not stop')
        f=json.loads((path/'freeze-receipt.json').read_text());r=json.loads((path/'report.json').read_text())
        assert f==r['freeze'] and f['historical_replay_unchanged'] and not r['complete']
    result=dict(passed=True,historical_training_replay_exact=True,historical_sha256=SOURCE_SHA,
        scope='CPU/CUDA implementation and historical-data contracts; no UE success claim',checks=checked,
        center_only_reward_and_gate=True,fixed_fov_gate=True,actual_exception_freeze_receipt=True,
        source_sha256={p.name:sha256(p) for p in HERE.glob('*.py')})
    write_json(REPRO/'verification/center-repair-v12a-20261031/contract-checks.json',result)
    print(json.dumps(result))


if __name__=='__main__':main()
