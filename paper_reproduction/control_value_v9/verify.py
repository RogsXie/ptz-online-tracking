"""Implementation checks only; synthetic transitions never enter a live run."""
import json
from pathlib import Path
import tempfile
import numpy as np
import torch
from control_value_v9.agent import FixedPulseAgent
from control_value_v9.rollout import guard_selection,ValueReviewRequired
from control_coverage_v8.run import frozen,network_hashes,write_json,sha256
from control_coverage_v8.scenes import cases,layout
from control_v5.environment import transition_terminal
from integration.runtime import REPRO


def main():
    torch.set_num_threads(4)
    checks=[]
    for device in ('cpu','cuda'):
        agent=FixedPulseAgent(device=device)
        state=np.array([.8,-.4,-.5,.9,0,0,0,0],np.float32)
        before=network_hashes(agent)
        updates=[]
        for i in range(400):
            action=agent.select_action(state,True)
            assert action[0] in (0,1,2,3,4) and action[1]==.04
            assert agent.epsilon==.4
            agent.observe(state,action,.01,state,False)
            value=agent.update()
            if value is not None:updates.append(value)
        assert len(updates)==37 and agent.update_count==37
        assert all(u['parameter_objective'] is None for u in updates)
        after=network_hashes(agent)
        assert before['param_net']==after['param_net'] and before['param_net_target']==after['param_net_target']
        assert not agent.param_net_optimiser.state
        receipt={}
        with frozen(agent,receipt):
            for _ in range(10):agent.select_action(state,False)
        assert all(receipt[k] for k in ('networks_unchanged','counters_unchanged','sampling_rng_unchanged'))
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'agent.pt'
            agent.save(path)
            restored=FixedPulseAgent.load(path,device=device)
            assert network_hashes(agent)==network_hashes(restored)
            for _ in range(4):
                action=agent.select_action(state,True)
                other=restored.select_action(state,True)
                assert action==other
                for a in (agent,restored):a.observe(state,action,.01,state,False)
                assert agent.update()==restored.update()
            assert network_hashes(agent)==network_hashes(restored)
        selection_check=FixedPulseAgent(device=device)
        with torch.no_grad():
            for net,bias in ((selection_check.actor_net,[1.,10.,2.,3.,4.,100.,100.]),
                             (selection_check.actor_target,[9.,1.,8.,7.,6.,100.,100.])):
                net.base.layers[-1].weight.zero_()
                net.base.layers[-1].bias.copy_(torch.tensor(bias,device=device))
            # Online selects action1, target evaluates it as1; max target would9.
            value=selection_check.bootstrap(torch.as_tensor(state,device=device)[None])
            assert value.item()==1.
        checks.append(dict(device=device,fixed_pulse_and_action_mask=True,update_schedule_37_of400=True,
                           double_selection_evaluation_separation=True,four_network_freeze=True,
                           exact_checkpoint_rng_restore=True,parameter_optimizer_unused=True))
    guard=FixedPulseAgent(device='cpu');guard.update_count=256;guard.episode_q_excess=[]
    with tempfile.TemporaryDirectory() as folder:
        triggered=False
        try:
            for i in range(200):guard_selection(Path(folder),guard,{'greedy_q_bound_excess':2. if i<11 else -1.},True)
        except ValueReviewRequired:
            triggered=True
        assert triggered and (Path(folder)/'value-review.json').exists()
    assert transition_terminal({'terminated':False,'truncated':True}) is False
    plan=cases('static_center','train',60)
    assert len({layout(c['seed'],'train')['group'] for c in plan})==60
    checks.append(dict(q_guard_stops_without_clipping=True,actual_next_state_bootstrap_contract=True,
                       full60_cell_plan_with_constant_exploration=True))
    result=dict(passed=True,checks=checks,scope='Implementation only; no real-control success claim',
        source_sha256={p.name:sha256(p) for p in Path(__file__).parent.glob('*.py')})
    write_json(REPRO/'verification/value-redesign-v9-20261031/contract-checks.json',result)
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
