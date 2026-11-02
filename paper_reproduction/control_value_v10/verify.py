"""CPU/CUDA algorithm, restoration, and interrupted-probe contracts; no UE."""
from pathlib import Path
import json
import tempfile
from unittest.mock import patch
import numpy as np
import torch
from control_value_v10.agent import FactoredPotentialAgent, potential_offset
from control_value_v10.rollout import evaluate, PolicyReviewRequired
from control_coverage_v8.run import frozen, network_hashes, write_json, sha256
from control_v5.environment import transition_terminal
from integration.runtime import REPRO


def main():
    torch.set_num_threads(4)
    checks = []
    for device in ('cpu', 'cuda'):
        agent = FactoredPotentialAgent(device=device)
        state = np.array([.8, -.4, -.5, .9, 0, 0, 0, 0], np.float32)
        before = network_hashes(agent)
        updates = []
        for _ in range(400):
            action = agent.select_action(state, True)
            assert action[0] in range(5) and action[1] == .04 and agent.epsilon == .4
            agent.observe(state, action, .01, state, False)
            update = agent.update()
            if update is not None:
                updates.append(update)
        assert len(updates) == 37
        assert before['param_net'] == network_hashes(agent)['param_net']
        assert before['param_net_target'] == network_hashes(agent)['param_net_target']
        assert not agent.param_net_optimiser.state
        receipt = {}
        with frozen(agent, receipt):
            for _ in range(5):
                agent.select_action(state, False)
        assert all(receipt[k] for k in ('four_networks_eval', 'four_networks_requires_grad_false', 'networks_unchanged', 'counters_unchanged', 'sampling_rng_unchanged'))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'agent.pt'; agent.save(path)
            restored = FactoredPotentialAgent.load(path, device=device)
            assert network_hashes(agent) == network_hashes(restored)
            for _ in range(4):
                action = agent.select_action(state, True)
                assert action == restored.select_action(state, True)
                for x in (agent, restored):
                    x.observe(state, action, .01, state, False)
                assert agent.update() == restored.update()
            assert network_hashes(agent) == network_hashes(restored)
        checks.append(dict(device=device, actual_next_bootstrap=True, fixed_pulse_and_exploration=True,
                           update_schedule=True, exact_checkpoint_rng_restore=True, permanently_frozen_parameter_networks=True,
                           parameter_optimizer_unused=True, full_four_network_freeze=True))

    agent = FactoredPotentialAgent(device='cpu')
    def failed_case(*args, **kwargs):
        assert all(not getattr(agent, name).training and all(not p.requires_grad for p in getattr(agent, name).parameters())
                   for name in ('actor_net', 'actor_target', 'param_net', 'param_net_target'))
        return dict(seed=2200000, passed=False, failures=['total_direction_reversal_frequency'], coverage={'layer': 0},
                    terminal_center_mean=.015, reward_sum=1., direction_reversals=499, longest_loss_run_seconds=0.,
                    stay_actions=0, saturated_nonstay=0, steps=501, max_q_bound_excess=-1.)
    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory) / 'probe'
        with patch('control_value_v10.rollout.episode', side_effect=failed_case) as mocked:
            try:
                evaluate([dict(seed=2200000, split='probe'), dict(seed=2200001, split='probe')], output,
                         None, None, None, agent, Path(directory))
            except PolicyReviewRequired:
                pass
            else:
                raise AssertionError('Centered failure did not stop')
            assert mocked.call_count == 1
        report = json.loads((output / 'report.json').read_text())
        receipt = json.loads((output / 'freeze-receipt.json').read_text())
        assert not report['complete'] and report['total'] == 1 and report['planned_total'] == 2
        assert receipt['before'] == receipt['after'] and receipt['four_networks_requires_grad_false']
        assert receipt['counters_unchanged'] and receipt['sampling_rng_unchanged']
        with patch('control_value_v10.rollout.episode', side_effect=InterruptedError('test-only')):
            output = Path(directory) / 'interrupted'
            try:
                evaluate([dict(seed=2200000, split='probe')], output, None, None, None, agent, Path(directory))
            except InterruptedError:
                pass
            assert (output / 'freeze-receipt.json').exists()
            assert json.loads((output / 'report.json').read_text())['total'] == 0
    assert transition_terminal({'terminated': False, 'truncated': True}) is False
    checks.append(dict(immediate_centered_failure_stop=True, actual_freeze_receipt_on_exception=True,
                       partial_report_not_full_validation=True, no_second_case_after_review=True))
    result = dict(passed=True, checks=checks, scope='Implementation contracts only; no control validation',
                  source_sha256={p.name: sha256(p) for p in Path(__file__).parent.glob('*.py')})
    write_json(REPRO / 'verification/value-redesign-v10-20261031/contract-checks.json', result)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
