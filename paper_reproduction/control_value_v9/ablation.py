"""Paired fixed-data experiment, not independent online control validation."""
from copy import deepcopy
from pathlib import Path
import hashlib
import json
import random
import time
import numpy as np
import torch
from torch.nn import functional as F
from control_coverage_v8.agent import CoverageAgent
from control_coverage_v8.run import write_json, network_hashes
from control_v6.temporal import DECAY_PER_SECOND
from control_v3.agent import require_finite
from integration.runtime import REPRO


class AblationAgent(CoverageAgent):
    def __init__(self, seed, double_q, learn_parameters, device='cuda'):
        super().__init__(seed=seed, device=device)
        self.double_q = double_q
        self.learn_parameters = learn_parameters
        if not learn_parameters:
            self.param_net.requires_grad_(False)

    def update(self):
        states, actions, params, rewards, next_states, dones = [
            torch.as_tensor(a, device=self.device) for a in self.memory.sample(self.config.batch_size)]
        with torch.no_grad():
            if self.double_q:
                candidates = self._project(self.param_net(next_states))
                choice = self.actor_net(next_states, candidates)[:, self.allowed_actions].argmax(1)
                q_next = self.actor_target(next_states, candidates)[:, self.allowed_actions].gather(1, choice[:, None]).squeeze(1)
            else:
                candidates = self._project(self.param_net_target(next_states))
                q_next = self.actor_target(next_states, candidates)[:, self.allowed_actions].max(1).values
            durations = params.gather(1, actions[:, None]).squeeze(1)
            target = rewards + (~dones).float() * torch.exp(-DECAY_PER_SECOND * durations) * q_next
        predicted = self.actor_net(states, params).gather(1, actions[:, None]).squeeze(1)
        loss = F.smooth_l1_loss(predicted, target)
        require_finite(loss, 'ablation Q loss')
        self.actor_optimiser.zero_grad(set_to_none=True)
        loss.backward()
        for parameter in self.actor_net.parameters():
            require_finite(parameter.grad, 'ablation Q gradient')
            parameter.grad.clamp_(-1, 1)
        self.actor_optimiser.step()
        objective = None
        if self.learn_parameters:
            with torch.no_grad():
                leaf = self._project(self.param_net(states))
            leaf.requires_grad_(True)
            value = self.actor_net(states, leaf)[:, self.allowed_actions].sum(1).mean()
            gradient, = torch.autograd.grad(value, leaf)
            for inactive in set(range(7)) - set(self.allowed_actions):
                gradient[:, inactive] = 0.
            proposed = self.param_net(states)
            inverted = self.invert_gradients(gradient, proposed.detach()).detach()
            surrogate = -(inverted * proposed).sum()
            self.param_net_optimiser.zero_grad(set_to_none=True)
            surrogate.backward()
            torch.nn.utils.clip_grad_norm_(self.param_net.parameters(), 10., error_if_nonfinite=True)
            self.param_net_optimiser.step()
            objective = float(value.detach())
        self.update_count += 1
        if self.update_count % self.config.target_update_interval == 0:
            self.actor_target.load_state_dict(self.actor_net.state_dict())
            self.param_net_target.load_state_dict(self.param_net.state_dict())
        return {'q_loss': float(loss.detach()), 'parameter_objective': objective}


def complete_jsonl(path):
    data = Path(path).read_bytes()
    return [json.loads(line) for line in data[:data.rfind(b'\n') + 1].splitlines() if line.strip()]


def image_cost(states):
    return torch.where(states[:, 3] > 0, torch.linalg.vector_norm(states[:, :2]/4, dim=1), 2.)


def diagnose(agent, states):
    values = []
    with torch.inference_mode():
        for part in torch.as_tensor(states, device=agent.device).split(256):
            params = agent._project(agent.param_net(part))
            q = agent.actor_net(part, params)[:, agent.allowed_actions]
            cost = image_cost(part)
            upper = 20*cost + 1/DECAY_PER_SECOND
            selected, choice = q.max(1)
            for i in range(len(part)):
                values.append((float(selected[i]-upper[i]), bool(cost[i] <= .025), int(choice[i])))
    excess = np.array([r[0] for r in values])
    near = [r for r in values if r[1]]
    return dict(states=len(values), bound_violations=int((excess > .01).sum()),
        violation_fraction=float((excess > .01).mean()), max_excess=float(excess.max()),
        near_center_states=len(near), near_center_stay_fraction=sum(r[2] == 0 for r in near)/len(near) if near else None)


def main():
    torch.set_num_threads(4)
    output = REPRO/'verification/value-redesign-v9-20261031/offline-ablation'
    output.mkdir(parents=True, exist_ok=False)
    source = REPRO/'runs/coverage-static-v8-attempt1'
    train, holdout, manifest = [], [], []
    for folder in sorted((source/'training').glob('round-*')):
        index = int(folder.name.split('-')[1])
        rows = complete_jsonl(folder/'steps.jsonl')[1:]
        manifest.append({'round':index, 'holdout':index % 5 == 0,
                         'sha256':hashlib.sha256((folder/'steps.jsonl').read_bytes()).hexdigest(), 'steps':len(rows)})
        if index % 5 == 0:
            holdout.extend(r['input_state'] for r in rows)
        else:
            train.extend((np.array(r['input_state'],np.float32), int(r['action_type']),
                np.array(r['policy_selection']['parameters'],np.float32), float(r['reward']),
                np.array(r['next_state'],np.float32), bool(r['terminated'])) for r in rows)
    # All probe states are diagnostics only; none enters optimization.
    diagnostic = []
    for folder in sorted((source/'probe-0020').glob('seed*')):
        diagnostic.extend(r['input_state'] for r in complete_jsonl(folder/'steps.jsonl')[1:])
    seeds = [20261103,20261104,20261105]
    arms = [(False, True), (True, True), (False, False), (True, False)]
    order = []
    for seed in seeds:
        block = list(arms)
        random.Random(seed + 31).shuffle(block)
        order.extend((seed,*arm) for arm in block)
    spec = dict(seeds=seeds, order=order, updates_per_arm=1000, batch=128,
        train_transitions=len(train), holdout_states=len(holdout), probe_diagnostic_states=len(diagnostic),
        source_manifest=manifest, units='three paired initialization blocks; transitions are not independent runs',
        live_candidate_gate='double+frozen: every seed finite, holdout and probe Q violation_fraction<=0.01 and max_excess<=1 after1000updates',
        no_online_control_claim=True, source_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    write_json(output/'design.json', spec)
    results = []
    for seed,double_q,learn_parameters in order:
        name=f'seed{seed}-'+('double' if double_q else 'max')+'-'+('learn' if learn_parameters else 'frozen')
        agent = AblationAgent(seed,double_q,learn_parameters)
        agent.memory.buffer=deepcopy(train)
        agent.memory.position=len(train)
        before=network_hashes(agent)
        started=time.perf_counter()
        history=[]
        for update in range(1,1001):
            metrics=agent.update()
            if update % 250 == 0:
                row=dict(update=update,**metrics,holdout=diagnose(agent,holdout),probe_diagnostic=diagnose(agent,diagnostic))
                history.append(row)
                write_json(output/'progress.json',dict(arm=name,finished_arms=len(results),total_arms=len(order),**row))
                print(json.dumps({'arm':name,'update':update,'holdout':row['holdout'],'probe':row['probe_diagnostic']}),flush=True)
        after=network_hashes(agent)
        result=dict(arm=name,seed=seed,double_q=double_q,learn_parameters=learn_parameters,
                    initial_hashes=before,final_hashes=after,history=history,wall_seconds=time.perf_counter()-started,
                    parameter_network_unchanged=before['param_net']==after['param_net'])
        if not learn_parameters:
            assert result['parameter_network_unchanged']
        write_json(output/(name+'.json'),result)
        results.append(result)
        del agent
    candidates=[r for r in results if r['double_q'] and not r['learn_parameters']]
    eligible=all(r['history'][-1][split]['violation_fraction']<=.01 and r['history'][-1][split]['max_excess']<=1
                 for r in candidates for split in ('holdout','probe_diagnostic'))
    report=dict(completed=True,results=results,live_candidate_eligible=eligible,
                interpretation='Fixed-data mechanism screening only; frozen duration is the same initialized proposal network, not a fixed-pulse deployment. No control qualification.')
    write_json(output/'report.json',report)
    print(json.dumps({'completed':True,'live_candidate_eligible':eligible}),flush=True)


if __name__=='__main__':
    main()
