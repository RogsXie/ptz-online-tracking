"""Fixed-pulse Double DQN feasibility candidate, not parameterized DQN."""
from dataclasses import replace
import numpy as np
import torch
from torch.nn import functional as F
from control_coverage_v8.agent import CoverageAgent, control_config as v8_config
from control_v3.agent import require_finite
from control_v6.temporal import DECAY_PER_SECOND

PULSE_SECONDS = .04
TRAIN_EPSILON = .4


def control_config():
    return replace(v8_config(), update_interval=4, reward_mode='fixed_pulse_double_dqn_v9')


class FixedPulseAgent(CoverageAgent):
    FORMAT = 'fixed_pulse_double_dqn_v9'

    def __init__(self, config=None, seed=20261106, device='cuda', allowed_actions=(0,1,2,3,4)):
        super().__init__(config or control_config(), seed, device, allowed_actions)
        if self.config.update_interval != 4:
            raise ValueError('v9 requires one Q update per four real transitions')
        # Retain four named networks for explicit freeze/checkpoint auditing.
        # The two parameter networks are inert fixed-duration constants, not learners.
        with torch.no_grad():
            self.param_net.layers[-1].weight.zero_()
            self.param_net.layers[-1].bias.fill_(PULSE_SECONDS)
        self.param_net.requires_grad_(False)
        self.param_net_target.load_state_dict(self.param_net.state_dict())
        self.param_net_target.requires_grad_(False)

    def select_action(self, state, explore=True):
        state = self._state(state)
        with torch.inference_mode():
            tensor = torch.as_tensor(state,device=self.device).unsqueeze(0)
            parameters = self._project(self.param_net(tensor))
            q = self.actor_net(tensor, parameters)[0]
            allowed = q[list(self.allowed_actions)]
            self.epsilon = TRAIN_EPSILON if explore else 0.
            randomized = bool(explore and self.python_rng.random() < self.epsilon)
            action = int(self.rng.choice(self.allowed_actions)) if randomized else self.allowed_actions[int(allowed.argmax())]
            values = parameters[0].cpu().numpy().copy()
            values[action] = PULSE_SECONDS
            self.last_action_parameters = values
            cost = float(np.linalg.norm(state[:2]/4)) if state[3] > 0 else 2.
            upper = 20*cost + 1/DECAY_PER_SECOND
            self.last_selection = dict(random=randomized, epsilon=self.epsilon, q_values=q.cpu().tolist(),
                parameters=values.tolist(), q_parameter_vector=parameters[0].cpu().tolist(),
                allowed_actions=list(self.allowed_actions), duration_unit='physical_seconds',
                fixed_pulse_seconds=PULSE_SECONDS, parameter_learning=False,
                q_upper_bound=upper, greedy_q_bound_excess=float(allowed.max())-upper)
            if explore:
                self.exploration_steps += 1
            return action, PULSE_SECONDS

    def bootstrap(self, states):
        parameters = self._project(self.param_net(states))
        choice = self.actor_net(states, parameters)[:, self.allowed_actions].argmax(1)
        return self.actor_target(states, parameters)[:, self.allowed_actions].gather(1, choice[:,None]).squeeze(1)

    def update(self):
        if len(self.memory) < 256 or self.environment_steps % 4:
            return None
        states, actions, params, rewards, next_states, dones = [
            torch.as_tensor(a, device=self.device) for a in self.memory.sample(self.config.batch_size)]
        durations = params.gather(1,actions[:,None]).squeeze(1)
        if not bool(torch.all(torch.abs(durations-PULSE_SECONDS)<1e-7)):
            raise ValueError('v9 replay contains non-fixed-duration actions')
        with torch.no_grad():
            target = rewards + (~dones).float()*torch.exp(-DECAY_PER_SECOND*durations)*self.bootstrap(next_states)
        predicted = self.actor_net(states,params).gather(1,actions[:,None]).squeeze(1)
        loss = F.smooth_l1_loss(predicted,target)
        require_finite(loss,'fixed-pulse Q loss')
        self.actor_optimiser.zero_grad(set_to_none=True)
        loss.backward()
        for parameter in self.actor_net.parameters():
            require_finite(parameter.grad,'fixed-pulse Q gradient')
            parameter.grad.clamp_(-1,1)
        self.actor_optimiser.step()
        for parameter in self.actor_net.parameters():
            require_finite(parameter,'updated fixed-pulse Q')
        self.update_count += 1
        sync = self.update_count % self.config.target_update_interval == 0
        if sync:
            self.actor_target.load_state_dict(self.actor_net.state_dict())
        return dict(update=self.update_count, q_loss=float(loss.detach()), parameter_objective=None,
                    parameter_grad_norm=None, parameter_updates=0, target_sync=sync,
                    q_abs_max=float(predicted.detach().abs().max()), target_abs_max=float(target.abs().max()))
