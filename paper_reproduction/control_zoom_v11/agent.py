"""Seven-action residual Double DQN for the declared RGB center-plus-scale reward."""
from copy import deepcopy
import torch
from torch import nn
from torch.nn import functional as F
from control_value_v9.agent import FixedPulseAgent, control_config as previous_config, PULSE_SECONDS
from dataclasses import replace
import numpy as np
from control_v6.temporal import DECAY_PER_SECOND
from control_v3.agent import require_finite


def potential_offset(states):
    cost = torch.where(states[:, 3] > 0, torch.linalg.vector_norm(states[:, :2] / 4, dim=1) + .8*states[:, 2].abs(), 2.)
    return 20 * cost


class PotentialQ(nn.Module):
    def __init__(self, residual):
        super().__init__()
        self.residual = residual

    def forward(self, states, parameters):
        return self.residual(states, parameters) + potential_offset(states)[:, None]


def control_config():
    return replace(previous_config(), reward_mode='active_zoom_double_dqn_v11')


class ZoomAgent(FixedPulseAgent):
    FORMAT = 'active_zoom_double_dqn_v11'

    def __init__(self, config=None, seed=20261114, device='cuda', allowed_actions=(0, 1, 2, 3, 4, 5, 6)):
        super().__init__(config or control_config(), seed, device, allowed_actions)
        self.actor_net = PotentialQ(self.actor_net).to(self.device)
        self.actor_target = deepcopy(self.actor_net).eval().requires_grad_(False)
        self.actor_optimiser = torch.optim.Adam(self.actor_net.parameters(), lr=self.config.actor_lr)

    def residual_target(self, states, rewards, next_states, dones, durations):
        factors = torch.exp(-DECAY_PER_SECOND * durations) * (~dones).float()
        with torch.no_grad():
            parameters = self._project(self.param_net(next_states))
            online = self.actor_net.residual(next_states, parameters)[:, self.allowed_actions]
            choices = online.argmax(1)
            residual_next = self.actor_target.residual(next_states, parameters)[:, self.allowed_actions].gather(1, choices[:, None]).squeeze(1)
            # Q=U+20*c preserves the declared joint reward's shaped TD residual,
            # including genuine termination if that is introduced in a later task.
            return rewards - potential_offset(states) + factors * (potential_offset(next_states) + residual_next)

    def update(self):
        if len(self.memory) < 256 or self.environment_steps % 4:
            return None
        states, actions, params, rewards, next_states, dones = [
            torch.as_tensor(a, device=self.device) for a in self.memory.sample(self.config.batch_size)]
        durations = params.gather(1, actions[:, None]).squeeze(1)
        if not bool(torch.all(torch.abs(durations - PULSE_SECONDS) < 1e-7)):
            raise ValueError('Non-fixed-duration replay')
        target = self.residual_target(states, rewards, next_states, dones, durations)
        predicted = self.actor_net.residual(states, params).gather(1, actions[:, None]).squeeze(1)
        loss = F.smooth_l1_loss(predicted, target)
        require_finite(loss, 'potential-factored Q loss')
        self.actor_optimiser.zero_grad(set_to_none=True)
        loss.backward()
        for p in self.actor_net.parameters():
            require_finite(p.grad, 'potential-factored Q gradient')
            p.grad.clamp_(-1, 1)
        self.actor_optimiser.step()
        for p in self.actor_net.parameters():
            require_finite(p, 'potential-factored updated weights')
        self.update_count += 1
        sync = self.update_count % self.config.target_update_interval == 0
        if sync:
            self.actor_target.load_state_dict(self.actor_net.state_dict())
        return dict(update=self.update_count, q_loss=float(loss.detach()), parameter_objective=None,
                    parameter_grad_norm=None, parameter_updates=0, target_sync=sync,
                    q_abs_max=float((predicted.detach() + potential_offset(states)).abs().max()),
                    target_abs_max=float((target + potential_offset(states)).abs().max()))

    def select_action(self, state, explore=True):
        action = super().select_action(state, explore)
        cost = float(np.linalg.norm(state[:2]/4) + .8*abs(state[2])) if state[3] > 0 else 2.
        upper = 20*cost + 1/DECAY_PER_SECOND
        self.last_selection.update(q_upper_bound=upper,greedy_q_bound_excess=max(self.last_selection['q_values'])-upper)
        return action

    def initialize_from_v10(self, path):
        checkpoint = torch.load(path,map_location=self.device,weights_only=False)
        if checkpoint['format'] != 'potential_factored_double_dqn_v10':
            raise ValueError('Expected v10 initialization checkpoint')
        if len(self.memory) or self.update_count or self.actor_optimiser.state:
            raise ValueError('Initialization requires empty replay and fresh optimizer')
        self.actor_net.load_state_dict(checkpoint['actor_net'])
        self.actor_target.load_state_dict(checkpoint['actor_net'])
        self.param_net.load_state_dict(checkpoint['param_net'])
        self.param_net_target.load_state_dict(checkpoint['param_net_target'])
        self.param_net.requires_grad_(False)
        self.param_net_target.requires_grad_(False)
