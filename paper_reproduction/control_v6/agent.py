"""Same action-isolated Q update, with discount based on executed time."""
from dataclasses import replace
import torch
from torch.nn import functional as F
from control_v5.agent import ContinuingAgent
from control_v4.agent import control_config as previous_config
from control_v3.agent import require_finite
from control_v6.temporal import DECAY_PER_SECOND, REFERENCE_GAMMA


def control_config():
    return replace(previous_config(),reward_mode="physical_time_rgb_v6")


class TemporalAgent(ContinuingAgent):
    FORMAT = "temporal_static_pdqn_v6"

    def __init__(self,config=None,seed=20261102,device="cuda",allowed_actions=(0,1,2,3,4)):
        config=config or control_config()
        if config.gamma != REFERENCE_GAMMA:
            raise ValueError("v6 temporal discount contract requires gamma0.98 at0.145s")
        super().__init__(config,seed,device,allowed_actions)

    def update(self):
        if len(self.memory) < max(256, self.config.batch_size):
            return None
        states, actions, params, rewards, next_states, dones = [
            torch.as_tensor(array, device=self.device) for array in self.memory.sample(self.config.batch_size)]
        require_finite(rewards, "replay rewards")
        with torch.no_grad():
            next_params = self._project(self.param_net_target(next_states))
            q_next = self.actor_target(next_states, next_params)[:, self.allowed_actions].max(dim=1).values
            durations = params.gather(1, actions[:, None]).squeeze(1)
            require_finite(durations, "executed replay durations")
            if bool(((durations < self.config.min_duration-1e-6) | (durations > self.config.max_duration+1e-6)).any()):
                raise ValueError("Replay duration outside physical contract")
            factors = torch.exp(-DECAY_PER_SECOND * durations)
            target = rewards + (~dones).float() * factors * q_next
        predicted = self.actor_net(states, params).gather(1, actions[:, None]).squeeze(1)
        loss = F.smooth_l1_loss(predicted, target)
        require_finite(loss, "Q loss")
        self.actor_optimiser.zero_grad(set_to_none=True)
        loss.backward()
        for parameter in self.actor_net.parameters():
            if parameter.grad is not None:
                require_finite(parameter.grad, "Q gradient")
                parameter.grad.clamp_(-1, 1)
        self.actor_optimiser.step()
        with torch.no_grad():
            raw_leaf = self.param_net(states)
            parameter_leaf = self._project(raw_leaf)
        parameter_leaf.requires_grad_(True)
        objective = self.actor_net(states, parameter_leaf)[:, self.allowed_actions].sum(dim=1).mean()
        gradient, = torch.autograd.grad(objective, parameter_leaf)
        require_finite(gradient, "duration gradient")
        for inactive in set(range(7)) - set(self.allowed_actions):
            gradient[:, inactive] = 0.
        proposed = self.param_net(states)
        inverted = self.invert_gradients(gradient, proposed.detach()).detach()
        surrogate = -(inverted * proposed).sum()
        self.param_net_optimiser.zero_grad(set_to_none=True)
        surrogate.backward()
        norm = torch.nn.utils.clip_grad_norm_(self.param_net.parameters(), self.config.param_grad_clip,
                                              error_if_nonfinite=True)
        self.param_net_optimiser.step()
        for parameter in self.param_net.parameters():
            require_finite(parameter, "updated parameter network")
        self.update_count += 1
        sync = self.update_count % self.config.target_update_interval == 0
        if sync:
            self.actor_target.load_state_dict(self.actor_net.state_dict())
            self.param_net_target.load_state_dict(self.param_net.state_dict())
        return {"update": self.update_count, "q_loss": float(loss.detach()),
            "parameter_objective": float(objective.detach()), "parameter_grad_norm": float(norm),
            "q_abs_max": float(predicted.detach().abs().max()), "target_abs_max": float(target.abs().max()),
            "raw_duration_min": float(raw_leaf.min()), "raw_duration_max": float(raw_leaf.max()),
            "gradient_query_min_s": float(parameter_leaf.detach().min()),
            "gradient_query_max_s": float(parameter_leaf.detach().max()), "target_sync": sync, "discount_min":float(factors.min()), "discount_max":float(factors.max())}

