"""Action-isolated parameterized Q learning, with physical-second replay."""
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from control_learning.agent import ControlAgent, control_config
from hascam.config import PDQNConfig


def require_finite(value, label):
    if not bool(torch.isfinite(value).all()):
        raise FloatingPointError(f"Nonfinite {label}; stop before further optimizer steps")


class ActionIsolatedQ(nn.Module):
    """Evaluate Q_a(s,p_a) with other parameter slots fixed at the midpoint.

    The shared MLP is evaluated once per action; diagonal outputs alone survive.
    Thus dQ_a/dp_b is identically zero for b!=a, including target Q queries.
    """
    def __init__(self, base, midpoint):
        super().__init__()
        self.base = base
        self.midpoint = float(midpoint)

    def forward(self, states, parameters):
        require_finite(states, "Q state")
        require_finite(parameters, "Q parameters")
        count = parameters.shape[-1]
        mask = torch.eye(count, device=parameters.device, dtype=parameters.dtype)
        isolated = self.midpoint + mask[None] * (parameters[:, None, :] - self.midpoint)
        repeated = states[:, None, :].expand(-1, count, -1)
        values = self.base(repeated.reshape(-1, states.shape[-1]), isolated.reshape(-1, count))
        result = values.reshape(-1, count, count).diagonal(dim1=1, dim2=2)
        require_finite(result, "Q values")
        return result


class ImprovedAgent(ControlAgent):
    FORMAT = "isolated_pdqn_v3"

    def __init__(self, config=None, seed=20261102, device="cuda", allowed_actions=(0,1,2,3,4)):
        super().__init__(config or control_config(), seed, device, allowed_actions)
        midpoint = (self.config.min_duration + self.config.max_duration) / 2
        self.actor_net = ActionIsolatedQ(self.actor_net, midpoint).to(self.device)
        self.actor_target = deepcopy(self.actor_net).eval().requires_grad_(False)
        self.actor_optimiser = torch.optim.Adam(self.actor_net.parameters(), lr=self.config.actor_lr)

    def _project(self, parameters):
        require_finite(parameters, "raw duration proposals")
        return super()._project(parameters)

    def update(self):
        if len(self.memory) < max(256, self.config.batch_size):
            return None
        states, actions, params, rewards, next_states, dones = [
            torch.as_tensor(array, device=self.device) for array in self.memory.sample(self.config.batch_size)]
        require_finite(rewards, "replay rewards")
        with torch.no_grad():
            next_params = self._project(self.param_net_target(next_states))
            q_next = self.actor_target(next_states, next_params)[:, self.allowed_actions].max(dim=1).values
            target = rewards + (~dones).float() * self.config.gamma * q_next
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
            "gradient_query_max_s": float(parameter_leaf.detach().max()), "target_sync": sync}

    def save(self, path, metadata=None, include_replay=True):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        value = {"format": self.FORMAT, "config": asdict(self.config), "seed": self.seed,
            "allowed_actions": self.allowed_actions, "exploration_steps": self.exploration_steps,
            "environment_steps": self.environment_steps, "update_count": self.update_count,
            "completed_episodes": self.completed_episodes, "epsilon": self.epsilon,
            "rng_state": self.rng.bit_generator.state, "replay_rng_state": self.replay_rng.bit_generator.state,
            "python_rng_state": self.python_rng.getstate(), "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "metadata": metadata or {}}
        for name in ("actor_net", "actor_target", "param_net", "param_net_target", "actor_optimiser", "param_net_optimiser"):
            value[name] = getattr(self, name).state_dict()
        if include_replay:
            value["replay"] = self.memory.state_dict()
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(value, temporary)
        temporary.replace(path)

    @classmethod
    def load(cls, path, device="cuda", restore_replay=True):
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        if checkpoint.get("format") != cls.FORMAT:
            raise ValueError("A v3 checkpoint is required; historical weights/replay are not reinterpreted")
        agent = cls(PDQNConfig(**checkpoint["config"]), checkpoint["seed"], device, checkpoint["allowed_actions"])
        for name in ("actor_net", "actor_target", "param_net", "param_net_target", "actor_optimiser", "param_net_optimiser"):
            getattr(agent, name).load_state_dict(checkpoint[name])
        for name in ("environment_steps", "update_count", "completed_episodes", "epsilon", "exploration_steps"):
            setattr(agent, name, checkpoint[name])
        agent.rng.bit_generator.state = checkpoint["rng_state"]
        agent.replay_rng.bit_generator.state = checkpoint["replay_rng_state"]
        agent.python_rng.setstate(checkpoint["python_rng_state"])
        torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
        if device != "cpu" and checkpoint.get("cuda_rng_state") is not None:
            torch.cuda.set_rng_state_all([value.cpu() for value in checkpoint["cuda_rng_state"]])
        if restore_replay and "replay" in checkpoint:
            agent.memory.load_state_dict(checkpoint["replay"])
        agent.checkpoint_metadata = checkpoint.get("metadata", {})
        return agent
