"""Author P-DQN core with a consistent physical-duration gradient domain.

New experimental profile: finer actions, dense updates and action masks are
declared changes. Historical HASCamAgent and its checkpoints are untouched.
"""
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from hascam.agent import HASCamAgent
from hascam.config import PDQNConfig


def control_config():
    return PDQNConfig(min_duration=.04, max_duration=.25, gamma=.98,
        batch_size=128, replay_capacity=50000, actor_lr=.0005, param_lr=.0001,
        update_interval=1, target_update_interval=100)


class ControlAgent(HASCamAgent):
    def __init__(self, config=None, seed=20261031, device="cuda", allowed_actions=(0,1,2,3,4)):
        super().__init__(config or control_config(), seed, device)
        self.set_allowed_actions(allowed_actions)
        self.exploration_steps = 0

    def set_allowed_actions(self, actions):
        self.allowed_actions = tuple(int(action) for action in actions)
        if not self.allowed_actions or any(not 0 <= action < 7 for action in self.allowed_actions):
            raise ValueError("Invalid stage action mask")

    def _project(self, parameters):
        projected = super()._project(parameters).clone()
        if hasattr(self, "allowed_actions"):
            inactive = [action for action in range(7) if action not in self.allowed_actions]
            projected[..., inactive] = (self.config.min_duration + self.config.max_duration) / 2
        return projected

    def select_action(self, state, explore=True):
        state = self._state(state)
        with torch.inference_mode():
            raw = self.param_net(torch.as_tensor(state, device=self.device).unsqueeze(0))
            params = self._project(raw)
            q = self.actor_net(torch.as_tensor(state, device=self.device).unsqueeze(0), params)
            masked = q.clone()
            forbidden = [action for action in range(7) if action not in self.allowed_actions]
            masked[:, forbidden] = -torch.inf
            self.epsilon = max(.02, .8 - .78 * self.exploration_steps / 4000.)
            random_action = bool(explore and self.python_rng.random() < self.epsilon)
            parameters = params[0].cpu().numpy().copy()
            if random_action:
                action = int(self.rng.choice(self.allowed_actions))
                parameters = self.rng.uniform(self.config.min_duration, self.config.max_duration, 7).astype(np.float32)
                for inactive in set(range(7)) - set(self.allowed_actions):
                    parameters[inactive] = (self.config.min_duration + self.config.max_duration) / 2
            else:
                action = int(masked.argmax(dim=1).item())
            duration = min(self.config.max_duration, max(self.config.min_duration, float(parameters[action])))
            parameters[action] = duration
            self.last_action_parameters = parameters
            self.last_selection = {"random": random_action, "epsilon": self.epsilon,
                "q_values": q[0].cpu().tolist(), "q_parameter_vector": params[0].cpu().tolist(),
                "raw_parameters": raw[0].cpu().tolist(), "parameters": parameters.tolist(),
                "allowed_actions": list(self.allowed_actions), "duration_unit": "physical_seconds"}
            if explore:
                self.exploration_steps += 1
            return action, duration

    def update(self):
        if len(self.memory) < max(256, self.config.batch_size):
            return None
        states, actions, params, rewards, next_states, dones = [
            torch.as_tensor(array, device=self.device) for array in self.memory.sample(self.config.batch_size)]
        with torch.no_grad():
            next_params = self._project(self.param_net_target(next_states))
            q_next = self.actor_target(next_states, next_params)[:, self.allowed_actions].max(dim=1).values
            target = rewards + (~dones).float() * self.config.gamma * q_next
        predicted = self.actor_net(states, params).gather(1, actions[:, None]).squeeze(1)
        loss = F.smooth_l1_loss(predicted, target)
        self.actor_optimiser.zero_grad(set_to_none=True)
        loss.backward()
        for parameter in self.actor_net.parameters():
            if parameter.grad is not None:
                parameter.grad.clamp_(-1, 1)
        self.actor_optimiser.step()
        # Q has seen physical clipped seconds in replay/targets/execution.
        # Query its gradient in that SAME domain. Inversion still uses raw
        # outputs so an out-of-domain ParamNet can move back inside bounds.
        with torch.no_grad():
            raw_leaf = self.param_net(states)
            parameter_leaf = self._project(raw_leaf)
        parameter_leaf.requires_grad_(True)
        objective = self.actor_net(states, parameter_leaf)[:, self.allowed_actions].sum(dim=1).mean()
        gradient, = torch.autograd.grad(objective, parameter_leaf)
        for inactive in set(range(7)) - set(self.allowed_actions):
            gradient[:, inactive] = 0.
        proposed = self.param_net(states)
        inverted = self.invert_gradients(gradient, proposed.detach()).detach()
        surrogate = -(inverted * proposed).sum()
        self.param_net_optimiser.zero_grad(set_to_none=True)
        surrogate.backward()
        norm = torch.nn.utils.clip_grad_norm_(self.param_net.parameters(), self.config.param_grad_clip)
        self.param_net_optimiser.step()
        self.update_count += 1
        sync = self.update_count % self.config.target_update_interval == 0
        if sync:
            self.actor_target.load_state_dict(self.actor_net.state_dict())
            self.param_net_target.load_state_dict(self.param_net.state_dict())
        return {"update": self.update_count, "q_loss": float(loss.detach()),
            "parameter_objective": float(objective.detach()), "parameter_grad_norm": float(norm),
            "raw_duration_min": float(raw_leaf.min()), "raw_duration_max": float(raw_leaf.max()),
            "gradient_query_min_s": float(parameter_leaf.detach().min()),
            "gradient_query_max_s": float(parameter_leaf.detach().max()), "target_sync": sync}

    def save(self, path, metadata=None, include_replay=True):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        value = {"format": "gated_pdqn_v2", "config": asdict(self.config), "seed": self.seed,
            "allowed_actions": self.allowed_actions, "exploration_steps": self.exploration_steps,
            "environment_steps": self.environment_steps, "update_count": self.update_count,
            "completed_episodes": self.completed_episodes, "epsilon": self.epsilon,
            "rng_state": self.rng.bit_generator.state, "replay_rng_state": self.replay_rng.bit_generator.state,
            "python_rng_state": self.python_rng.getstate(), "metadata": metadata or {}}
        for name in ("actor_net", "actor_target", "param_net", "param_net_target", "actor_optimiser", "param_net_optimiser"):
            value[name] = getattr(self, name).state_dict()
        if include_replay:
            value["replay"] = self.memory.state_dict()
        torch.save(value, path)

    @classmethod
    def load(cls, path, device="cuda", restore_replay=True):
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        if checkpoint.get("format") != "gated_pdqn_v2":
            raise ValueError("Only this explicit v2 profile can be loaded; no legacy weight reinterpretation")
        agent = cls(PDQNConfig(**checkpoint["config"]), checkpoint["seed"], device, checkpoint["allowed_actions"])
        for name in ("actor_net", "actor_target", "param_net", "param_net_target", "actor_optimiser", "param_net_optimiser"):
            getattr(agent, name).load_state_dict(checkpoint[name])
        for name in ("environment_steps", "update_count", "completed_episodes", "epsilon", "exploration_steps"):
            setattr(agent, name, checkpoint[name])
        agent.rng.bit_generator.state = checkpoint["rng_state"]
        agent.replay_rng.bit_generator.state = checkpoint["replay_rng_state"]
        agent.python_rng.setstate(checkpoint["python_rng_state"])
        if restore_replay and "replay" in checkpoint:
            agent.memory.load_state_dict(checkpoint["replay"])
        agent.checkpoint_metadata = checkpoint.get("metadata", {})
        return agent
