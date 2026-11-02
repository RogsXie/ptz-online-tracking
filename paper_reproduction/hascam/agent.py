"""Parameterised DQN used by HAS-CAM, audited against the author repository.

Q_Actor(s, seven durations)->seven Q-values; ParamNet(s)->seven durations.
The parameter objective is mean(sum_a Q), optimised with inverting gradients.
This is not PPO, a discretised-duration DQN, or a scalar-critic DDPG agent.
"""
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import random
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .config import PDQNConfig, epsilon


class MLP(nn.Module):
    def __init__(self, inputs, outputs, hidden):
        super().__init__()
        sizes = [inputs, *hidden, outputs]
        self.layers = nn.ModuleList(nn.Linear(a, b) for a, b in zip(sizes, sizes[1:]))

    def forward(self, values):
        for layer in self.layers[:-1]:
            values = F.relu(layer(values))
        return self.layers[-1](values)


class QActor(MLP):
    def __init__(self, cfg):
        super().__init__(cfg.state_dim + cfg.num_actions, cfg.num_actions, cfg.hidden_layers)

    def forward(self, state, all_parameters):
        return super().forward(torch.cat((state, all_parameters), dim=-1))


class ReplayBuffer:
    """Uniform replay without replacement, retaining all 7 action parameters."""
    def __init__(self, capacity, rng):
        self.capacity = capacity
        self.rng = rng
        self.buffer = []
        self.position = 0

    def push(self, state, action, parameters, reward, next_state, done):
        item = (state.copy(), int(action), parameters.copy(), float(reward), next_state.copy(), bool(done))
        if len(self.buffer) < self.capacity:
            self.buffer.append(item)
        else:
            self.buffer[self.position] = item
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        idx = self.rng.choice(len(self.buffer), size=batch_size, replace=False)
        rows = [self.buffer[i] for i in idx]
        states, actions, parameters, rewards, next_states, dones = zip(*rows)
        return (np.stack(states).astype(np.float32), np.array(actions, dtype=np.int64),
                np.stack(parameters).astype(np.float32), np.array(rewards, dtype=np.float32),
                np.stack(next_states).astype(np.float32), np.array(dones, dtype=np.bool_))

    def __len__(self):
        return len(self.buffer)

    def state_dict(self):
        return {"capacity": self.capacity, "position": self.position, "buffer": self.buffer}

    def load_state_dict(self, state):
        if state["capacity"] != self.capacity:
            raise ValueError("Replay capacity differs from checkpoint")
        self.position = state["position"]
        self.buffer = state["buffer"]


class HASCamAgent:
    def __init__(self, config=None, seed=0, device="cpu"):
        self.config = config or PDQNConfig()
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but this PyTorch build has no available CUDA device")
        self.seed = int(seed)
        # Local sampling generators avoid interference from detector/data code.
        self.rng = np.random.default_rng(self.seed)
        self.replay_rng = np.random.default_rng(self.seed + 1)
        self.python_rng = random.Random(self.seed)
        with torch.random.fork_rng():
            torch.manual_seed(self.seed)
            self.actor_net = QActor(self.config).to(self.device)
            self.param_net = MLP(self.config.state_dim, self.config.num_actions,
                                 self.config.hidden_layers).to(self.device)
            if self.config.feasible_param_init:
                nn.init.uniform_(self.param_net.layers[-1].weight, -.0003, .0003)
                nn.init.constant_(self.param_net.layers[-1].bias,
                                  (self.config.min_duration + self.config.max_duration) / 2)
        self.actor_target = deepcopy(self.actor_net).eval()
        self.param_net_target = deepcopy(self.param_net).eval()
        for network in (self.actor_target, self.param_net_target):
            network.requires_grad_(False)
        self.actor_optimiser = torch.optim.Adam(self.actor_net.parameters(), lr=self.config.actor_lr)
        self.param_net_optimiser = torch.optim.Adam(self.param_net.parameters(), lr=self.config.param_lr)
        self.memory = ReplayBuffer(self.config.replay_capacity, self.replay_rng)
        self.environment_steps = 0
        self.update_count = 0
        self.completed_episodes = 0
        self.epsilon = epsilon(0)
        self.last_action_parameters = None
        self.last_selection = None

    def _state(self, state):
        s = np.asarray(state, dtype=np.float32)
        if s.shape != (self.config.state_dim,) or not np.isfinite(s).all():
            raise ValueError("State must be a finite eight-vector from rendered detections")
        return s

    def _project(self, parameters):
        return parameters.clamp(self.config.min_duration, self.config.max_duration)

    def set_episode(self, completed_episodes):
        self.completed_episodes = int(completed_episodes)
        self.epsilon = epsilon(self.completed_episodes)

    def select_action(self, state, explore=True):
        s = self._state(state)
        with torch.inference_mode():
            tensor = torch.as_tensor(s, device=self.device).unsqueeze(0)
            raw = self.param_net(tensor)
            params = self._project(raw)
            q = self.actor_net(tensor, params)
            random_action = bool(explore and self.python_rng.random() < self.epsilon)
            if random_action:
                action = int(self.rng.integers(self.config.num_actions))
                parameters = self.rng.uniform(self.config.min_duration, self.config.max_duration,
                                              size=self.config.num_actions).astype(np.float32)
            else:
                action = int(q.argmax(dim=1).item())
                parameters = params[0].cpu().numpy().copy()
            duration = float(parameters[action])
            # Keep the RPC representation inside the exact bounds despite f32 .04.
            duration = min(self.config.max_duration, max(self.config.min_duration, duration))
            parameters[action] = duration
            self.last_action_parameters = parameters
            self.last_selection = {"random": random_action, "epsilon": self.epsilon,
                                   "q_values": q[0].cpu().tolist(),
                                   "q_parameter_vector": params[0].cpu().tolist(),
                                   "q_values_scope": "greedy_duration_proposals_before_epsilon_randomization",
                                   "raw_parameters": raw[0].cpu().tolist(),
                                   "parameters": parameters.tolist()}
            return action, duration

    def observe(self, state, action, reward, next_state, done, all_parameters=None):
        """action is (type,duration); preserve the whole parameter vector in replay."""
        state, next_state = self._state(state), self._state(next_state)
        action_type, duration = int(action[0]), float(action[1])
        if not 0 <= action_type < self.config.num_actions:
            raise ValueError("Action type must be 0..6")
        if not self.config.min_duration - 1e-7 <= duration <= self.config.max_duration + 1e-7:
            raise ValueError("Executed duration lies outside paper action bounds")
        if not np.isfinite(reward):
            raise ValueError("Reward must be finite")
        params = all_parameters if all_parameters is not None else self.last_action_parameters
        if params is None:
            raise ValueError("Replay needs all seven parameter proposals; call select_action or supply all_parameters")
        params = np.asarray(params, dtype=np.float32).copy()
        if params.shape != (self.config.num_actions,) or not np.isfinite(params).all():
            raise ValueError("Replay parameters must be a finite seven-vector")
        if np.any(params < self.config.min_duration - 1e-7) or np.any(params > self.config.max_duration + 1e-7):
            raise ValueError("Replay parameter vector is outside physical bounds")
        params[action_type] = duration
        self.memory.push(state, action_type, params, reward, next_state, done)
        self.environment_steps += 1

    def invert_gradients(self, gradients, parameters):
        lo, hi = self.config.min_duration, self.config.max_duration
        # This is source agent.py's bound-aware scaling; out-of-range values
        # reverse the outward gradient and push predictions back into bounds.
        return gradients * torch.where(gradients > 0, (hi - parameters) / (hi-lo),
                                        (parameters-lo) / (hi-lo))

    def update(self):
        """One optimisation. Caller schedules every 32 steps as in main.py."""
        if len(self.memory) < self.config.batch_size:
            return None
        arrays = self.memory.sample(self.config.batch_size)
        states, actions, params, rewards, next_states, dones = [
            torch.as_tensor(a, device=self.device) for a in arrays]
        with torch.no_grad():
            next_params = self.param_net_target(next_states)
            if self.config.project_durations:
                next_params = self._project(next_params)
            q_next = self.actor_target(next_states, next_params).max(dim=1).values
            target = rewards + (~dones).float() * self.config.gamma * q_next
        predicted = self.actor_net(states, params).gather(1, actions[:, None]).squeeze(1)
        actor_loss = F.smooth_l1_loss(predicted, target)
        self.actor_optimiser.zero_grad(set_to_none=True)
        actor_loss.backward()
        for param in self.actor_net.parameters():
            if param.grad is not None:
                param.grad.clamp_(-1, 1)
        self.actor_optimiser.step()

        # P-DQN maximises sum of Q-values wrt all action parameters. Inverting
        # gradients, rather than sigmoid/tanh outputs, constrains ParamNet.
        with torch.no_grad():
            parameter_leaf = self.param_net(states)
        parameter_leaf.requires_grad_(True)
        q_objective = self.actor_net(states, parameter_leaf).sum(dim=1).mean()
        parameter_gradient, = torch.autograd.grad(q_objective, parameter_leaf)
        proposed = self.param_net(states)
        inverted = self.invert_gradients(parameter_gradient, proposed.detach()).detach()
        surrogate_loss = -(inverted * proposed).sum()
        self.param_net_optimiser.zero_grad(set_to_none=True)
        surrogate_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.param_net.parameters(), self.config.param_grad_clip)
        self.param_net_optimiser.step()
        self.update_count += 1
        synchronised = self.update_count % self.config.target_update_interval == 0
        if synchronised:
            self.actor_target.load_state_dict(self.actor_net.state_dict())
            self.param_net_target.load_state_dict(self.param_net.state_dict())
        return {"q_loss": float(actor_loss.detach()), "parameter_objective": float(q_objective.detach()),
                "parameter_grad_norm": float(grad_norm), "update_count": self.update_count,
                "replay_size": len(self.memory), "target_synchronised": synchronised}

    def save(self, path, metadata=None, include_replay=True):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {"format": "hascam_pdqn_v1", "config": asdict(self.config), "seed": self.seed,
                      "actor_net": self.actor_net.state_dict(), "actor_target": self.actor_target.state_dict(),
                      "param_net": self.param_net.state_dict(), "param_net_target": self.param_net_target.state_dict(),
                      "actor_optimiser": self.actor_optimiser.state_dict(),
                      "param_net_optimiser": self.param_net_optimiser.state_dict(),
                      "environment_steps": self.environment_steps, "update_count": self.update_count,
                      "completed_episodes": self.completed_episodes, "epsilon": self.epsilon,
                      "rng_state": self.rng.bit_generator.state,
                      "replay_rng_state": self.replay_rng.bit_generator.state,
                      "python_rng_state": self.python_rng.getstate(),
                      "metadata": metadata or {}}
        if include_replay:
            checkpoint["replay"] = self.memory.state_dict()
        torch.save(checkpoint, path)

    @classmethod
    def load(cls, path, device="cpu", restore_replay=True):
        # Only load checkpoints created by this local training pipeline; the
        # replay contains numpy arrays, hence weights_only=False is necessary.
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        if checkpoint.get("format") != "hascam_pdqn_v1":
            raise ValueError("Not a HAS-CAM checkpoint created by this implementation")
        agent = cls(PDQNConfig(**checkpoint["config"]), checkpoint["seed"], device)
        for name in ("actor_net", "actor_target", "param_net", "param_net_target"):
            getattr(agent, name).load_state_dict(checkpoint[name])
        for name in ("actor_optimiser", "param_net_optimiser"):
            getattr(agent, name).load_state_dict(checkpoint[name])
        for name in ("environment_steps", "update_count", "completed_episodes", "epsilon"):
            setattr(agent, name, checkpoint[name])
        agent.rng.bit_generator.state = checkpoint["rng_state"]
        agent.replay_rng.bit_generator.state = checkpoint["replay_rng_state"]
        agent.python_rng.setstate(checkpoint["python_rng_state"])
        if restore_replay and "replay" in checkpoint:
            agent.memory.load_state_dict(checkpoint["replay"])
        agent.checkpoint_metadata = checkpoint.get("metadata", {})
        return agent
