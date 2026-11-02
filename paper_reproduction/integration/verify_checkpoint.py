"""Verify real trained parameters/replay/checkpoint inference without a simulator."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from hascam.agent import HASCamAgent
from hascam.config import PDQNConfig


def verify(path, output):
    agent = HASCamAgent.load(path, device="cpu")
    initial = HASCamAgent(agent.config, agent.seed, device="cpu")
    differences = {}
    for name in ("actor_net", "param_net", "actor_target", "param_net_target"):
        current_state = getattr(agent, name).state_dict()
        initial_state = getattr(initial, name).state_dict()
        differences[name] = float(torch.sqrt(sum(
            (current_state[k] - initial_state[k]).double().square().sum() for k in current_state)))
    states = [np.array([-2, -2, -1, 0, 0, 0, 0, 0], np.float32),
              np.array([.5, -.2, .01, .7, 3/7, .2, .05, -.03], np.float32),
              np.array([0, 0, .05, .95, 0, .04, 0, 0], np.float32)]
    first = [agent.select_action(s, explore=False) for s in states]
    loaded_again = HASCamAgent.load(path, device="cpu")
    second = [loaded_again.select_action(s, explore=False) for s in states]
    valid = all(0 <= action < 7 and .04 <= duration <= 1 for action, duration in first)
    result = {"checkpoint": str(Path(path).resolve()), "environment_steps": agent.environment_steps,
              "completed_episodes": agent.completed_episodes, "updates": agent.update_count,
              "replay_size": len(agent.memory), "epsilon": agent.epsilon,
              "parameter_l2_change_from_seeded_initialization": differences,
              "trained_q_and_parameter_nets_changed": differences["actor_net"] > 0 and differences["param_net"] > 0,
              "independent_reload_greedy_actions_identical": first == second,
              "legal_hybrid_actions": valid, "probe_actions": first,
              "purpose": "real saved learning/update/inference verification; no convergence or task success claim"}
    if not all((result["trained_q_and_parameter_nets_changed"], result["independent_reload_greedy_actions_identical"], valid)):
        raise AssertionError(result)
    Path(output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    verify(args.checkpoint, args.output)
