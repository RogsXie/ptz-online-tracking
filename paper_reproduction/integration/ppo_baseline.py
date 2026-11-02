"""Fixed-duration PPO comparison using the same SSD/CFFF/8-state UE wrapper.

This is an additional fair-interface baseline. The historical HSV 10-state
PPO and its weights remain separate in scripts/train_ptz.py and runs/ppo-baseline.
"""
from pathlib import Path
import json
import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from integration.runtime import REPRO, StandaloneArena, new_run
from integration.transport import RGBTransport
from integration.detector import SSDDetector, sha256
from integration.environment import PaperEnvironment
from hascam.cfff import CFFF
from hascam.reid import FastReIDExtractor
from integration.compute import resolve_device, compute_metadata


class FixedDurationGym(gym.Env):
    def __init__(self, environment, scenarios, seeds):
        super().__init__()
        self.environment, self.scenarios, self.seeds = environment, scenarios, seeds
        self.observation_space = gym.spaces.Box(-2, 2, shape=(8,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(7)
        self.episode = 0
        self.completed_episodes = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        index = self.episode
        self.episode += 1
        scenario = self.scenarios[index % len(self.scenarios)]
        state = self.environment.reset(REPRO / "scenarios" / f"{scenario}.json",
                                       self.seeds[index % len(self.seeds)], "PPO_FIXED_SSD_TRAIN")
        return state, {"scenario": scenario}

    def step(self, action):
        state, reward, done, info = self.environment.step(int(action), .2)
        if done:
            self.completed_episodes += 1
        # This 128-action finite task calls the time limit a terminal failure
        # with -40, matching the paper objective rather than a bootstrapped cut.
        return state, reward, done, False, info


def train(args):
    device = resolve_device(getattr(args, "device", "cpu"))
    if args.checkpoint:
        raise ValueError("PPO comparison currently supports fresh training only; checkpoint is for evaluate/demo")
    output = args.output or new_run("train-ppo-fixed-ssd")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    splits = json.loads((REPRO / "scenarios/splits.json").read_text(encoding="utf-8-sig"))
    scenarios = args.scenarios or splits["train_scenarios"]
    if any(s not in splits["train_scenarios"] for s in scenarios):
        raise ValueError("PPO training may only use training scenarios")
    detector = SSDDetector(args.detector, device=device)
    with StandaloneArena(output, headless=args.headless) as process:
        transport = RGBTransport(process.port)
        try:
            paper = PaperEnvironment(transport, detector, CFFF(FastReIDExtractor(device=device)), args.reward_mode)
            env = FixedDurationGym(paper, scenarios, splits["seed_partitions"]["train"])
            model = PPO("MlpPolicy", env, n_steps=128, batch_size=64, n_epochs=4,
                        learning_rate=3e-4, policy_kwargs={"net_arch": [64, 64]},
                        seed=args.seed, verbose=1, device=device)
            model.set_logger(configure(str(output), ["stdout", "csv"]))
            model.learn(total_timesteps=args.episodes * 128)
            model.save(output / "ppo_fixed_ssd_short")
            (output / "configuration.json").write_text(json.dumps({
                "algorithm": "PPO discrete fixed-duration comparison", "duration_s": .2,
                "compute": compute_metadata(device),
                "actual_steps": model.num_timesteps, "state_dim": 8, "seed": args.seed,
                "gamma": model.gamma, "gae_lambda": model.gae_lambda,
                "rollout_steps": model.n_steps, "batch_size": model.batch_size, "epochs_per_rollout": model.n_epochs,
                "completed_episodes": env.completed_episodes,
                "requested_rollout_blocks_128_steps": args.episodes,
                "scenarios": scenarios, "reward_mode": args.reward_mode,
                "detector": detector.provenance, "cfff": paper.cfff.metadata(),
                "note": "Short comparison only, no convergence claim; historical HSV PPO retained separately."}, indent=2), encoding="utf-8")
        finally:
            transport.close()
    print(f"PPO checkpoint: {output / 'ppo_fixed_ssd_short.zip'}", flush=True)
    return output
