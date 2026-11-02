"""Train a seven-action PPO baseline on live UE RGB-derived boxes.

Short runs verify the training pipeline only; they do not establish tracking
performance. This is not the paper's hybrid-action HAS-CAM implementation.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor

from ptz_env import PTZTrackingEnv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--port", type=int, default=9001)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "runs" / "ppo-baseline")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    env = PTZTrackingEnv(port=args.port)
    try:
        check_env(env, warn=True)
        print("Gymnasium/SB3 environment check passed", flush=True)
        monitored = Monitor(env, str(args.output / "monitor.csv"))
        model = PPO("MlpPolicy", monitored, n_steps=64, batch_size=32, n_epochs=4,
                    learning_rate=3e-4, seed=args.seed, verbose=1, device="cpu",
                    policy_kwargs=dict(net_arch=[64, 64]))
        model.set_logger(configure(str(args.output), ["stdout", "csv"]))
        model.learn(total_timesteps=args.steps)
        model.save(str(args.output / "ppo_ptz"))
        loaded = PPO.load(str(args.output / "ppo_ptz"), device="cpu")
        # A separate held-out episode checks checkpoint loading and inference.
        obs, info = env.reset(seed=10000 + args.seed)
        records = []
        for _ in range(64):
            action, _ = loaded.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            records.append(dict(reward=reward, visible=info["visible"], center_error=info["center_error"]))
            if terminated or truncated:
                break
        result = dict(algorithm="PPO", purpose="pipeline smoke test" if args.steps < 10000 else "baseline training",
                      requested_steps=args.steps, actual_steps=model.num_timesteps, training_seed=args.seed,
                      evaluation_seed=10000 + args.seed, evaluation_steps=len(records),
                      evaluation_return=float(sum(x["reward"] for x in records)),
                      evaluation_visible_fraction=float(np.mean([x["visible"] for x in records])),
                      evaluation_center_error=float(np.mean([x["center_error"] for x in records])),
                      observation="RGB HSV bbox + PTZ state + previous bbox offset; 10 floats",
                      action="7 discrete actions, fixed 0.2 s duration",
                      simulator_ground_truth_used=False,
                      checkpoint_reload_verified=True,
                      limitations="Single simple scene, no learned detector, no HAS-CAM hybrid action, no performance claim")
        (args.output / "training-results.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
