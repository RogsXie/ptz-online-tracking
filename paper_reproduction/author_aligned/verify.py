"""Numerically check source reuse and read-only loss instrumentation."""
from copy import deepcopy
import argparse
import json
from pathlib import Path
import random
from types import SimpleNamespace
import warnings
import gymnasium as gym
import numpy as np
import torch
from author_aligned.source import load_core, recorded_update, verify_source, REPRO, create_agent
from author_aligned.environment import bind_environment
from hascam.detection import Detection


def make_agent(seed=1, device="cpu"):
    return create_agent(seed, device)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--device",choices=("cpu","cuda"),default="cpu")
    args=parser.parse_args()
    warnings.filterwarnings("ignore", category=UserWarning)
    torch.set_num_threads(4)
    checks = {}
    sources = verify_source()
    agent = make_agent(device=args.device)
    with torch.no_grad():
        initial = agent.param_net(torch.tensor([.1, -.1, .5, .88, 0, 0, 0, 0],device=args.device)).tolist()
    _, pad, epsilon = load_core()
    checks["source_replay_capacity_100000"] = agent.memory.capacity == 100000
    checks["epsilon_first_and_last"] = abs(epsilon(1)-.8973333333333333)<1e-12 and abs(epsilon(300)-.1)<1e-12 and epsilon(301)==.1
    checks["source_default_initialization_not_midpoint"] = any(x < .3 for x in initial)
    rng = np.random.RandomState(41)
    for i in range(640):
        state = rng.normal(size=8).astype(np.float32)
        nxt = rng.normal(size=8).astype(np.float32)
        action = np.concatenate(([i % 7], rng.uniform(.3, 5, 7)))
        agent.memory.push(state, action, float(rng.normal()), nxt, i%11 == 0)
    reference = deepcopy(agent)
    updates = []
    same_rng = True
    for i in range(100):
        rng_state = random.getstate()
        reference.update()
        reference_rng = random.getstate()
        random.setstate(rng_state)
        updates.append(recorded_update(agent))
        same_rng = same_rng and random.getstate() == reference_rng
    for name in ("actor_net", "actor_target", "param_net", "param_net_target"):
        checks[f"100_updates_{name}_bitwise_equal"] = all(torch.equal(v, getattr(reference, name).state_dict()[k])
            for k,v in getattr(agent,name).state_dict().items())
    checks["100_actual_updates_and_sync"] = agent.update_count==reference.update_count==100 and updates[-1]["target_synchronised"]
    checks["actual_loss_and_objective_captured"] = all(np.isfinite(r["q_loss"]) and np.isfinite(r["parameter_objective"]) for r in updates)
    checks["read_only_logger_preserves_sample_rng"] = same_rng

    class Transport:
        rgb = np.zeros((540,960,3),np.uint8)
        diagnostics = {"sim_time": 0.}
        def scenario(self, path): pass
        def reset(self, seed): self.diagnostics={"sim_time":0.}
        def capture(self): return self.rgb
        def rpc(self, text):
            self.diagnostics["sim_time"] += float(text.split()[-1])
            return self.diagnostics
    class Features:
        device = "cpu"
        def model(self, image): return torch.ones((1,2048))
    detection = Detection((.4,.4,.6,.6), .95, 0, "car")
    env = bind_environment(Transport(),lambda rgb:[detection],Features())
    env.io_bridge.scenario="fixture";env.io_bridge.seed=123;env.io_bridge.target_count=1
    state = np.array(env.reset())
    checks["source_area_times_100"] = abs(state[2]-4)<1e-5
    checks["source_reset_history_zero"] = np.array_equal(state[4:],np.zeros(4))
    for index in range(10):
        state,reward,done,_ = env.step(pad(2,np.array([.5])))
        if index==0:
            checks["original_delayed_history_retained"] = state[4:6]==[0,0]
        if index==1:
            checks["original_duration_index_behavior_retained"] = state[4:6]==[2/6,2/5]
    checks["source_confidence_completion_10_steps"] = done and env.object_detect_num==1 and env.iteration_counter==10
    checks["source_completion_bonus_44"] = abs(float(reward)-43.91)<1e-5
    env = bind_environment(Transport(),lambda rgb:[],Features())
    env.io_bridge.scenario="fixture";env.io_bridge.seed=123;env.io_bridge.target_count=1
    checks["source_lost_sentinel"] = env.reset()==[-2,-2,-1,0,0,0,0,0]
    for index in range(128): state,reward,done,_=env.step(pad(0,np.array([.5])))
    checks["source_128_timeout_minus44"] = done and reward == -44
    try:
        env.step(pad(0,np.array([-.1])))
        checks["reject_undefined_out_of_box_execution"] = False
    except ValueError:
        checks["reject_undefined_out_of_box_execution"] = True
    env.io_bridge.execution_boundary="clip_box"
    env.step(pad(0,np.array([-.1])))
    checks["explicit_execution_clamp_min_preserves_raw"] = env.io_bridge.last_action["raw_parameter"]==-.1 and abs(env.io_bridge.last_action["duration_s"]-.06)<1e-12
    env.step(pad(0,np.array([6.])))
    checks["explicit_execution_clamp_max"] = env.io_bridge.last_action["duration_s"]==1.
    checks = {key:bool(value) for key,value in checks.items()}
    result={"passed":all(checks.values()),"device":args.device,"checks":checks,"source_sha256":sources,
            "initial_raw_parameter_example":initial,"updates":100,
            "last_measured_update":updates[-1],"note":"Synthetic source-correspondence checks, not training/control success"}
    output=REPRO/"verification/author-aligned-20261031"
    output.mkdir(parents=True,exist_ok=True)
    (output/f"source-checks-{args.device}.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result,indent=2))
    if not result["passed"]: raise SystemExit(1)


if __name__=="__main__": main()
