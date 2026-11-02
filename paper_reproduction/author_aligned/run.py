"""Author-source training on UE; static phase only, with per-episode reporting."""
import argparse
import csv
import json
import math
from pathlib import Path
import random
import shutil
import time
import warnings
import cv2
import numpy as np
import torch
from author_aligned.source import REPRO, REFERENCE, COMMIT, digest, verify_source, load_core, create_agent, recorded_update
from author_aligned.environment import bind_environment, AtomicTransport
from integration.runtime import StandaloneArena, new_run, ROOT
from integration.pretrained_detector import create_detector
from integration.experiment import make_extractor

NETWORKS = ("actor_net", "actor_target", "param_net", "param_net_target")
OPTIMIZERS = ("actor_optimiser", "param_net_optimiser")
FORMAT = "author_source_ue_v1"


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def rng_state(agent):
    return {"python": random.getstate(), "numpy_global": np.random.get_state(),
            "numpy_agent": agent.np_random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(agent, states):
    random.setstate(states["python"]); np.random.set_state(states["numpy_global"])
    agent.np_random.set_state(states["numpy_agent"]); torch.set_rng_state(states["torch"].cpu())
    if states["cuda"]: torch.cuda.set_rng_state_all([s.cpu() for s in states["cuda"]])


def save(agent, path, metadata, completed, steps):
    payload={"format": FORMAT, "metadata": metadata, "completed_episodes": completed,
        "environment_steps": steps, "update_count": agent.update_count,
        "actions_count": agent.actions_count, "epsilon": agent.epsilon,
        "rng":rng_state(agent), "replay": {"capacity":agent.memory.capacity,
        "position":agent.memory.position,"buffer":agent.memory.buffer}}
    for name in (*NETWORKS,*OPTIMIZERS): payload[name]=getattr(agent,name).state_dict()
    temporary=Path(path).with_suffix(".tmp.pt")
    torch.save(payload,temporary);temporary.replace(path)


def restore(agent, path, metadata):
    saved=torch.load(path,map_location=agent.device,weights_only=False)
    if saved.get("format")!=FORMAT: raise ValueError("Old modified-policy checkpoints cannot enter source-aligned training")
    for name in ("source_sha256","runtime_sources","seed","execution_boundary","seconds_per_parameter","detector","reid","scene_sha256","pdqn"):
        if saved["metadata"].get(name)!=metadata.get(name): raise ValueError(f"Resume contract differs: {name}")
    for name in (*NETWORKS,*OPTIMIZERS): getattr(agent,name).load_state_dict(saved[name])
    for key,value in saved["replay"].items(): setattr(agent.memory,key,value)
    for key in ("update_count","actions_count","epsilon"): setattr(agent,key,saved[key])
    restore_rng(agent,saved["rng"])
    return saved["completed_episodes"],saved["environment_steps"]


def image_metrics(state):
    visible = state[3] > 0
    error = math.hypot(float(state[0]),float(state[1])) if visible else None
    area = float(state[2])/100 if visible else None
    quality=bool(visible and state[3]>=.9 and error<=.15 and .02<=area<=.08)
    return {"visible":bool(visible),"center_error":error,"area_ratio":area,
            "confidence":float(state[3]),"quality_now":quality}


def episode(agent, env, transport, folder, scene, seed, training, episode_number, stop_path):
    folder.mkdir(parents=True,exist_ok=False)
    bridge=env.io_bridge;bridge.scenario=scene;bridge.seed=seed;bridge.target_count=1
    state=np.array(env.reset(),dtype=np.float32)
    if env.detected["error"]: raise env.detected["error"]
    if transport.diagnostics.get("demo_geometry_follow") or transport.diagnostics.get("free_preview"):
        raise RuntimeError("Unexpected non-policy control mode")
    write_json(folder/"resolved-scene.json",transport.diagnostics)
    cv2.imwrite(str(folder/"initial-rgb.jpg"),cv2.cvtColor(transport.rgb,cv2.COLOR_RGB2BGR))
    _,pad,_=load_core()
    updates=[];rewards=[];centers=[];quality_streak=quality_max=0;clipped=0;action_counts=[0]*7
    initial_updates=agent.update_count;started=time.perf_counter()
    with (folder/"steps.jsonl").open("w",encoding="utf-8",buffering=1) as step_log, \
         (folder/"updates.jsonl").open("w",encoding="utf-8",buffering=1) as update_log:
        step_log.write(json.dumps({"initial":True,"state":state.tolist(),**image_metrics(state)})+"\n")
        for index in range(128):
            if stop_path.exists(): raise InterruptedError("STOP_REQUEST received at an action boundary")
            # The author train=False branch is broken. epsilon=0 uses the same
            # source train=True greedy path; evaluation state/RNG restored later.
            act,param,all_params=agent.choose_action(state,train=True)
            action=pad(act,param)
            next_state,reward,done,_=env.step(action)
            if env.detected["error"]: raise env.detected["error"]
            next_state=np.array(next_state,dtype=np.float32)
            metrics=image_metrics(next_state)
            if training:
                agent.memory.push(state,np.concatenate(([act],all_params.data)).ravel(),reward,next_state,done)
                if index % 32 == 0:
                    update=recorded_update(agent)
                    if update:
                        update.update(episode=episode_number,decision_zero_based=index)
                        updates.append(update);update_log.write(json.dumps(update)+"\n")
            quality_streak=quality_streak+1 if metrics["quality_now"] else 0
            quality_max=max(quality_max,quality_streak)
            if metrics["visible"]: centers.append(metrics["center_error"])
            clipped+=int(bridge.last_action["execution_clipped"]);action_counts[int(act)]+=1
            row={"decision_zero_based":index,"input_state":state.tolist(),"next_state":next_state.tolist(),
                 "all_raw_parameters":all_params.tolist(),"reward":float(reward),"done":bool(done),
                 **bridge.last_action,**metrics,"updates_total":agent.update_count,
                 "sim_time_s":transport.diagnostics["sim_time"],"rotation":transport.diagnostics["rotation"],
                 "fov":transport.diagnostics["fov"],"completed_targets":env.object_detect_num,
                 "source_confidence_streak":env.end_iteration,"quality_streak":quality_streak}
            step_log.write(json.dumps(row,allow_nan=False)+"\n")
            rewards.append(float(reward));state=next_state
            if done: break
    cv2.imwrite(str(folder/"final-rgb.jpg"),cv2.cvtColor(transport.rgb,cv2.COLOR_RGB2BGR))
    result={"round":episode_number,"seed":seed,"training":training,"steps":len(rewards),
            "reward_sum":sum(rewards),"q_loss_mean":float(np.mean([r["q_loss"] for r in updates])) if updates else None,
            "parameter_objective_mean":float(np.mean([r["parameter_objective"] for r in updates])) if updates else None,
            "gradient_norm_after_clip_mean":float(np.mean([r["parameter_grad_norm_after_clip"] for r in updates])) if updates else None,
            "epsilon":float(agent.epsilon),"network_updates":agent.update_count-initial_updates,
            "updates_total":agent.update_count,"replay_size":len(agent.memory),
            "q_learning_rate":agent.actor_lr,"parameter_learning_rate":agent.param_net_lr,"batch_size":agent.batch_size,
            "center_mean":float(np.mean(centers)) if centers else None,
            "center_p95":float(np.percentile(centers,95)) if centers else None,
            "visible_fraction":len(centers)/len(rewards),"source_success":env.object_num==0,
            "quality_streak_max":quality_max,"quality_success":quality_max>=10,
            "execution_clip_count":clipped,"action_counts":action_counts,
            "wall_seconds":time.perf_counter()-started}
    write_json(folder/"summary.json",result)
    return result


def formatted(r):
    def num(key,places=5): return "未更新" if r.get(key) is None else f"{r[key]:.{places}f}"
    return (f"第{r['round']}轮 | Q损失={num('q_loss_mean')} | 参数目标(最大化)={num('parameter_objective_mean')} | "
            f"奖励={num('reward_sum',3)} | ε={num('epsilon',4)} | 步数={r['steps']} | "
            f"本轮/累计更新={r['network_updates']}/{r['updates_total']} | 回放={r['replay_size']} | "
            f"中心均值/P95={num('center_mean',4)}/{num('center_p95',4)} | "
            f"源码完成={r['source_success']} | 质量连续10帧={r['quality_success']} | 执行裁剪={r['execution_clip_count']}")


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes",type=int,default=32,help="Current static audit block, not a claim of 4060-episode completion")
    parser.add_argument("--seed",type=int,default=1)
    parser.add_argument("--output",type=Path)
    parser.add_argument("--checkpoint",type=Path)
    # Explicit human authorization in this chat on 2026-10-31: execution-only
    # Box clipping, raw/5 mapping, unchanged raw values in every learning path.
    parser.add_argument("--execution-boundary",choices=("reject","clip_box"),default="clip_box")
    parser.add_argument("--headless",action="store_true")
    parser.add_argument("--validate",action="store_true",help="40 frozen held-out static episodes after this block")
    args=parser.parse_args(argv)
    if not 1<=args.episodes<=4060: parser.error("episodes must be 1..4060")
    if not torch.cuda.is_available(): raise RuntimeError("Existing CUDA environment required")
    warnings.filterwarnings("ignore",message="To copy construct from a tensor")
    warnings.filterwarnings("ignore",category=SyntaxWarning)
    torch.set_num_threads(4)
    sources=verify_source();output=args.output or new_run("author-source-ue-static")
    output=output.resolve();output.mkdir(parents=True,exist_ok=True)
    if (output/"configuration.json").exists(): raise FileExistsError("Use a fresh run folder")
    detector=create_detector(device="cuda");extractor=make_extractor("cuda")
    expected=json.loads((REPRO/"control_curriculum/protocol.json").read_text())["frozen_frontend"]
    assert detector.provenance["sha256"]==expected["detector_sha256"]
    assert detector.threshold==.15 and detector.imgsz==960 and detector.half is False
    assert extractor.metadata()["weights_sha256"]==expected["reid_sha256"]
    detector.warmup()
    scene=REPRO/"scenarios/static_near.json"
    metadata={"profile":FORMAT,"author_commit":COMMIT,"source_sha256":sources,
        "seconds_per_parameter":.2,"unit_mapping_authorized_by_user":True,
        "execution_box_clip_authorized_by_user_2026_10_31":args.execution_boundary=="clip_box",
        "execution_boundary":args.execution_boundary,"scene":str(scene),"scene_sha256":digest(scene),
        "training_seed_base":700000,"validation_seed_base":800000,
        "detector":{k:detector.provenance[k] for k in ("sha256","threshold","imgsz","half","target_class_ids")},
        "reid":{"sha256":extractor.metadata()["weights_sha256"],"feature_dim":2048,"input_size":[256,256]},
        "pdqn":{"actor_lr":.001,"param_lr":.0001,"gamma":.9,"batch_size":512,"replay_capacity":100000,
                 "update_decisions_zero_based":[0,32,64,96],"target_update_interval":100,"max_steps":128},
        "original_env_score_filter":.3,"original_success_confidence_strict":.9,
        "seed":args.seed,"requested_episodes":args.episodes,"reference_budget":4060,
        "fresh_initialization":not bool(args.checkpoint),"parent":str(args.checkpoint) if args.checkpoint else None,
        "no_auto_stage_advance":True,"no_videos":True,"no_gt_state_reward_or_action":True,
        "source_goal":"sequential confidence-based enhancement; not continuous centering",
        "adaptations":["UE instead of unavailable Unity","explicit user raw/5 physical time mapping",
            "frozen user-approved detector and verified ReID","Gymnasium space containers",
            "headless in-memory sensor I/O","1-D reshape for current SciPy cosine",
            "seed torch/global numpy for reproducibility","fresh run bypasses missing author epoch_4060 recovery",
            "frozen evaluation via epsilon=0 source training branch","original source bugs retained"],
        "runtime_sources":{str(p.relative_to(REPRO)):digest(p) for p in [
            *sorted((REPRO/"author_aligned").glob("*.py")),
            *[REPRO/name for name in ("integration/transport.py","integration/focused_environment.py",
                "integration/runtime.py","integration/pretrained_detector.py","integration/experiment.py",
                "hascam/reid.py","hascam/detection.py")]]},
        "ue_module_sha256":digest(ROOT/"PTZSim/Binaries/Win64/UnrealEditor-PTZSim.dll")}
    write_json(output/"configuration.json",metadata)
    for name in sources:
        dest=output/"source-snapshot/reference"/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(REFERENCE/name,dest)
    for name in metadata["runtime_sources"]:
        dest=output/"source-snapshot"/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(REPRO/name,dest)
    agent=create_agent(args.seed,"cuda");completed=steps=0
    if args.checkpoint: completed,steps=restore(agent,args.checkpoint,metadata)
    if completed+args.episodes>4060: raise ValueError("Requested run exceeds original 4060-episode budget")
    _,_,epsilon=load_core()
    frozen={str(p):digest(p) for p in (REPRO/"weights/yolo26s_ue_gpu30.pt",extractor.weights,REPRO/"weights/hascam_focused_gpu.pt")}
    write_json(output/"frozen-before.json",frozen)
    save(agent,output/"initial.pt",metadata,completed,steps)
    history=[];status="running";validation=None
    print(f"SOURCE {COMMIT}; Qlr=.001 Paramlr=.0001 batch=512 gamma=.9 replay=100000; execution={args.execution_boundary}",flush=True)
    try:
        with StandaloneArena(output,headless=args.headless) as arena:
            transport=AtomicTransport(arena.port,hud=not args.headless)
            try:
                env=bind_environment(transport,detector,extractor,args.execution_boundary)
                with (output/"episodes.jsonl").open("w",encoding="utf-8",buffering=1) as log, \
                     (output/"round-metrics.log").open("w",encoding="utf-8",buffering=1) as human, \
                     (output/"round-metrics.csv").open("w",encoding="utf-8-sig",newline="",buffering=1) as csvfile:
                    writer=None
                    for _ in range(args.episodes):
                        number=completed+1;agent.epsilon=epsilon(number)
                        case_seed=700000+completed
                        result=episode(agent,env,transport,output/"training"/f"episode-{number:04d}-seed{case_seed}",
                                       scene,case_seed,True,number,output/"STOP_REQUEST")
                        completed=number;steps+=result["steps"];history.append(result)
                        log.write(json.dumps(result)+"\n");human.write(formatted(result)+"\n");print(formatted(result),flush=True)
                        if writer is None: writer=csv.DictWriter(csvfile,fieldnames=list(result));writer.writeheader()
                        writer.writerow(result)
                        write_json(output/"progress.json",{"completed_episodes":completed,"this_run_episodes":len(history),
                            "environment_steps":steps,"updates":agent.update_count,"status":"training","last_round":result})
                        if completed%10==0: save(agent,output/f"checkpoint-{completed:04d}.pt",metadata,completed,steps)
                save(agent,output/"last.pt",metadata,completed,steps)
                if args.validate:
                    states=rng_state(agent);saved_epsilon=agent.epsilon;count=agent.actions_count
                    parameters={n:{k:v.detach().clone() for k,v in getattr(agent,n).state_dict().items()} for n in NETWORKS}
                    before_updates=agent.update_count;agent.epsilon=0;results=[]
                    try:
                        for index in range(40):
                            seed=800000+index
                            r=episode(agent,env,transport,output/"validation"/f"seed{seed}",scene,seed,False,index+1,output/"STOP_REQUEST")
                            results.append(r)
                            print(f"VALIDATION {index+1}/40 source_success={r['source_success']} quality_success={r['quality_success']}",flush=True)
                    finally:
                        agent.epsilon=saved_epsilon;agent.actions_count=count;restore_rng(agent,states)
                    identical=all(torch.equal(v,getattr(agent,n).state_dict()[k]) for n,p in parameters.items() for k,v in p.items())
                    if not identical or agent.update_count!=before_updates: raise RuntimeError("Evaluation changed training weights")
                    validation={"episodes":results,"source_success_count":sum(r["source_success"] for r in results),
                        "quality_success_count":sum(r["quality_success"] for r in results),
                        "no_weight_updates":identical,"continuous_tracking_gate_passed":False,"no_auto_stage_advance":True}
                    write_json(output/"validation.json",validation)
                status="static_block_complete_unqualified"
            finally: transport.close()
    except BaseException as error:
        status="stopped_on_error"
        save(agent,output/"interrupted.pt",metadata,completed,steps)
        write_json(output/"failure.json",{"error":repr(error),"completed_episodes":completed,
            "note":"Partial episode may have durable transitions; completed-episode count excludes it"})
        raise
    finally:
        after={p:digest(p) for p in frozen};write_json(output/"frozen-after.json",after)
        write_json(output/"status.json",{"status":status,"completed_episodes":completed,"this_run_episodes":len(history),
            "completed_episode_steps":steps,"updates":agent.update_count,"frozen_unchanged":after==frozen,
            "later_stages_started":False,"validation_performed":validation is not None})
        if after!=frozen: raise RuntimeError("Frozen artifact changed")
    print(f"Outputs: {output}",flush=True)


if __name__=="__main__": main()
