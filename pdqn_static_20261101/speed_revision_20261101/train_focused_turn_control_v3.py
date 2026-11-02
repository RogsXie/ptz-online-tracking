"""Continue this session's fresh PTZ lineage on new turn layouts only.

Perception and author's four-network P-DQN update are fixed. New RGB execution
data use the declared v3 camera-only zoom/edge budgets. No GT teacher or old
published PTZ weights/replay are imported.
"""
import speed_bootstrap
from speed_bootstrap import REV,configure_precision_project
import argparse,copy,hashlib,json,shutil,time,warnings
from collections import Counter
from pathlib import Path
import cv2,torch
from threadpoolctl import threadpool_limits
from author_aligned.source import verify_source
from author_aligned.run import NETWORKS,OPTIMIZERS,restore_rng
from train import exclusive,checkpoint
from train_precision_from_scratch import rollout
from precision_training_scenes import FAMILIES
from control_latency_runtime import SharedTransport
from stream_arena import StreamArena
from target_roi_detector import Detector
import live_crossroad_robust_tracker as scenario
import unified_core as core
import precision_policy_v2 as policy
from camera_control_v3 import PROFILE
from calibrate import write


def main():
    p=argparse.ArgumentParser();p.add_argument('--parent',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--segments',type=int,default=8);p.add_argument('--steps',type=int,default=96);args=p.parse_args()
    exclusive();configure_precision_project();warnings.filterwarnings('ignore');torch.set_num_threads(4);cv2.setNumThreads(1)
    root=args.output.resolve();root.mkdir(exist_ok=False);sources=root/'frozen-sources';sources.mkdir()
    for n in ['train_focused_turn_control_v3.py','train_precision_from_scratch.py','precision_policy_v2.py','camera_control_v3.py','stable_locked_control.py','precision_training_scenes.py']:
        shutil.copy2(REV/n,sources/n)
    saved=torch.load(args.parent,map_location='cuda',weights_only=False)
    assert saved['metadata']['from_scratch'] and saved['metadata']['parent_checkpoint'] is None
    assert saved['metadata']['motion_profile']==policy.PROFILE
    a=policy.create_agent()
    for n in (*NETWORKS,*OPTIMIZERS):getattr(a,n).load_state_dict(saved[n])
    for k,v in saved['replay'].items():setattr(a.memory,k,v)
    for n in ['update_count','actions_count','epsilon']:setattr(a,n,saved[n])
    restore_rng(a,saved['rng']);a.precision_control_v3=True
    for g in a.actor_optimiser.param_groups:g['lr']=1e-4
    for g in a.param_net_optimiser.param_groups:g['lr']=1e-5
    core.LOW,core.HIGH=policy.LOW,policy.HIGH
    original={n:{k:v.clone() for k,v in getattr(a,n).state_dict().items()} for n in NETWORKS}
    start_steps=saved['environment_steps'];start_updates=a.update_count;counts=Counter({family:start_steps//8 for family in FAMILIES})
    detector,extractor=scenario.previous.previous.previous.previous.frontend();detector=Detector(detector.base)
    metadata=copy.deepcopy(saved['metadata']);metadata.update(camera_control_profile=PROFILE,
        focused_parent=str(args.parent.resolve()),focused_parent_sha256=hashlib.sha256(args.parent.read_bytes()).hexdigest(),
        focused_training='8 new disjoint-seed turn layouts x96 real UE transitions; original source P-DQN update every2steps; parent replay retained.',
        control_changes='RGB motion/own timing lookahead limits acquisition zoom; normalized predicted edge risk prioritizes the escaping image axis. No perception/tracker changes.',
        focused_source_sha256=verify_source(),focused_detector=detector.provenance,focused_reid=extractor.metadata(),
        auxiliary_duration_calibration_used=False)
    write(root/'configuration.json',metadata);checkpoint(a,root/'before.pt',saved['completed_episodes'],start_steps,metadata)
    start=time.perf_counter();completed=0;status='running'
    try:
        with StreamArena(root,headless=True) as arena:
            t=SharedTransport(arena.port)
            try:
                env=policy.observation_class(scenario.Observation)(t,detector,extractor)
                for i in range(args.segments):
                    # All are unseen train seeds; held-out development/check
                    # seed ranges and the published viaduct are never trained.
                    index=1004+8*i
                    result=rollout(a,env,root/f'train/turn-{i:03d}',index,'train',args.steps,.15 if i<args.segments//2 else .05,True,counts,root)
                    completed=i+1
                    progress=dict(completed_segments=completed,new_real_steps=sum(counts.values())-start_steps,
                        total_real_steps=sum(counts.values()),new_source_updates=a.update_count-start_updates,total_source_updates=a.update_count,
                        last=result,wall_s=time.perf_counter()-start)
                    write(root/'progress.json',progress);print('TURN CONTROL TRAIN',completed,progress['new_real_steps'],progress['new_source_updates'],result['visible'],result['centered'],flush=True)
                    if completed%4==0:checkpoint(a,root/f'checkpoint-{completed:03d}.pt',saved['completed_episodes']+completed,sum(counts.values()),metadata)
                checkpoint(a,root/'last.pt',saved['completed_episodes']+completed,sum(counts.values()),metadata);status='complete'
            finally:t.close()
    finally:
        proof={n:dict(changed_tensors=sum(not torch.equal(v,getattr(a,n).state_dict()[k]) for k,v in original[n].items()),
            all_finite=all(torch.isfinite(v).all().item() for v in getattr(a,n).state_dict().values())) for n in NETWORKS}
        write(root/'status.json',dict(status=status,completed_segments=completed,new_real_steps=sum(counts.values())-start_steps,
            total_real_steps=sum(counts.values()),new_source_updates=a.update_count-start_updates,total_source_updates=a.update_count,
            four_network_proof=proof,wall_s=time.perf_counter()-start,perception_fixed=True,qualified=False))


if __name__=='__main__':
    with threadpool_limits(limits=1,user_api='blas'):main()
