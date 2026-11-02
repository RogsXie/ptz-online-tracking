"""Controlled update-budget contrast using original learner and real UE replay."""
import bootstrap
from bootstrap import HERE
import argparse,json,math,time,warnings
from pathlib import Path
import torch
from author_aligned.source import create_agent,recorded_update
from train import exclusive,resume,checkpoint
from calibrate import write

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path);p.add_argument('--updates',type=int,default=2048)
    args=p.parse_args();exclusive();warnings.filterwarnings('ignore');torch.set_num_threads(4)
    root=args.output.resolve();root.mkdir(exist_ok=False);agent=create_agent(20261101,'cuda')
    completed,steps=resume(agent,args.checkpoint);initial_updates=agent.update_count
    previous=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    meta=previous['metadata']|dict(parent=str(args.checkpoint),
        controlled_contrast=dict(difference='2048 additional uniform replay updates; exact source update math',
            new_environment_interactions=0,source_replay_size=len(agent.memory),target_updates=args.updates,
            note='Offline optimization on existing real UE data, not extra episodes or simulator rollouts'))
    write(root/'configuration.json',meta)
    checkpoint(agent,root/'initial.pt',completed,steps,meta)
    started=time.perf_counter()
    with (root/'updates.jsonl').open('w',encoding='utf-8',buffering=1) as log:
        for index in range(args.updates):
            if (root/'STOP_REQUEST').exists():break
            record=recorded_update(agent)
            if record is None or not all(math.isfinite(record[k]) for k in ('q_loss','parameter_objective','parameter_grad_norm_after_clip')):
                raise RuntimeError('Nonfinite or missing source update')
            log.write(json.dumps(record)+'\n')
            if (index+1)%128==0:
                print('SOURCE REPLAY UPDATE',index+1,'total',agent.update_count,'q_loss',round(record['q_loss'],5),'objective',round(record['parameter_objective'],5),flush=True)
                write(root/'progress.json',dict(additional_updates=index+1,total_updates=agent.update_count,wall_s=time.perf_counter()-started))
            if index+1 in (512,1024,2048):checkpoint(agent,root/f'updates-{index+1:04d}.pt',completed,steps,meta)
    checkpoint(agent,root/'last.pt',completed,steps,meta)
    write(root/'summary.json',dict(completed_episodes=completed,environment_steps=steps,new_environment_steps=0,
        initial_updates=initial_updates,additional_updates=agent.update_count-initial_updates,
        wall_seconds=time.perf_counter()-started,qualified=False))

if __name__=='__main__':main()
