"""Actual GPU fine-tuning with empty new replay and unchanged source updates."""
import speed_bootstrap
from speed_bootstrap import REV,configure_precision_project
import argparse,json,time,warnings,hashlib
from pathlib import Path
import torch
import train as legacy
from author_aligned.source import load_core
from integration.runtime import StandaloneArena
from author_aligned.environment import AtomicTransport
from speed_agent_v3 import create_agent,PROFILE
from speed_runner_v3 import frontend,initialize,campaign,AuditedSpeed
from speed_scenes import layout
from calibrate import write

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--episodes',type=int,default=160);args=p.parse_args()
    legacy.exclusive();project=configure_precision_project();warnings.filterwarnings('ignore');torch.set_num_threads(4)
    out=REV/'training-v3';out.mkdir(exist_ok=False);detector,extractor=frontend();agent=create_agent()
    parent=initialize(agent,out);meta=dict(experiment='stationary_author_core_pdqn_speed_revision',speed_profile=PROFILE['profile'],
        profile=PROFILE,parent_transfer=parent,detector=detector.provenance,reid=extractor.metadata(),project=str(project),
        source_files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in REV.glob('*.py')})
    write(out/'configuration.json',meta);legacy.checkpoint(agent,out/'initial.pt',0,0,meta)
    completed=steps=0;_,_,epsilon=load_core();start=time.perf_counter()
    try:
        with StandaloneArena(out,headless=True) as arena:
            t=AtomicTransport(arena.port,hud=False)
            try:
                env=AuditedSpeed(t,detector,extractor)
                with (out/'episodes.jsonl').open('w',encoding='utf-8',buffering=1) as log:
                    for number in range(1,args.episodes+1):
                        agent.epsilon=epsilon(480+number)
                        result=legacy.episode(agent,env,layout('train',40000+number-1),out/f'training/episode-{number:04d}',True,4,out,video=False)
                        completed=number;steps+=result['steps'];log.write(json.dumps(result)+'\n')
                        write(out/'progress.json',dict(completed=completed,steps=steps,updates=agent.update_count,last=result,wall_s=time.perf_counter()-start))
                        if number%10==0:print('NEW REAL TRAIN',number,'steps',steps,'updates',agent.update_count,'lastquality',result['success'],flush=True)
                        if number in (20,40,80,160):
                            path=out/f'checkpoint-{number:04d}.pt';legacy.checkpoint(agent,path,completed,steps,meta)
                            campaign(agent,env,out/f'development-{number:04d}',path,count=80)
                        if number%10==0:legacy.checkpoint(agent,out/'last.pt',completed,steps,meta)
                legacy.checkpoint(agent,out/'last.pt',completed,steps,meta)
            finally:t.close()
    except BaseException as exc:
        legacy.checkpoint(agent,out/'interrupted.pt',completed,steps,meta)
        write(out/'failure.json',dict(error=repr(exc),completed=completed,steps=steps));raise
    finally:write(out/'status.json',dict(completed=completed,steps=steps,updates=agent.update_count,qualified=False))

if __name__=='__main__':main()
