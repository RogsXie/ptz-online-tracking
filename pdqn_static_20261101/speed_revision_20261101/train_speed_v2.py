"""Real GPU critic adaptation and source P-DQN fine-tuning, declared v2."""
import speed_bootstrap
from speed_bootstrap import REV,configure_precision_project
import json,time,warnings,hashlib
import torch
import train as legacy
from author_aligned.source import recorded_update,load_core
from author_aligned.environment import AtomicTransport
from integration.runtime import StandaloneArena
from speed_runner_v2 import create_agent,frontend,initialize,campaign,AuditedSpeed,PROFILE
from speed_scenes import layout
from calibrate import write

def main():
    legacy.exclusive();configure_precision_project();warnings.filterwarnings('ignore');torch.set_num_threads(4)
    out=REV/'training-v2';out.mkdir(exist_ok=False);detector,extractor=frontend();a=create_agent();parent=initialize(a,out)
    prior=REV/'training-v1/checkpoint-0040.pt';saved=torch.load(prior,map_location='cpu',weights_only=False)
    assert saved['environment_steps']==3401 and saved['metadata']['speed_profile']=='paper25_precision_duration_transfer_v1'
    for k,v in saved['replay'].items():setattr(a.memory,k,v)
    meta=dict(experiment='stationary_author_core_pdqn_speed_revision',speed_profile=PROFILE['profile'],profile=PROFILE,parent_transfer=parent,
        real_new_dynamics_replay_parent=str(prior),reused_real_transitions=len(a.memory),detector=detector.provenance,reid=extractor.metadata(),
        source_files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in REV.glob('*.py')})
    write(out/'configuration.json',meta);legacy.checkpoint(a,out/'initial.pt',0,3401,meta)
    frozen={k:v.clone() for k,v in a.param_net.state_dict().items()}
    for p in a.param_net_optimiser.param_groups:p['lr']=0.
    with (out/'critic-warmup.jsonl').open('w',encoding='utf-8',buffering=1) as f:
        for n in range(512):
            row=recorded_update(a);assert row and all(__import__('math').isfinite(row[k]) for k in ('q_loss','parameter_objective'))
            f.write(json.dumps(row)+'\n')
    assert all(torch.equal(v,a.param_net.state_dict()[k]) for k,v in frozen.items())
    a.param_net_optimiser=torch.optim.Adam(a.param_net.parameters(),lr=.00001)
    write(out/'warmup-proof.json',dict(original_updates=512,replay_real_transitions=3401,online_parameter_weights_unchanged=True,
        q_lr=.0001,parameter_lr_during_warmup=0.,parameter_lr_after=.00001,scope='Value adaptation only, not additional environment episodes'))
    legacy.checkpoint(a,out/'after-critic-warmup.pt',0,3401,meta)
    completed=0;steps=3401;_,_,epsilon=load_core();started=time.perf_counter()
    try:
        with StandaloneArena(out,headless=True) as arena:
            t=AtomicTransport(arena.port,hud=False)
            try:
                env=AuditedSpeed(t,detector,extractor)
                with (out/'episodes.jsonl').open('w',encoding='utf-8',buffering=1) as f:
                    for n in range(1,81):
                        a.epsilon=epsilon(480+n);r=legacy.episode(a,env,layout('train',20000+n-1),out/f'training/episode-{n:04d}',True,4,out,video=False)
                        completed=n;steps+=r['steps'];f.write(json.dumps(r)+'\n')
                        delta=sum(float((v-a.param_net.state_dict()[k]).abs().sum()) for k,v in frozen.items())
                        write(out/'progress.json',dict(completed=n,steps=steps,updates=a.update_count,postwarm_updates=a.update_count-512,parameter_weight_abs_change=delta,last=r,wall_s=time.perf_counter()-started))
                        if n%10==0:print('V2 REAL TRAIN',n,'steps',steps,'updates',a.update_count,'paramdelta',delta,flush=True)
                        if n in (20,40,80):
                            path=out/f'checkpoint-{n:04d}.pt';legacy.checkpoint(a,path,completed,steps,meta)
                            campaign(a,env,out/f'development-{n:04d}',path,count=80)
                        if n%10==0:legacy.checkpoint(a,out/'last.pt',completed,steps,meta)
                legacy.checkpoint(a,out/'last.pt',completed,steps,meta)
            finally:t.close()
    except BaseException as exc:
        legacy.checkpoint(a,out/'interrupted.pt',completed,steps,meta);write(out/'failure.json',dict(error=repr(exc)));raise
    finally:write(out/'status.json',dict(completed=completed,steps=steps,updates=a.update_count,qualified=False))

if __name__=='__main__':main()
