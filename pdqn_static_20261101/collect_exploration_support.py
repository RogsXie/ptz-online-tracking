"""Real UE data collection through the unmodified author's exploration branch."""
import bootstrap
from bootstrap import HERE,REPRO
import argparse,json,math,random,time,copy,warnings
from pathlib import Path
import cv2,numpy as np,torch
from scenes import layout as base_layout,ground_position,INIT,materialize
from environment import Environment
from author_aligned.environment import AtomicTransport
from author_aligned.source import create_agent
from author_aligned.run import NETWORKS,OPTIMIZERS
from integration.runtime import StandaloneArena
from integration.experiment import make_extractor
from control_model_v13.detector import create_detector
from train import exclusive,resume,checkpoint
from calibrate import write

DECLARATION=json.loads((HERE/'exploration_support_declaration.json').read_text(encoding='utf-8'))
SIGNS=[(-1,-1),(1,-1),(-1,1),(1,1)]

def collection_layout(index):
    item=base_layout('train',120000+index);rng=random.Random(item['seed'])
    q=index%4;mode=index//4%4;sx,sy=SIGNS[q]
    for attempt in range(1000):
        fov=rng.uniform(56.,63.)
        if mode<2:ox,oy=rng.uniform(.34,.39),rng.uniform(.34,.39)
        elif mode==2:ox,oy=rng.uniform(.34,.39),rng.uniform(.04,.28)
        else:ox,oy=rng.uniform(.04,.28),rng.uniform(.34,.39)
        center=[.5+sx*ox,.5+sy*oy]
        pos=ground_position(*center,fov,INIT)
        dx,dy=pos[0]-INIT['camera_location_cm'][0],pos[1]-INIT['camera_location_cm'][1]
        yaw=math.degrees(math.atan2(dy,dx));pitch=math.degrees(math.atan2(90.-4500.,math.hypot(dx,dy)))
        if -78.<=pitch<=-40.5 and abs(yaw)<=64.:break
    else:raise RuntimeError('Offline feasible layout sampling failed')
    item.update(source_case_index=120000+index,index=index,quadrant=q,layer=4,scale_group=0,
        planned_center=center,fov_deg=fov,sampling_profile=DECLARATION['profile'],collection_mode=mode,
        offline_reference_pitch=pitch,offline_reference_yaw=yaw,offline_rejection_attempts=attempt,
        scope='Unique real initial condition and8-step training fragment, not a complete control evaluation episode')
    return item

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--fragments',type=int,default=640)
    args=p.parse_args();exclusive();warnings.filterwarnings('ignore');torch.set_num_threads(4)
    if not 1<=args.fragments<=640:raise ValueError('This preregistered block permits1..640fragments')
    out=args.output.resolve();out.mkdir(exist_ok=False)
    detector=create_detector(checkpoint=REPRO/'weights/yolo26s_ue_overhead_v1.pt',device='cuda',threshold=.15,imgsz=960)
    assert detector.base.provenance['sha256']=='1dd0de55c1791dce289cc7dc1d8eb6c50e488b26d38941f4c20845a6cadb5ae2'
    detector.warmup();extractor=make_extractor('cuda')
    assert extractor.sha256=='57fb9c17d88911ea64390bf5427f43511435e7f88f6eed9dbc969d4b611e53cd'
    agent=create_agent(20261101,'cuda');completed,base_steps=resume(agent,args.checkpoint)
    saved=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    meta=saved['metadata']|dict(parent=str(args.checkpoint),exploration_support=DECLARATION,
        real_collection_only=True,detector=detector.provenance,reid=extractor.metadata())
    write(out/'configuration.json',meta)
    frozen={n:{k:v.clone() for k,v in getattr(agent,n).state_dict().items()} for n in NETWORKS}
    optimizers={n:copy.deepcopy(getattr(agent,n).state_dict()) for n in OPTIMIZERS}
    before_updates=agent.update_count;saved_epsilon=agent.epsilon;agent.epsilon=1.
    actual_steps=finished=0;action_counts=[0]*7;started=time.perf_counter();status='collecting'
    try:
        with StandaloneArena(out,headless=True) as arena:
            t=AtomicTransport(arena.port,hud=False)
            try:
                env=Environment(t,detector,extractor)
                with (out/'fragments.jsonl').open('w',encoding='utf-8',buffering=1) as summaries:
                    for index in range(args.fragments):
                        folder=out/f'fragments/fragment-{index:04d}';folder.mkdir(parents=True)
                        item=collection_layout(index);scene=materialize(folder,item)
                        state,initial=env.reset(scene,item['seed'])
                        write(folder/'initial.json',initial|item)
                        cv2.imwrite(str(folder/'initial.jpg'),cv2.cvtColor(t.rgb,cv2.COLOR_RGB2BGR))
                        rows=[]
                        with (folder/'steps.jsonl').open('w',encoding='utf-8',buffering=1) as log:
                            log.write(json.dumps(dict(initial=True,state=state.tolist(),**initial))+'\n')
                            for step in range(8):
                                if (out/'STOP_REQUEST').exists():raise InterruptedError('Stop at actual action boundary')
                                act,param,parameters=agent.choose_action(state,train=True)
                                next_state,reward,done,row=env.step(int(act),float(param[0]))
                                row.update(input_state=state.tolist(),next_state=next_state.tolist(),all_parameters=parameters.tolist(),behavior_epsilon=1.)
                                log.write(json.dumps(row,allow_nan=False)+'\n')
                                agent.memory.push(state,np.concatenate(([act],parameters)).ravel(),float(reward),next_state,bool(done))
                                actual_steps+=1;action_counts[int(act)]+=1;rows.append(row);state=next_state
                                if done:break
                        cv2.imwrite(str(folder/'final.jpg'),cv2.cvtColor(t.rgb,cv2.COLOR_RGB2BGR))
                        finished+=1
                        summary=dict(fragment=index,seed=item['seed'],quadrant=item['quadrant'],mode=item['collection_mode'],
                            actual_steps=len(rows),sim_time=rows[-1]['sim_time'],initial_visible=initial['visible'],
                            actual_task_terminated=bool(rows[-1]['done']),completed_control_episode_claimed=False)
                        summaries.write(json.dumps(summary)+'\n')
                        write(out/'progress.json',dict(finished_fragments=finished,actual_steps=actual_steps,
                            replay_size=len(agent.memory),updates=agent.update_count,action_counts=action_counts,wall_s=time.perf_counter()-started))
                        if finished%20==0:print('REAL EXPLORATION',finished,'fragments',actual_steps,'actual steps','actions',action_counts,flush=True)
                        if finished%80==0:
                            agent.epsilon=saved_epsilon;checkpoint(agent,out/f'data-{finished:04d}.pt',completed,base_steps+actual_steps,meta);agent.epsilon=1.
            finally:t.close()
        status='real_data_collection_complete'
    except BaseException:
        status='interrupted';raise
    finally:
        agent.epsilon=saved_epsilon
        assert agent.update_count==before_updates
        assert all(torch.equal(v,getattr(agent,n).state_dict()[k]) for n,d in frozen.items() for k,v in d.items())
        for n,old in optimizers.items():
            new=getattr(agent,n).state_dict();assert old['param_groups']==new['param_groups']
            for key,values in old['state'].items():
                for name,value in values.items():
                    assert torch.equal(value,new['state'][key][name]) if torch.is_tensor(value) else value==new['state'][key][name]
        checkpoint(agent,out/'data.pt',completed,base_steps+actual_steps,meta)
        write(out/'summary.json',dict(status=status,unique_initial_rollout_fragments=finished,new_real_transitions=actual_steps,
            parent_completed_episodes=completed,completed_control_episodes_claimed=0,
            weights_targets_optimizers_unchanged=True,updates=agent.update_count,action_counts=action_counts,
            replay_size=len(agent.memory),wall_s=time.perf_counter()-started))

if __name__=='__main__':main()
