"""Frozen source baseline and a one-change termination diagnostic, actual UE."""
import bootstrap
from bootstrap import HERE,REPRO
from calibrate import write
from scenes import layout,materialize
import cv2,json,numpy as np,torch,warnings,time
from author_aligned.source import create_agent,load_core
from author_aligned.environment import AtomicTransport,bind_environment
from author_aligned.run import NETWORKS,image_metrics
from integration.runtime import StandaloneArena
from integration.experiment import make_extractor
from control_model_v13.detector import create_detector

def main():
    warnings.filterwarnings('ignore');torch.set_num_threads(4)
    out=HERE/'source-baseline';out.mkdir(exist_ok=False)
    agent=create_agent(1,'cuda')
    checkpoint=REPRO/'runs/author-source-ue-static-v1-to300/interrupted.pt'
    saved=torch.load(checkpoint,map_location='cuda',weights_only=False)
    for name in NETWORKS:getattr(agent,name).load_state_dict(saved[name])
    agent.epsilon=0
    detector=create_detector(checkpoint=REPRO/'weights/yolo26s_ue_overhead_v1.pt',device='cuda',threshold=.15,imgsz=960)
    detector.warmup();extractor=make_extractor('cuda');_,pad,_=load_core()
    summary=[]
    write(out/'configuration.json',dict(checkpoint=str(checkpoint),trained_episodes=saved['completed_episodes'],
        source_environment='complete original class via author_aligned',current_detector=detector.provenance,
        task_adaptation='fixed overhead static scene; calibrated rates 4/2/5; raw/5 execution clip only',
        protocols=['source_exact','confidence_termination_disabled_only'],frozen_updates=0))
    with StandaloneArena(out,headless=True) as arena:
        t=AtomicTransport(arena.port,hud=False)
        try:
            for profile in ('source_exact','confidence_termination_disabled_only'):
                env=bind_environment(t,detector,extractor,'clip_box')
                for i,index in enumerate((0,1,2,7,8,14,21,27,34,41,48,54)):
                    item=layout('calibration',index);folder=out/profile/f'case{index:03d}';folder.mkdir(parents=True)
                    scene=materialize(folder,item);env.io_bridge.scenario=scene
                    env.io_bridge.seed=item['seed'];env.io_bridge.target_count=1
                    state=np.asarray(env.reset(),np.float32);rows=[];streak=0.;maximum=0.;start=time.perf_counter()
                    cv2.imwrite(str(folder/'initial.jpg'),cv2.cvtColor(t.rgb,cv2.COLOR_RGB2BGR))
                    writer=cv2.VideoWriter(str(folder/'actual-action-end-frames.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),10,(960,540))
                    writer.write(cv2.cvtColor(t.rgb,cv2.COLOR_RGB2BGR))
                    try:
                        for step in range(128):
                            if profile!='source_exact':env.end_iteration=-100000
                            action,parameter,parameters=agent.choose_action(state,train=True)
                            next_state,reward,done,_=env.step(pad(action,parameter))
                            next_state=np.asarray(next_state,np.float32)
                            m=image_metrics(next_state)
                            quality=bool(m['visible'] and m['center_error']<=.05 and .025<=m['area_ratio']<=.055)
                            streak=streak+env.io_bridge.last_action['duration_s'] if quality else 0.;maximum=max(maximum,streak)
                            row=dict(step=step,input=state.tolist(),next=next_state.tolist(),reward=float(reward),done=bool(done),
                                **env.io_bridge.last_action,**m,sim_time=t.diagnostics['sim_time'])
                            rows.append(row);state=next_state
                            # A sample-hold diagnostic video, explicitly not realtime sensor capture.
                            for _ in range(max(1,round(env.io_bridge.last_action['duration_s']*10))):
                                writer.write(cv2.cvtColor(t.rgb,cv2.COLOR_RGB2BGR))
                            if done:break
                    finally:writer.release()
                    cv2.imwrite(str(folder/'final.jpg'),cv2.cvtColor(t.rgb,cv2.COLOR_RGB2BGR))
                    write(folder/'steps.json',rows)
                    result=dict(profile=profile,index=index,source_success=env.object_num==0,steps=len(rows),
                        final_center=m['center_error'],final_area=m['area_ratio'],hold_max_s=maximum,quality_success=maximum>=5.,wall_s=time.perf_counter()-start)
                    summary.append(result);write(out/'summary.json',summary);print(result,flush=True)
        finally:t.close()

if __name__=='__main__':main()
