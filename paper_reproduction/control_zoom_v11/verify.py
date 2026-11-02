"""Numerical, freeze, action and joint precision contracts; no UE or real training."""
from pathlib import Path
import json
import tempfile
from unittest.mock import patch
import numpy as np
import torch
from .agent import ZoomAgent,potential_offset
from .environment import temporal_reward,image_cost
from .scenes import cases,probe_cases,layout,STAGE
from .gates import assess_episode
from .rollout import evaluate,PolicyReviewRequired
from control_coverage_v8.run import frozen,network_hashes,write_json,sha256
from control_v6.temporal import discount,DECAY_PER_SECOND
from integration.runtime import REPRO
from .run import review
from .detector import ScaleFallbackDetector
from hascam.detection import Detection
from types import SimpleNamespace
import cv2


def io_contracts():
    from .rollout import episode
    from .video import CaseVideo
    box=(.4,.4,.6,.6)
    template=dict(sim_time_s=0.,visible=True,matched=True,center_error=0.,x=0.,y=0.,area_ratio=.04,
        fov=22.,rotation=[-60.,0.,0.],action_type=0,duration_s=0.,reward=0.,selected_bbox=list(box),
        confidence=.9,zoom=0.,terminated=False,truncated=False,bootstrap_from_actual_next_observation=True)
    state=np.array([0.,0.,0.,.9,0.,0.,0.,0.],np.float32)
    class RecordedEnvironment:
        def __init__(self,*args):
            self.rows=[template.copy()];self.steps=0;self.detections=[Detection(box,.9,0,'car')]
        def reset(self,*args):return state.copy()
        def step(self,action,duration):
            self.steps+=1;done=self.steps==501
            row=template|dict(sim_time_s=self.steps*.04,action_type=action,duration_s=duration,truncated=done)
            self.rows.append(row);return state.copy(),0.,done,row
    transport=SimpleNamespace(rgb=np.zeros((540,960,3),np.uint8),diagnostics={})
    agent=ZoomAgent(device='cpu')
    with tempfile.TemporaryDirectory() as directory:
        folder=Path(directory)
        with patch('control_zoom_v11.rollout.ImprovedEnvironment',RecordedEnvironment),patch('control_zoom_v11.rollout.initial_zone',return_value=True):
            result=episode(cases(STAGE,'train',1)[0],folder/'case',transport,None,None,agent,True,folder)
        assert result['initial_task_ready'] is True and result['steps']==501
        assert json.loads((folder/'case/summary.json').read_text())['initial_task_ready'] is True
        video=CaseVideo(folder,0)
        try:
            video.write(transport.rgb,template);video.write(transport.rgb,template|{'sim_time_s':.04})
        finally:video.close()
        for name in ('sensor-rgb.mp4','policy-boxed.mp4'):
            cap=cv2.VideoCapture(str(folder/name));count=0
            while cap.read()[0]:count+=1
            cap.release();assert count==2
    detector=ScaleFallbackDetector.__new__(ScaleFallbackDetector)
    calls=[]
    def base(rgb):
        calls.append(rgb)
        return [] if len(calls)%2 else [Detection((.35,.4,.55,.6),.8,0,'car')]
    detector.base=base
    mapped=detector(np.zeros((540,960,3),np.uint8))
    assert np.allclose(mapped[0].bbox,[.2,.3,.6,.7]) and detector.last_fallback
    detector.base=lambda rgb:[Detection(box,.9,0,'car')]
    assert detector(np.zeros((540,960,3),np.uint8))[0].bbox==box and not detector.last_fallback
    # The full sixty-case result must be compared on common seeds, not raw counts.
    previous=dict(episodes=[dict(seed=i,passed=True) for i in range(20)])
    entries=[dict(seed=i,passed=i<30,initial_task_ready=False,q_bound_diagnostic_count=501,
        q_bound_violation_count=0,max_q_bound_excess=-1.) for i in range(60)]
    probe=dict(episodes=entries,collapsed_cases=0,lost_cases=0,planned_total=60,reversal_mean=13.,
        complete=True,passed_cases=30)
    assert any('common seeds' in reason for reason in review(probe,previous,60))
    return dict(complete_rollout_serialization=True,video_encoding_decoded=True,
        rgb_fallback_inverse_coordinates=True,full_rgb_preserved=True,review_common_seed_comparison=True)


def main():
    torch.set_num_threads(4);checks=[]
    for device in ('cpu','cuda'):
        agent=ZoomAgent(device=device);state=np.array([.8,-.4,-.5,.9,0,0,0,0],np.float32)
        before=network_hashes(agent);actions=set();updates=[]
        for _ in range(400):
            action=agent.select_action(state,True);actions.add(action[0]);assert action[1]==.04
            agent.observe(state,action,.01,state,False)
            value=agent.update()
            if value is not None:updates.append(value)
        assert actions==set(range(7)) and len(updates)==37
        assert all(network_hashes(agent)[n]==before[n] for n in ('param_net','param_net_target'))
        assert not agent.param_net_optimiser.state
        receipt={}
        with frozen(agent,receipt):
            for _ in range(7):agent.select_action(state,False)
        assert all(receipt[k] for k in ('four_networks_eval','four_networks_requires_grad_false','networks_unchanged','counters_unchanged','sampling_rng_unchanged'))
        states=torch.tensor([[.8,-.4,-.5,.9,0,0,0,0],[0,0,0,.8,0,0,0,0],[0,0,.5,0,0,0,0,0]],device=device)
        nxt=states.roll(1,0);rewards=torch.tensor([1.,-.3,-2.],device=device)
        dones=torch.tensor([False,False,True],device=device);durations=torch.full((3,),.04,device=device)
        residual=agent.residual_target(states,rewards,nxt,dones,durations)
        with torch.no_grad():
            parameters=agent._project(agent.param_net(nxt));chosen=agent.actor_net(nxt,parameters).argmax(1)
            ordinary=rewards+(~dones).float()*torch.exp(-DECAY_PER_SECOND*durations)*agent.actor_target(nxt,parameters).gather(1,chosen[:,None]).squeeze(1)
        assert torch.allclose(residual+potential_offset(states),ordinary,atol=2e-5)
        for tensor in states:
            row={'visible':bool(tensor[3]>0),'center_error':float(torch.linalg.vector_norm(tensor[:2]/4)), 'area_ratio':.04*np.exp(float(tensor[2])*4)}
            assert abs(float(potential_offset(tensor[None]))-20*image_cost(row))<2e-5
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'agent.pt';agent.save(path);restored=ZoomAgent.load(path,device=device)
            assert network_hashes(restored)==network_hashes(agent)
            for _ in range(4):
                action=agent.select_action(state,True);assert action==restored.select_action(state,True)
                for a in (agent,restored):a.observe(state,action,.01,state,False)
                assert agent.update()==restored.update()
            assert network_hashes(restored)==network_hashes(agent)
        checks.append(dict(device=device,seven_actions=True,update_schedule=True,exact_state_cost=True,
            bellman_equivalence_with_termination=True,parameter_frozen=True,checkpoint_rng_restore=True))
    for area,passed in [(.04,True),(.01,False),(.09,False)]:
        rows=[dict(sim_time_s=i*.04,visible=True,matched=True,center_error=.01,x=.01,y=0.,
            area_ratio=area,fov=23.,action_type=0,duration_s=.04 if i else 0.,initial_zone_confirmed=True,
            zoom=float(np.log(area/.04))) for i in range(502)]
        result=assess_episode(STAGE,rows);assert result['passed']==passed
        if not passed:assert not result['terminal_checks']['scale_log_error']
    before={'visible':True,'center_error':.1,'area_ratio':.01};after={'visible':True,'center_error':.1,'area_ratio':.02}
    r,parts=temporal_reward(before,after,5,.04)
    assert abs(r-20*(image_cost(before)-discount(.04)*image_cost(after))-parts['rate_reward']-parts['effort'])<1e-8
    plan=cases(STAGE,'train',60);assert len({layout(c['seed'],'train')['group'] for c in plan})==60
    subset=probe_cases();assert len(subset)==20
    assert len({(layout(c['seed'],'probe')['quadrant'],layout(c['seed'],'probe')['scale_group']) for c in subset})==20
    for split in ['final','reserved_final_test']:
        try:cases(STAGE,split)
        except ValueError:pass
        else:raise AssertionError('Final-test access unexpectedly allowed')
    agent=ZoomAgent(device='cpu');agent.initialize_from_v10(REPRO/'runs/potential-factored-v10-attempt1/checkpoint-0060.pt')
    assert len(agent.memory)==agent.update_count==agent.environment_steps==0 and not agent.actor_optimiser.state
    def failed_case(*args,**kwargs):
        return dict(seed=11200000,passed=False,failures=['terminal_stability'],coverage={'layer':0},initial_task_ready=True,
            terminal_center_mean=.03,terminal_scale_log_error=.05,reward_sum=0.,direction_reversals=0,
            longest_loss_run_seconds=0.,stay_actions=501,saturated_nonstay=0,steps=501,max_q_bound_excess=-1.)
    with tempfile.TemporaryDirectory() as directory:
        out=Path(directory)/'probe'
        with patch('control_zoom_v11.rollout.episode',side_effect=failed_case) as mocked:
            try:evaluate([{'seed':11200000,'split':'probe'},{'seed':11200001,'split':'probe'}],out,None,None,None,agent,Path(directory))
            except PolicyReviewRequired:pass
            else:raise AssertionError('Ready case failure must stop')
            assert mocked.call_count==1
        f=json.loads((out/'freeze-receipt.json').read_bytes());r=json.loads((out/'report.json').read_bytes())
        assert f==r['freeze'] and f['networks_unchanged'] and not r['complete']
    checks.append(dict(joint_precision_gate=True,ready_case_failure_stop=True,actual_exception_freeze_receipt=True,
                       balanced60_and20=True,final_test_sealed=True,warm_start_empty_replay_optimizer=True))
    checks.append(io_contracts())
    result=dict(passed=True,checks=checks,scope='Implementation contracts only; no control-success claim',
                source_sha256={p.name:sha256(p) for p in Path(__file__).parent.glob('*.py')})
    write_json(REPRO/'verification/active-zoom-v11-20261031/contract-checks.json',result)
    print(json.dumps(result))


if __name__=='__main__':main()
