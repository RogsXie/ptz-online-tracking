"""Temporal invariance, replay action-duration and unchanged gate regressions."""
import json
import math
import tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from control_v6.temporal import discount, discounted_seconds, temporal_reward, DECAY_PER_SECOND
from control_v6.agent import TemporalAgent
from control_v6.environment import TemporalEnvironment
from control_v6 import scenes
from control_v5.agent import ContinuingAgent
from control_v5.environment import ContinuingEnvironment
from control_v5.run import frozen, network_hashes, write_json
from integration.runtime import REPRO


def observation(cost):
    return {"visible": True, "center_error": cost}


def constant_agent(device, duration, terminal=False):
    agent=TemporalAgent(seed=92,device=device)
    with torch.no_grad():
        for net in (agent.actor_net,agent.actor_target):
            for p in net.parameters():p.zero_()
            net.base.layers[-1].bias.fill_(2.)
    s=np.array([.04,0.,-.5,.9,0.,0.,0.,0.],np.float32)
    parameters=np.full(7,.145,np.float32)
    parameters[0]=.04 if duration>.1 else .25
    parameters[4]=duration
    for _ in range(256):agent.observe(s,(4,duration),0.,s,terminal,parameters)
    return agent


def main():
    torch.set_num_threads(2)
    checks=[]
    def checked(name, **values):checks.append({"check":name,"passed":True,**values})
    a,b=.04,.21
    assert abs(discount(a+b)-discount(a)*discount(b))<1e-14
    assert abs(discounted_seconds(a+b)-(discounted_seconds(a)+discount(a)*discounted_seconds(b)))<1e-14
    values=[]
    for d in [.04,.145,.25]:
        reward,_=temporal_reward(observation(.03),observation(.03),0,d)
        value=reward/(1-discount(d));values.append(value)
    assert max(values)-min(values)<1e-12 and max(values)<0
    checked("identical stationary physical process has duration-invariant negative off-center return",values=values)
    phi=lambda c:-20*c
    costs=[.05,.2,.03]
    left=(discount(a)*phi(costs[1])-phi(costs[0]))+discount(a)*(discount(b)*phi(costs[2])-phi(costs[1]))
    right=discount(a+b)*phi(costs[2])-phi(costs[0])
    assert abs(left-right)<1e-14
    checked("discount and rate integrals compose; potential shaping telescopes across segment subdivisions")
    row=observation(.03)|{"terminated":False,"truncated":True,"bootstrap_from_actual_next_observation":True}
    state=np.array([.12,0.,-.5,.9,0.,0.,0.,0.],np.float32)
    env=TemporalEnvironment.__new__(TemporalEnvironment);env.previous=observation(.05);env.rows=[row]
    with patch.object(ContinuingEnvironment,"step",return_value=(state,123.,True,row)):
        actual,reward,done,record=env.step(4,.25)
    expected,parts=temporal_reward(observation(.05),observation(.03),4,.25)
    assert actual is state and reward==expected and done and record['truncated'] and not record['terminated']
    assert env.rows[-1]['reward']==expected and record['reward_parts']==parts
    checked("environment replaces only temporal reward and preserves exact observation/truncation/audit row")
    for device in ["cpu","cuda"]:
        for duration in [.04,.25]:
            agent=constant_agent(device,duration)
            result=agent.update()
            expected=.5*(2-2*discount(float(np.float32(duration))))**2
            assert abs(result['q_loss']-expected)<1e-7,(result,expected)
            checked("Bellman target discounts executed action4 duration, not Stay or proposed next duration",device=device,duration=duration,q_loss=result['q_loss'])
        terminal=constant_agent(device,.25,True)
        assert abs(terminal.update()['q_loss']-1.5)<1e-7
        receipt={}
        with frozen(agent,receipt):agent.select_action(state,False)
        assert all(receipt[k] for k in ['networks_unchanged','sampling_rng_unchanged','counters_unchanged'])
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'checkpoint.pt';agent.save(p)
            restored=TemporalAgent.load(p,device=device)
            assert agent.update()==restored.update()
            assert network_hashes(agent)==network_hashes(restored)
            ContinuingAgent(seed=92,device=device).save(p)
            try:TemporalAgent.load(p,device=device)
            except ValueError:pass
            else:raise AssertionError('Old objective checkpoint accepted')
        checked("true terminal masks Q; frozen evaluation; exact restore; v5 weights/replay rejected",device=device)
    expected=scenes.cases('static_center','validation')
    assert len(expected)==40 and {x['seed'] for x in expected}==set(range(510040,510080))
    for seed in [510040,510079]:
        value=scenes.resolve('static_center','validation',seed)
        value['control_protocol'].pop('evaluation_manifest_sha256')
        with patch.object(scenes.original,'_partition',return_value=(510000,80)):
            reference=scenes.original.resolve('static_center','validation',seed)
        assert value==reference
    protocol_sha=scenes.MANIFEST['quality_protocol_sha256']
    rows=[{'seed':c['seed'],'stage':'static_center','passed':True,'protocol_sha256':protocol_sha} for c in expected]
    assert scenes.gate_aggregate('static_center',rows)['passed']
    rows[0]['passed']=False;rows[1]['passed']=False
    assert scenes.gate_aggregate('static_center',rows)['passed']
    rows[1]['passed']=True;rows[4]['passed']=False
    assert not scenes.gate_aggregate('static_center',rows)['passed']
    prior=[r|{'seed':r['seed']-40,'passed':True} for r in rows]
    assert not scenes.gate_aggregate('static_center',prior)['passed']
    try:scenes.cases('static_center','final_test')
    except ValueError:pass
    else:raise AssertionError('Final test accessible')
    checked("new40-seed block preserves scene distribution; 38/40 and9/10 gates unchanged; old/final-test seeds rejected")
    write_json(REPRO/'verification/temporal-static-v6-20261031/contract-checks.json',
        {'passed':True,'checks':checks,'scope':'implementation and mathematical invariants, not live control success'})
    evidence=[]
    for duration in [.04,.145,.25]:
        evidence.append({'duration_s':duration,'v5_discount_after1s':.98**(1/duration),
            'v5_constant_unit_rate_value':duration/(1-.98),'v6_discount_after1s':discount(duration)**(1/duration),
            'v6_constant_unit_rate_value':discounted_seconds(duration)/(1-discount(duration))})
    training=REPRO/'runs/continuing-static-v5-attempt1/training'
    stable=[json.loads(line) for path in training.glob('*/steps.jsonl') for line in path.read_text().splitlines(keepends=True) if line.endswith('\n')]
    stable=[r for r in stable if not r.get('initial') and r['action_type']==0 and abs(r['reward_parts']['image_progress'])<.01]
    write_json(REPRO/'verification/temporal-static-v6-20261031/time-consistency-evidence.json',
        {'basis':'mathematical counterexample verified against v5 code; not established sole cause of five holdout failures',
         'v5_nearly_stationary_training_stay_records':len(stable),
         'v5_recorded_duration_range':[min(r['duration_s'] for r in stable),max(r['duration_s'] for r in stable)],
         'decay_per_second':DECAY_PER_SECOND,'examples':evidence,'v6_stay_outside_gate_value':values})
    print({'passed':True,'checks':len(checks),'evidence_rows':len(stable)})


if __name__=='__main__':main()
