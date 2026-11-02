"""Verify changed coverage, FOV, partition and grouped-gate contracts."""
import hashlib
import math
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch
from .scenes import cases, layout, resolve, load_protocol, stage_spec, initial_zone
from .environment import CoverageEnvironment
from .agent import CoverageAgent
from .gates import assess_episode, gate_aggregate
from control_learning.environment import ControlEnvironment
from control_corners.run import write_json, frozen, network_hashes
from integration.runtime import REPRO


def main():
    torch.set_num_threads(2)
    checks = []
    protocol = load_protocol()
    for split in ('train', 'probe', 'validation'):
        plan = cases('static_center', split)
        groups = [layout(c['seed'], split)['group'] for c in plan]
        assert len(set(groups)) == 60
        assert len(set(c['seed'] for c in plan)) == len(plan)
    partitions = [{c['seed'] for c in cases('static_center', s)} for s in ('train','probe','validation','calibration')]
    assert all(not a & b for i,a in enumerate(partitions) for b in partitions[i+1:])
    try:
        cases('static_center', 'final_test')
    except ValueError:
        pass
    else:
        raise AssertionError('Reserved final test exposed')
    for a,b in zip(protocol['coverage']['train_fov_ranges_deg'], protocol['coverage']['validation_fov_ranges_deg']):
        assert a[1] < b[0]
    travel=[]; axis_ratios=[]
    for split in ('train','calibration'):
        for case in cases('static_center', split):
            item=layout(case['seed'],split);scene=resolve('static_center',split,case['seed']);cam=scene['camera']
            assert cam['location_cm']==[-2200,0,4500] and cam['rotation_deg']==[-60,0,0]
            assert cam['fov_deg']==cam['fov_min_deg']==cam['fov_max_deg']
            dx,dy,_=scene['targets'][0]['path']['origin_cm'];dx+=2200
            pitch=math.degrees(math.atan2(90-4500,math.hypot(dx,dy)));yaw=math.degrees(math.atan2(dy,dx))
            assert -80<=pitch<=-40 and -65<=yaw<=65
            travel.append((abs(pitch+60)+abs(yaw))/8)
            if item['layer']>0:
                ox,oy=[abs(c-.5) for c in item['planned_center']]
                axis_ratios.append(min(ox,oy)/max(ox,oy))
    assert max(travel)<12 and min(axis_ratios)<.25 and max(axis_ratios)>.95
    # All frustum rays remain downward and within the ground board. The upper
    # corner at FOV66 and pitch-40 bounds the horizontal radius for all cases.
    r=math.tan(math.radians(33));u=r*540/960;p=math.radians(-40)
    dz=math.sin(p)+u*math.cos(p)
    radius=4500*math.hypot(math.cos(p)-u*math.sin(p),r)/-dz
    steep=math.radians(-80)
    lower=-2200+4500*(min(0.,math.cos(steep)-u*abs(math.sin(steep)))-r*math.sin(math.radians(65)))/-dz
    assert dz<0 and radius<15000 and -2200+radius<26000 and lower>-14000
    checks.append(dict(check='partitions,2D spread,scale holdout and analytic full-ground view',passed=True,
                       ideal_max_serial_pan_tilt_s=max(travel),max_ground_radius_cm=radius,x_lower_bound_cm=lower))
    for fov in (41.25,52,66):
        before=dict(rotation=[-60.,0.,0.],sim_time_s=0.,fov=fov,visible=True,center_error=1.)
        env=CoverageEnvironment.__new__(CoverageEnvironment);env.rows=[before];env.fixed_fov=fov
        env.camera_contract=protocol['control_execution'];env.fixed_location=(-2200,0,4500)
        env.transport=SimpleNamespace(diagnostics={'camera_location':env.fixed_location})
        after=before|dict(rotation=[-60,.8,0],sim_time_s=.1,center_error=.9)
        with patch.object(ControlEnvironment,'step',return_value=(np.zeros(8),0.,False,after)):
            _,reward,_,row=env.step(3,.1)
        assert reward>0 and row['actuator_error_deg']<1e-6
        bad=after|{'fov':fov+.1}
        with patch.object(ControlEnvironment,'step',return_value=(np.zeros(8),0.,False,bad)):
            try:env.step(3,.1)
            except RuntimeError:pass
            else:raise AssertionError('Within-episode zoom escaped fixed-FOV contract')
    checks.append(dict(check='per-episode FOV diversity accepted, active zoom rejected',passed=True))
    sample=dict(visible=True,center_error=0.,x=0.,y=0.,area_ratio=.004,matched=True,fov=52.,action_type=0,duration_s=.25)
    rows=[sample|dict(sim_time_s=0.,duration_s=0.,initial_zone_confirmed=True)]+[sample|dict(sim_time_s=i*.25) for i in range(1,81)]
    good=assess_episode('static_center',rows);assert good['passed']
    rows[-1]['x']=.03;rows[-1]['center_error']=.03
    original=load_protocol()['default_episode_gate']
    import json
    assert original==json.loads((REPRO/'control_corners/protocol.json').read_text())['default_episode_gate']
    records=[good|dict(seed=c['seed']) for c in cases('static_center','validation')]
    assert gate_aggregate('static_center',records)['passed']
    near=next(i for i,r in enumerate(records) if layout(r['seed'],'validation')['layer']==0)
    records[near]=records[near]|{'passed':False}
    assert not gate_aggregate('static_center',records)['passed']
    records[near]=records[near]|{'passed':True}
    group=next(layout(r['seed'],'validation')['group'] for r in records if layout(r['seed'],'validation')['layer']>0)
    for record in records:
        if layout(record['seed'],'validation')['group']==group:record['passed']=False
    assert not gate_aggregate('static_center',records)['passed']
    assert not gate_aggregate('static_center',records[:-1])['passed']
    checks.append(dict(check='unchanged precision/stability; new grouped120case gates reject near-center failure and hidden cell failure',passed=True))
    for device in ('cpu','cuda'):
        agent=CoverageAgent(device=device)
        state=np.array([2.,-2.,-.5,.91,0,0,0,0],np.float32)
        for _ in range(256):
            action=agent.select_action(state,True);agent.observe(state,action,.1,state,False)
        assert agent.update()['q_loss']>=0
        receipt={}
        with frozen(agent,receipt):agent.select_action(state,False)
        assert receipt['networks_unchanged'] and receipt['counters_unchanged'] and receipt['sampling_rng_unchanged']
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'checkpoint.pt';agent.save(path);restored=CoverageAgent.load(path,device=device)
            assert agent.update()==restored.update() and network_hashes(agent)==network_hashes(restored)
        checks.append(dict(check='finite learning, frozen four networks, exact restore',device=device,passed=True))
    result=dict(passed=True,checks=checks,scope='Implementation and scene geometry only, not learned control qualification')
    write_json(REPRO/'verification/coverage-static-v8-20261031/contract-checks.json',result)
    print(result)


if __name__=='__main__':main()
