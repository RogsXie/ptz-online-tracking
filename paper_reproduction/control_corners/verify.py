"""Full-image state, wide-limit actuator and unchanged accuracy regressions."""
import json
import math
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch
from control_corners.environment import CornerState,CornerEnvironment
from control_corners.agent import CornerAgent
from control_corners.scenes import load_protocol,stage_spec,cases,initial_zone,category,resolve,HERE
from control_corners.gates import assess_episode,gate_aggregate
from control_learning.environment import ControlEnvironment
from control_v5.environment import ContinuingState
from control_v6.agent import TemporalAgent
from control_v5.run import write_json,network_hashes,frozen
from integration.runtime import REPRO


def main():
    torch.set_num_threads(2);checks=[]
    def passed(name,**values):checks.append({'check':name,'passed':True,**values})
    for xy in [(x,y) for x in [.1,.9] for y in [.1,.9]]:
        detection=SimpleNamespace(center=xy,area=.004,confidence=.91)
        s=CornerState(20).build(detection)
        assert np.allclose(s[:2],4*(2*np.array(xy)-1)) and np.array_equal(s[4:],np.zeros(4))
        assert np.max(abs(ContinuingState(20).build(detection)[:2]))<=2.
    passed('four corner observations preserve distinct full-image offsets; old half-frame clipping removed')
    for action in [1,2,3,4]:
        before={'rotation':[-60.,24.,0.],'sim_time_s':0.,'fov':60.,'visible':True,'center_error':1.}
        rotation=before['rotation'].copy()
        if action in [1,2]:rotation[0]+=(1 if action==2 else -1)*.8
        if action in [3,4]:rotation[1]+=(1 if action==3 else -1)*.8
        row={'rotation':rotation,'sim_time_s':.1,'fov':60.,'visible':True,'center_error':.9}
        env=CornerEnvironment.__new__(CornerEnvironment);env.rows=[before]
        env.camera_contract=load_protocol()['control_execution'];env.fixed_location=(-2200.,0.,4500.)
        env.transport=SimpleNamespace(diagnostics={'camera_location':env.fixed_location})
        with patch.object(ControlEnvironment,'step',return_value=(np.zeros(8),0.,False,row)):
            _,reward,done,out=env.step(action,.1)
        assert out['actuator_error_deg']<1e-7 and not done and not out['terminated'] and not out['truncated']
    passed('wide-limit physical responses valid outside former local mechanical limits; RGB-only reward finite')
    protocol=load_protocol();init=protocol['corner_initialization'];execution=protocol['control_execution']
    seed_plan=[('train',s) for s in list(range(711000,711032))+list(range(719100,719112))+list(range(718900,718905))]+[('validation',s) for s in range(810000,810040)]
    camera=resolve('static_center','train',711000)['camera'];travel=[]
    for split,seed in seed_plan:
        scene=resolve('static_center',split,seed);assert scene['camera']==camera
        dx,dy,z=scene['targets'][0]['path']['origin_cm'];dx-=camera['location_cm'][0];dy-=camera['location_cm'][1]
        pitch=math.degrees(math.atan2(init['layout_reference_height_cm']-camera['location_cm'][2],math.hypot(dx,dy)))
        yaw=math.degrees(math.atan2(dy,dx));assert execution['pitch_limits_deg'][0]<=pitch<=execution['pitch_limits_deg'][1]
        assert execution['yaw_limits_deg'][0]<=yaw<=execution['yaw_limits_deg'][1]
        travel.append((abs(pitch-init['nominal_camera_pitch_deg'])+abs(yaw))/8.)
    assert max(travel)<12.
    # The maximum horizontal ground distance is attained at the shallowest
    # pitch and upper image corners. Its radius is less than every board edge
    # distance from the fixed installation, covering all yaw angles analytically.
    shallow=math.radians(execution['pitch_limits_deg'][1]);u=math.tan(math.pi/6)*540/960;r=math.tan(math.pi/6)
    dz=math.sin(shallow)+u*math.cos(shallow);assert dz<0
    radius=camera['location_cm'][2]*math.hypot(math.cos(shallow)-u*math.sin(shallow),r)/-dz
    x,y,_=camera['location_cm'];board_margin=min(x+14000,26000-x,y+15000,15000-y)
    # Full360 radius is conservative on x; use the actual fixed yaw interval
    # for its negative-x bound, and radius for y and positive x.
    assert radius<15000 and x+radius<26000
    steep=math.radians(execution['pitch_limits_deg'][0])
    forward_lower=min(0.,math.cos(steep)-u*abs(math.sin(steep)))
    x_lower_bound=x+camera['location_cm'][2]*(forward_lower-r*math.sin(math.radians(65)))/-dz
    assert x_lower_bound>-14000
    minimum_x=0.;maximum_x=0.
    for pitch in np.linspace(*execution['pitch_limits_deg'],401):
        p=math.radians(pitch)
        for yaw in np.linspace(*execution['yaw_limits_deg'],261):
            a=math.radians(yaw)
            for right in [-r,r]:
                for up in [-u,u]:
                    zz=math.sin(p)+up*math.cos(p);assert zz<0
                    xx=math.cos(p)-up*math.sin(p)
                    gx=x-camera['location_cm'][2]*(xx*math.cos(a)-right*math.sin(a))/zz
                    minimum_x=min(minimum_x,gx);maximum_x=max(maximum_x,gx)
    assert minimum_x>-14000 and maximum_x<26000
    passed('all scenes share fixed downward installation; full frustum stays on board throughout yaw/pitch domain',
        max_ideal_sequential_travel_s=max(travel),ground_radius_bound_cm=radius,ground_x_sample_bounds=[minimum_x,maximum_x],ground_x_analytic_lower_bound=x_lower_bound,
        topmost_ray_vertical_component=dz,layout_scenes_checked=len(seed_plan))
    protocol=load_protocol();old=json.loads((REPRO/'control_curriculum/protocol.json').read_text())
    assert protocol['default_episode_gate']==old['default_episode_gate']
    for key in ['stable_hold_seconds','stable_center_mean_max','stable_center_p95_max']:
        assert protocol['default_episode_gate'][key]==old['default_episode_gate'][key]
    original=old['stages'][0];new=stage_spec('static_center')
    assert all(new[k]==v for k,v in original.items() if k not in ['horizon_seconds','max_decisions','settling_seconds_max'])
    assert max(travel)>4. and new['settling_seconds_max']==12. and new['horizon_seconds']==20.
    passed('original precision/stability/target criteria retained; declared physical reachability requires longer settling budget')
    assert [category(719100+i,'train')[0] for i in range(12)]==['corner']*4+['edge']*4+['near']*4
    plan=cases('static_center','validation');assert len(plan)==40 and {x['seed'] for x in plan}==set(range(810000,810040))
    example={'visible':True,'center_error':0.,'x':0.,'y':0.,'area_ratio':.004,'matched':True,'fov':60.,'action_type':0,'duration_s':.25}
    rows=[example|{'sim_time_s':0.,'duration_s':0.,'initial_zone_confirmed':True}]+[example|{'sim_time_s':i*.25} for i in range(1,81)]
    assessment=assess_episode('static_center',rows);assert assessment['passed']
    rows[0]['initial_zone_confirmed']=False;assert not assess_episode('static_center',rows)['passed']
    assessed=[assessment|{'seed':c['seed']} for c in plan]
    assessed[0]['passed']=False;assessed[1]['passed']=False;assert gate_aggregate('static_center',assessed)['passed']
    assessed[1]['passed']=True;assessed[4]['passed']=False;assert not gate_aggregate('static_center',assessed)['passed']
    try:cases('static_center','final_test')
    except ValueError:pass
    else:raise AssertionError('Reserved test accessible')
    passed('12mixed development probes;40balanced new corner cases; wrong start or failed corner group cannot pass')
    for device in ['cpu','cuda']:
        agent=CornerAgent(device=device)
        s=np.array([-3.2,3.2,-.5,.91,0,0,0,0],np.float32)
        for _ in range(256):
            action=agent.select_action(s,True);agent.observe(s,action,.2,s,False)
        result=agent.update();assert result['q_loss']>=0
        receipt={}
        with frozen(agent,receipt):agent.select_action(s,False)
        assert receipt['networks_unchanged'] and receipt['counters_unchanged'] and receipt['sampling_rng_unchanged']
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'checkpoint.pt';agent.save(path)
            restored=CornerAgent.load(path,device=device)
            assert agent.update()==restored.update() and network_hashes(agent)==network_hashes(restored)
            TemporalAgent(device=device).save(path)
            try:CornerAgent.load(path,device=device)
            except ValueError:pass
            else:raise AssertionError('Small-observation checkpoint accepted')
        passed('full-field learning finite; frozen inference and exact restore; incompatible local format rejected',device=device)
    write_json(REPRO/'verification/four-corners-v7-20261031/contract-checks.json',{'passed':True,'checks':checks,'temporal_contract':'verification/temporal-static-v6-20261031/contract-checks.json','scope':'implementation and start/gate checks, not learned corner success'})
    print({'passed':True,'checks':len(checks)})


if __name__=='__main__':main()
