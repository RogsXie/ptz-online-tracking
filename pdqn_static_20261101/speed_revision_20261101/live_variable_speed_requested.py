"""Requested alternating speeds on the same 200m road and tiny-corner start."""
import speed_bootstrap
import json
import live_unified_motion as demo

PROFILE=[[0.,.2],[3.,.5],[6.,1.],[10.,.35],[13.,.8],[17.,.15],[20.,1.],[25.,0.]]
original_materialize=demo.materialize

def materialize(folder,item):
    path=original_materialize(folder,item)
    cfg=json.loads(path.read_text(encoding='utf-8'))
    cfg['targets'][0]['path']['speed_profile']=PROFILE
    demo.MOTION.update(speed_profile=PROFILE,scenario='user_alternating_speeds_same_road',
        description='Piecewise speeds1.8,4.5,9,3.15,7.2,1.35,9m/s then stop at25s; actual native motion, no video acceleration',
        travel_before_stop_m=142.2)
    cfg.update(name='user_alternating_speeds_same_200m_road',motion_schedule=demo.MOTION,
        scope_note='Same initial geometry, road and frozen unified P-DQN as known9 comparison. Only native speed profile changes. Schedule/speed never enter policy. Piecewise speed changes, not continuous acceleration.')
    path.write_text(json.dumps(cfg,indent=2),encoding='utf-8');return path

if __name__=='__main__':
    runner=demo.runner;demo.CASE='known9'
    runner.create_agent=demo.create_agent;runner.resume=demo.core.resume;runner.frontend=demo.core.frontend
    runner.SpeedEnvironment=demo.Observation;runner.LOW=demo.core.LOW;runner.HIGH=demo.core.HIGH;runner.SECONDS_PER_RAW=demo.core.SECONDS
    runner.layout=demo.layout;runner.materialize=materialize;runner.AtomicTransport=demo.Transport;runner.write=demo.write
    runner.main()
