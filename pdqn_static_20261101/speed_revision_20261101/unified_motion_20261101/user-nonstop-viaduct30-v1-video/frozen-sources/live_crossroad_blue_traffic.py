"""Real blue oncoming traffic passes below the red tracked turning vehicle."""
import speed_bootstrap
import json,copy,shutil
from pathlib import Path
import live_control_latency as runner

original_materialize=runner.cross.materialize

def materialize(folder,item):
    path=original_materialize(folder,item);cfg=json.loads(path.read_text(encoding='utf-8'))
    blue=copy.deepcopy(cfg['targets'][0])
    blue.update(id='car_blue_foreground',color_rgb=[.015,.12,.85])
    blue['path']=dict(type='line',origin_cm=[5700.,7625.,8.],velocity_cm_s=[0.,-750.,0.],
        speed_profile=[[0.,1.],[23.,0.]],phase_s=0.,random_phase=False)
    cfg['targets'].append(blue)
    traffic=dict(role='Independent blue oncomingvehicle, no controller target ID input',speed_m_s=7.5,
        description='Opposite lane toward-Y. Initially outside camera; passes right-to-left in lower image near8..11s while red turns. Native world motion, never camera/video overlay.',
        path=blue['path'])
    runner.demo.MOTION.update(scenario='crossroad_red_turn_with_blue_foreground_traffic',secondary_traffic=traffic)
    cfg.update(name='user_crossroad_redturn_blue_oncoming_traffic',motion_schedule=runner.demo.MOTION,
        features=['two_vehicles','cross_intersection','rounded_turn','blue_foreground_traffic','fixed_installation','active_pan_tilt_zoom'],
        scope_note='Only adds independently moving blue scene actor. Same frozen red-initialized image association and P-DQN, no target-ID/color/GT direction rule. Native actual traffic motion and source60fps capture.')
    path.write_text(json.dumps(cfg,indent=2),encoding='utf-8')
    shutil.copy2(Path(__file__),folder/'frozen-sources'/Path(__file__).name)
    return path

if __name__=='__main__':
    runner.cross.materialize=materialize
    runner.main()
