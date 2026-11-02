"""Single frozen P-DQN, real-time UE acquisition through motion and stop."""
import speed_bootstrap
import sys,json,math,argparse
import numpy as np
import live_speed_v5 as runner
import unified_core as core
from scenes import configuration,INIT,ground_position
from unified_scenes import road

AGENT=None
CASE='known9'
MOTION={}

def create_agent():
    global AGENT
    AGENT=core.create_agent();return AGENT

class Observation(core.Observation):
    def observe(self,*args,**kwargs):
        state,row=super().observe(*args,**kwargs)
        if AGENT is not None:
            AGENT.camera_diagnostics=self.t.diagnostics;AGENT.camera_limits=self.camera_limits
        if row['visible']:
            from unified_perception import bearing,project
            diag=self.t.diagnostics;angles=bearing(((row['x']+1)/2,(row['y']+1)/2),diag)
            age=min(.2,max(0.,row['detector_s']+row['reid_s']+float(diag.get('capture_readback_seconds',0.))+.003))
            xy=project(angles+np.asarray(row['image_motion_bearing_rate_deg_s'])*age,diag)
            actual=state.copy();state=state.copy();state[:2]=np.clip(2*np.asarray(xy)-1,-1.5,1.5)
            row.update(observed_state=actual.tolist(),latency_compensated_state=state.tolist(),measured_perception_age_s=age)
        row['controller_architecture']='one unified P-DQN, no phase switch'
        return state,row

class Transport(runner.AtomicTransport):
    def rpc(self,command):
        if command.startswith('vset /ptz/play '):
            p,t,z,d=map(float,command.split()[2:]);g,zg=core.gain(float(self.diagnostics['fov']))
            command=f'vset /ptz/play {p*g:.12f} {t*g:.12f} {z*zg:.12f} {d:.12f}'
        return super().rpc(command)

def layout(split,index):
    return dict(split='development',index=index,seed={'known9':39500000,'variable':40501001,'staticcorner':40502002,'staticlarge':40503003}[CASE],
        quadrant=0,layer=4,scale_group=0,planned_center=[.075,.09] if CASE in ('known9','staticcorner') else [.24,.72] if CASE=='variable' else [.7,.68],
        fov_deg=66. if CASE in ('known9','staticcorner') else 42. if CASE=='variable' else 16.,camera_height_cm=4500. if CASE=='staticlarge' else 6500.)

def materialize(folder,item):
    global MOTION
    cfg=configuration(item,(180.,100.,100.));init=dict(INIT,camera_location_cm=[-2200.,0.,item['camera_height_cm']])
    origin=ground_position(*item['planned_center'],item['fov_deg'],init)
    cfg['camera'].update(location_cm=init['camera_location_cm'],pitch_limits_deg=[-85.,-15.],fov_min_deg=8.)
    direction=[.6,.8] if CASE=='variable' else [0.,1.]
    profile=[[0.,1.],[18.,0.]] if CASE=='known9' else [[0.,.65],[2.,1.],[5.,.4],[7.,0.],[9.,.85],[13.,.2],[16.,0.],[18.,.9],[23.,0.]] if CASE=='variable' else [[0.,0.]]
    speed=9. if CASE in ('known9','variable') else 0.
    cfg['targets'][0]['path']=dict(type='line',origin_cm=origin,velocity_cm_s=[100*speed*v for v in (*direction,0.)],speed_profile=profile,phase_s=0.,random_phase=False)
    road(cfg,origin,direction,20000.)
    # Known 9m/s comparison retains the exact road and initial geometry.
    if CASE=='known9':
        travel=16200.
        cfg['occluders']=[dict(kind='road_surface',location_cm=[origin[0],origin[1]+travel/2,.15],size_cm=[900.,20000.,.2],color_rgb=[.13,.14,.14])]
        for side in (-1,1):
            for i in range(-3,int(math.ceil(travel/500))+4):
                cfg['occluders'].append(dict(kind='road_paint',location_cm=[origin[0]+side*350.,origin[1]+i*500.,.5],size_cm=[12.,250.,.2],color_rgb=[.78,.78,.70]))
    MOTION=dict(speed_m_s=speed,speed_profile=profile,velocity_cm_s=[100*speed*v for v in (*direction,0.)],road_length_m=200.,scenario=CASE)
    cfg.update(name='unified_live_'+CASE,motion_schedule=MOTION,scope_note='Native real motion, original wall capture. Target pose/speed/schedule never provided to controller. Known9 comparison or focused heldout demonstration, not statistical qualification.')
    folder.mkdir(parents=True,exist_ok=True);path=folder/'scene.json';path.write_text(json.dumps(cfg,indent=2),encoding='utf-8');return path

original_write=runner.write
def write(path,value):
    if path.name=='configuration.json':
        value.update(motion=MOTION,architecture='Single15input P-DQN, seven discrete actions with continuous duration, four networks. No coarse policy or stage switch.',
            execution_profile=core.PROFILE,rates=[180.,100.,100.],requested_vehicle_speed_m_s=MOTION['speed_m_s'],road_length_m=200.,
            qualification='Focused development/heldout demonstrations only; not broad qualification',
            extensions='RGB/camera motion features, image-latency prediction capped200ms, squared opticsgain4, short exclusive pulses, mechanical-limit Q mask, frozen adapted detector and RGB association.',
            no_groundtruth_policy_or_reward=True,no_video_acceleration=True)
    return original_write(path,value)

if __name__=='__main__':
    p=argparse.ArgumentParser(add_help=False);p.add_argument('--case',choices=['known9','variable','staticcorner','staticlarge'],default='known9')
    extra,remaining=p.parse_known_args();CASE=extra.case;sys.argv=[sys.argv[0],*remaining]
    runner.create_agent=create_agent;runner.resume=core.resume;runner.frontend=core.frontend;runner.SpeedEnvironment=Observation
    runner.LOW=core.LOW;runner.HIGH=core.HIGH;runner.SECONDS_PER_RAW=core.SECONDS
    runner.layout=layout;runner.materialize=materialize;runner.AtomicTransport=Transport;runner.write=write
    runner.main()
