"""Requested9m/s uniform straight travel on an extended200m road, then stop."""
import speed_bootstrap
import json,math
import live_motion_tracking_latency as latency
from motion_learning15 import LOW,HIGH,SECONDS

hierarchy=latency.hierarchy
runner=latency.runner
scene=hierarchy.scene
SPEED=9.
STOP=18.
MOTION=dict(scene.MOTION,speed_m_s=SPEED,velocity_cm_s=[0.,100*SPEED,0.],
    start_sim_s=0.,stop_sim_s=STOP,speed_profile=[[0.,1.],[STOP,0.]],loop=False,
    road_length_m=200.,travel_before_stop_m=SPEED*STOP,
    description='Uniform9m/s straighttravel18s (162m), thenpermanentstop; road200m')
scene.MOTION=MOTION
base_materialize=scene.materialize

def materialize(folder,item):
    path=base_materialize(folder,item);cfg=json.loads(path.read_text(encoding='utf-8'))
    origin=cfg['targets'][0]['path']['origin_cm'];travel=100*SPEED*STOP
    cfg['camera'].update(pitch_limits_deg=[-85.,-15.],fov_min_deg=8.)
    cfg['occluders']=[dict(kind='road_surface',location_cm=[origin[0],origin[1]+travel/2,.15],
        size_cm=[900.,20000.,.2],color_rgb=[.13,.14,.14])]
    for side in (-1,1):
        for i in range(-3,int(math.ceil(travel/500))+4):
            cfg['occluders'].append(dict(kind='road_paint',location_cm=[origin[0]+side*350.,origin[1]+i*500.,.5],
                size_cm=[12.,250.,.2],color_rgb=[.78,.78,.70]))
    cfg.update(name='user_uniform9ms_then_stop_200mroad',motion_schedule=MOTION)
    cfg['scope_note']='Userrequested9m/s, nativeactualmotionnotvideoacceleration. Same initialtinyview and162mline,200mroad. Pitchupperlimit-15andFOVmin8extendmechanicalreach/lensrangeforfarendpoint. PolicyreceivesnoGT,pathvelocityorschedule.'
    path.write_text(json.dumps(cfg,indent=2),encoding='utf-8');return path

def write(path,value):
    if path.name=='configuration.json':
        value.update(requested_vehicle_speed_m_s=9.,road_length_m=200.,vehicle_travel_m=162.,
            reach_extension='Camera pitchlimit-85..-15 andFOV8..66 forlongroad; nativepathmovesactual9m/s. No videospeedchange.')
    return latency.write(path,value)

if __name__=='__main__':
    runner.create_agent=hierarchy.create_agent;runner.resume=hierarchy.resume;runner.SpeedEnvironment=latency.Observation
    runner.LOW=LOW;runner.HIGH=HIGH;runner.SECONDS_PER_RAW=SECONDS
    runner.layout=scene.layout;runner.materialize=materialize
    runner.AtomicTransport=hierarchy.Transport;runner.DisplayTransport=hierarchy.DisplayTransport
    runner.write=write;runner.main()
