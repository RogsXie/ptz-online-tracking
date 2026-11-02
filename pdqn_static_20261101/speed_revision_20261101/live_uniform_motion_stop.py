"""Fresh real UE uniform-motion then permanent-stop development scene.

Motion is executed by the native analytic line path with speed_profile;
the scene continues during perception and camera idle time. Motion schedule
and world coordinates are never supplied to the frozen P-DQN controller.
"""
import speed_bootstrap
import json
import live_tiny_optics_precision_v2 as precision
import live_fast_tiny_corner as fast
from scenes import configuration,INIT,ground_position

runner=precision.runner
MOTION=dict(speed_m_s=1.5,velocity_cm_s=[0.,150.,0.],start_sim_s=0.,stop_sim_s=18.,
    speed_profile=[[0.,1.],[18.,0.]],loop=False,description='Uniform straight travel18s, then stays at the endpoint')

def layout(split,index):
    return dict(split='development',index=0,seed=39500000,quadrant=0,layer=4,scale_group=0,
        planned_center=[.075,.09],fov_deg=66.,camera_height_cm=6500.,
        role='User requested uniform motion then stop; fresh dynamic demonstration')

def materialize(folder,item):
    folder.mkdir(parents=True,exist_ok=True)
    cfg=configuration(item,fast.RATES)
    init=dict(INIT,camera_location_cm=[-2200.,0.,item['camera_height_cm']])
    origin=ground_position(*item['planned_center'],item['fov_deg'],init)
    cfg.update(name='user_uniform_motion_then_stop_v1',task='rgb_tracking_uniform_motion_then_stop',
        features=['single_vehicle','uniform_motion','stop_and_remain','fixed_camera_installation','active_pan_tilt_zoom'])
    cfg['camera'].update(location_cm=init['camera_location_cm'],pitch_limits_deg=[-85.,-30.])
    cfg['targets'][0]['path']=dict(type='line',origin_cm=origin,velocity_cm_s=MOTION['velocity_cm_s'],
        speed_profile=MOTION['speed_profile'],phase_s=0.,random_phase=False)
    # Flat road and paint are visual world references, below the vehicle body.
    # They never move or supply controller inputs beyond the captured RGB.
    road_center_y=origin[1]+1350.
    cfg['occluders']=[dict(kind='road_surface',location_cm=[origin[0],road_center_y,.15],
        size_cm=[900.,6500.,.2],color_rgb=[.13,.14,.14])]
    for side in (-1,1):
        for i in range(-1,9):
            cfg['occluders'].append(dict(kind='road_paint',location_cm=[origin[0]+side*350.,origin[1]+i*500.,.5],
                size_cm=[12.,250.,.2],color_rgb=[.78,.78,.70]))
    cfg['motion_schedule']=MOTION
    cfg['scope_note']='Native line path + analytically integrated speed profile.1.5m/s for18s then0. Scene clock remains real-time through Python inference. Frozen staticallytrained policy used as dynamic development demonstration, no GT-follow or trajectory replay.'
    path=folder/'scene-uniform-motion-stop.json';path.write_text(json.dumps(cfg,indent=2),encoding='utf-8')
    return path

def write(path,value):
    if path.name=='configuration.json':
        value.update(execution_profile='uniform_motion_stop_with_optics_precision',rates=list(fast.RATES),motion=MOTION,
            execution_calibration='Unchanged source directions/raw durations; angulargain=min(1,(tan(FOV/2)/tan(33deg))**2),zoomgain=min(1,FOV/66). Same as preceding stable static demonstration.',
            actuator_inputs='CameraFOV for actuatorcalibration only; no motion schedule,worldtargetcoordinates,GTorvelocity in policy.',
            qualification='Fresh dynamic development example using frozen statictrained checkpoint; no newtraining or fullvalidation',
            scope='Same initialsmallupperleftview; nativeuniformvehicle motion thenpermanentstop; fixedinstallation and exclusive7policyactions')
    return fast.original_write(path,value)

if __name__=='__main__':
    runner.layout=layout
    runner.materialize=materialize
    runner.resume=fast.original_resume
    runner.AtomicTransport=precision.OpticsTransport
    runner.DisplayTransport=precision.OpticsDisplayTransport
    runner.SECONDS_PER_RAW=(.1,.4/100.,.4/100.,.8/180.,.8/180.,1./100.,1./100.)
    runner.write=write
    runner.main()
