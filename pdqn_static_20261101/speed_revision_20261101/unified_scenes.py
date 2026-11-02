"""Training layouts only. No scene schedule, target pose or speed in policy."""
import speed_bootstrap
import json,math,random
from scenes import configuration,INIT,ground_position

def road(cfg,origin,direction,length=22000.):
    dx,dy=direction;cx=origin[0]+dx*7000;cy=origin[1]+dy*7000
    yaw=math.degrees(math.atan2(dy,dx))-90.
    cfg['occluders']=[dict(kind='road_surface',location_cm=[cx,cy,.15],size_cm=[900.,length,.2],rotation_deg=[0.,yaw,0.],color_rgb=[.13,.14,.14])]
    for side in (-1,1):
        for i in range(-5,38):
            s=i*500.;cfg['occluders'].append(dict(kind='road_paint',location_cm=[origin[0]+dx*s-dy*side*350,origin[1]+dy*s+dx*side*350,.5],size_cm=[12.,250.,.2],rotation_deg=[0.,yaw,0.],color_rgb=[.78,.78,.70]))

def scene(folder,index):
    r=random.Random(40100000+index);group=index%3;serial=index//3;near=group==1
    pitch=r.uniform(-54.,-38.);height=4500. if group==0 and serial%2 else 6500.
    fov=r.uniform(17.,27.) if near else r.uniform(42.,66.) if group==2 else r.choice([20.,29.,43.,62.])
    sx,sy=[(-1,-1),(1,-1),(-1,1),(1,1)][serial%4]
    radius=r.uniform(.0,.04) if near else r.uniform(.2,.41)
    center=[.5+sx*radius,.5+sy*r.uniform(.06,radius)] if not near else [.5+r.uniform(-.035,.035),.5+r.uniform(-.035,.035)]
    item=dict(split='train',index=index,seed=40100000+index,group=['static','near_dynamic','far_dynamic'][group],quadrant=serial%4,layer=0,scale_group=0,planned_center=center,fov_deg=fov)
    cfg=configuration(item,(180.,100.,100.));init=dict(INIT,camera_location_cm=[-2200.,0.,height],nominal_camera_pitch_deg=pitch)
    cfg['camera'].update(location_cm=init['camera_location_cm'],rotation_deg=[pitch,0.,0.],pitch_limits_deg=[-85.,-10.],yaw_limits_deg=[-85.,85.],fov_min_deg=6.)
    origin=ground_position(*center,fov,init)
    angle=r.choice([0.,45.,90.,135.,-45.,-90.]);direction=[math.cos(math.radians(angle)),math.sin(math.radians(angle))]
    speed=0. if group==0 else [1.5,4.5,9.][serial%3]
    profiles=[[[0.,1.]],[[0.,1.],[5.,0.]],[[0.,.3],[2.,.6],[4.,1.],[7.,.5],[9.,0.]],[[0.,1.],[3.,0.],[5.,.5],[7.,1.],[10.,0.]]]
    profile=profiles[(serial//3)%4] if group else [[0.,0.]]
    # Backward oblique routes are shorter to stay in the camera's physical reach.
    if direction[0]<0 and speed*100*12*abs(direction[0])>.55*(origin[0]+2200):
        direction=[abs(direction[0]),direction[1]]
    cfg['targets'][0]['path']=dict(type='line',origin_cm=origin,velocity_cm_s=[100*speed*direction[0],100*speed*direction[1],0.],speed_profile=profile)
    road(cfg,origin,direction)
    cfg.update(name='unified_mixed_training',scope_note='Scene-only groundtruth, never passed to policy/reward. One policy for all groups.',experiment_coverage=item)
    item.update(speed_m_s=speed,direction=direction,speed_profile=profile)
    folder.mkdir(parents=True,exist_ok=True);path=folder/'scene.json';path.write_text(json.dumps(cfg,indent=2),encoding='utf-8')
    return path,item
