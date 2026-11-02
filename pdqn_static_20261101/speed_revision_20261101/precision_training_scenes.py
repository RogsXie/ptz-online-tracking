"""Deterministic, disjoint procedural layouts. Geometry is scene-only metadata."""
import speed_bootstrap
import copy,json,math,random
from scenes import configuration,INIT,ground_position

FAMILIES=('fine_static','slow_motion','fast_motion','acceleration','turns','inclines','occlusion','tiny_acquisition')
BASES={'warmup':51000000,'train':52000000,'development':53000000,'check':54000000}


def scene(folder,index,split='train'):
    seed=BASES[split]+index;r=random.Random(seed);family=FAMILIES[index%8]
    pitch=r.uniform(-57.,-39.);height=r.choice([4500.,6500.,8500.])
    fov=r.uniform(15.,25.) if family=='fine_static' else r.uniform(48.,66.) if family=='tiny_acquisition' else r.uniform(22.,58.)
    radius=.045 if family=='fine_static' else .32 if family=='tiny_acquisition' else .18
    center=[.5+r.uniform(-radius,radius),.5+r.uniform(-radius*.7,radius*.7)]
    item=dict(split=split,index=index,seed=seed,family=family,quadrant=index%4,layer=0,scale_group=0,
        planned_center=center,fov_deg=fov)
    cfg=configuration(item,(180.,100.,100.));init=dict(INIT,camera_location_cm=[-2200.,0.,height],nominal_camera_pitch_deg=pitch)
    cfg['camera'].update(location_cm=init['camera_location_cm'],rotation_deg=[pitch,0.,0.],
        yaw_limits_deg=[-85.,85.],pitch_limits_deg=[-85.,-5.],fov_min_deg=6.)
    origin=ground_position(*center,fov,init)
    angle=r.choice([-45.,0.,30.,60.,90.]);d=[math.cos(math.radians(angle)),math.sin(math.radians(angle))]
    speed=0. if family=='fine_static' else r.uniform(1.,6.) if family=='slow_motion' else r.uniform(10.,36.)
    # Include a short RGB-observable acceleration phase before sustained fast
    # motion; the tracker starts without an angular-velocity estimate.
    profile=[[0.,.05],[.4,.3],[1.4,1.]] if speed>10. else [[0.,1.]]
    if family=='acceleration':profile=[[0.,.15],[1.8,1.],[4.,.3],[6.,.9],[9.,.05],[11.,.7]]
    if family=='slow_motion':profile=[[0.,1.],[5.,0.],[7.,.7]]
    if family=='tiny_acquisition':speed=r.uniform(3.,18.)
    length=max(35000.,speed*100*26)
    path=dict(type='line',origin_cm=origin,velocity_cm_s=[speed*100*d[0],speed*100*d[1],0.],
        speed_profile=profile,phase_s=0.,random_phase=False)
    points=[origin,[origin[0]+d[0]*length,origin[1]+d[1]*length,origin[2]]]
    if family in ('turns','inclines'):
        speed=r.uniform(12.,34.);run=speed*100*r.uniform(1.8,3.0)
        if family=='turns':
            radius_cm=r.uniform(500.,1400.);sign=r.choice([-1.,1.]);theta=math.atan2(d[1],d[0])
            start=[origin[0]+d[0]*run,origin[1]+d[1]*run,origin[2]]
            points=[origin,start]
            for k in range(1,13):
                a=(math.pi/2)*k/12
                forward=radius_cm*math.sin(a);side=sign*radius_cm*(1-math.cos(a))
                points.append([start[0]+d[0]*forward-d[1]*side,start[1]+d[1]*forward+d[0]*side,origin[2]])
            end=points[-1];points.append([end[0]-sign*d[1]*length,end[1]+sign*d[0]*length,end[2]])
        else:
            rise=r.uniform(400.,1100.);ramp=r.uniform(6500.,11000.)
            points=[origin,[origin[0]+d[0]*run,origin[1]+d[1]*run,origin[2]],
                [origin[0]+d[0]*(run+ramp),origin[1]+d[1]*(run+ramp),origin[2]+rise],
                [origin[0]+d[0]*length,origin[1]+d[1]*length,origin[2]+rise]]
        path=dict(type='waypoints',origin_cm=origin,waypoints_cm=points,speed_cm_s=speed*100,
            speed_profile=[[0.,.05],[.4,.3],[1.4,1.]],phase_s=0.,random_phase=False,follow_elevation=True,surface_z_offset_cm=8.)
    target=cfg['targets'][0];target['path']=path
    target['scale']=r.uniform(.8,1.15);target['color_rgb']=r.choice([[.75,.025,.012],[.025,.12,.7],[.45,.025,.65]])
    # Minimal real 3D road per segment, including visible ascending surfaces.
    cfg['occluders']=[]
    for p,q in zip(points,points[1:]):
        dx,dy,dz=[q[i]-p[i] for i in range(3)];flat=math.hypot(dx,dy)
        cfg['occluders'].append(dict(kind='road_surface',location_cm=[(p[0]+q[0])/2,(p[1]+q[1])/2,(p[2]+q[2])/2-8.],
            size_cm=[math.sqrt(flat*flat+dz*dz)+60.,1200.,6.],
            rotation_deg=[math.degrees(math.atan2(dz,flat)),math.degrees(math.atan2(dy,dx)),0.],color_rgb=[.13,.14,.14]))
    if family=='occlusion':
        # Crossing opaque truck plus lookalike following car; no control oracle.
        truck=copy.deepcopy(target);truck.update(id='truck_occluder',vehicle_kind='box_truck',scale=1.,color_rgb=[.65,.65,.62])
        cross=[origin[0]+d[0]*speed*100*3.,origin[1]+d[1]*speed*100*3.,8.]
        side=[-d[1],d[0]];v=r.uniform(700.,1200.)
        truck['path']=dict(type='line',origin_cm=[cross[0]-side[0]*v*3.,cross[1]-side[1]*v*3.,8.],
            velocity_cm_s=[side[0]*v,side[1]*v,0.],speed_profile=[[0.,1.]])
        follower=copy.deepcopy(target);follower['id']='car_same_colour_follower'
        follower['path']['origin_cm']=[origin[0]-d[0]*1200.,origin[1]-d[1]*1200.,origin[2]]
        cfg['targets'].extend([truck,follower])
    item.update(speed_m_s=speed,initial_fov_deg=fov,initial_camera_height_cm=height)
    cfg.update(name='fresh_precision_'+family,experiment_coverage=item,
        scope_note='Procedural training layout. No target coordinates, route, ID, speed or family enter policy, reward or runtime control.')
    folder.mkdir(parents=True,exist_ok=True);p=folder/'scene.json';p.write_text(json.dumps(cfg,indent=2),encoding='utf-8')
    return p,item
