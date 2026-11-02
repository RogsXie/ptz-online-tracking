"""Long elevated road after the crossing; uninterrupted native30m/s driving."""
import speed_bootstrap
import copy,json,math,shutil
from pathlib import Path
import live_crossroad_robust_tracker as previous
import live_unified_motion as demo

RAMP_START=8000.
DECK_START=16000.
DECK_END=58000.
ROAD_END=66000.
HEIGHT=800.
CY=1000.
ALIGNMENT_TIME=24.
CRUISE_SPEED=30.
VIDEO_SECONDS=28.

def distance_at(profile,speed,t):
    total=0.
    for i,(start,multiplier) in enumerate(profile):
        end=profile[i+1][0] if i+1<len(profile) else t
        if start<t:total+=max(0.,min(t,end)-start)*multiplier*speed
    return total

def time_at_distance(profile,speed,distance):
    for i,(start,multiplier) in enumerate(profile):
        end=profile[i+1][0] if i+1<len(profile) else math.inf
        velocity=speed*multiplier
        if velocity>0 and distance<=velocity*(end-start):return start+distance/velocity
        distance-=velocity*(end-start)
    raise ValueError('Original motion never reaches the turn exit')

def height(x):
    if x<RAMP_START:return 0.
    if x<DECK_START:return HEIGHT*(x-RAMP_START)/(DECK_START-RAMP_START)
    if x<=DECK_END:return HEIGHT
    return max(0.,HEIGHT*(ROAD_END-x)/(ROAD_END-DECK_END))

def materialize(folder,item):
    path=previous.materialize(folder,item)
    cfg=json.loads(path.read_text(encoding='utf-8'))
    # A fixed PTZ must tilt closer to the horizon to see the far elevated deck.
    cfg['camera']['pitch_limits_deg']=[-85.,-5.]
    red=cfg['targets'][0]
    # Keep the original approach and rounded turn. Replace only the long exit
    # straight with a12m connector followed immediately by the up-ramp.
    lane_centers={'car_0':600.,'car_red_close_follower':1000.,'car_purple_following':1400.}
    deck_distances={};acceleration={}
    for car in cfg['targets']:
        if car['id'] not in {'car_0','car_red_close_follower','car_purple_following'}:continue
        route=car['path'];old_speed=route['speed_cm_s']
        turn_points=route['waypoints_cm'][:-1]
        turn_distance=sum(math.dist(a,b) for a,b in zip(turn_points,turn_points[1:]))
        turn_exit=time_at_distance(route['speed_profile'],old_speed,turn_distance)
        profile=[[t,multiplier*old_speed/(100.*CRUISE_SPEED)] for t,multiplier in route['speed_profile'] if t<turn_exit-1e-8]
        profile.extend([[turn_exit,.4],[turn_exit+.5,.7],[turn_exit+1.1,1.]])
        acceleration[car['id']]=dict(turn_exit_sim_s=turn_exit,acceleration_end_sim_s=turn_exit+1.1,
            acceleration_speed_steps_m_s=[.4*CRUISE_SPEED,.7*CRUISE_SPEED,CRUISE_SPEED],catchup_speed_m_s=CRUISE_SPEED)
        extension=[]
        for x in range(int(RAMP_START),int(ROAD_END)+1,500):
            fraction=min(1.,max(0.,(x-RAMP_START)/(DECK_START-RAMP_START)))
            blend=fraction*fraction*(3.-2.*fraction)
            y=700.+(lane_centers[car['id']]-700.)*blend
            extension.append([x,y,height(x)+8.])
        route.update(waypoints_cm=[*route['waypoints_cm'][:-1],*copy.deepcopy(extension)],
            speed_cm_s=100.*CRUISE_SPEED,speed_profile=profile,
            follow_elevation=True,surface_z_offset_cm=8.)
        points=route['waypoints_cm'];index=next(i for i,p in enumerate(points) if p[0]==DECK_START)
        deck_distances[car['id']]=sum(math.dist(a,b) for a,b in zip(points[:index],points[1:index+1]))/100.
    # The leading target never slows or stops after turn exit. Followers use
    # brief catch-up speeds on their own lanes, then match cruise side-by-side.
    alignment_time=ALIGNMENT_TIME
    leader_distance=distance_at(red['path']['speed_profile'],100.*CRUISE_SPEED,alignment_time)/100.
    common_deck_distance=leader_distance-deck_distances[red['id']]
    for car in cfg['targets']:
        if car['id'] not in lane_centers or car is red:continue
        needed=deck_distances[car['id']]+common_deck_distance
        profile=car['path']['speed_profile'];start=profile[-1][0]
        completed=distance_at(profile,100.*CRUISE_SPEED,start)/100.
        speed=(needed-completed)/(alignment_time-start)
        assert CRUISE_SPEED<speed<1.5*CRUISE_SPEED
        profile[-1][1]=speed/CRUISE_SPEED;profile.append([alignment_time,1.])
        acceleration[car['id']]['catchup_speed_m_s']=speed
    points=red['path']['waypoints_cm'];profile=red['path']['speed_profile']
    parts=cfg['occluders']
    for part in parts:
        if part['kind'] not in {'road_surface','road_edge','road_centerline'}:continue
        sx,sy,_=part['size_cm'];x=part['location_cm'][0]
        if sx<=sy or x+sx/2<=RAMP_START:continue
        left=x-sx/2;length=RAMP_START-left
        assert length>0
        part['location_cm'][0]=(left+RAMP_START)/2;part['size_cm'][0]=length
    asphalt=[.13,.14,.14];white=[.84,.84,.78];yellow=[.92,.68,.06]
    concrete=[.46,.48,.48];steel=[.66,.70,.72]
    def box(kind,location,size,color,rotation=0.):
        parts.append(dict(kind=kind,location_cm=location,size_cm=size,color_rgb=color,
            rotation_deg=[rotation,0.,0.],instance_geometry=True,
            cast_shadow=kind in {'bridge_structure','bridge_crossbeam','bridge_pier'}))
    # Extend terrain and ground-level roads beneath the far bridge.
    ground=cfg.get('environment',{}).get('ground_color_rgb',[.26,.30,.26])
    box('terrain_extension',[46000.,1000.,-25.],[40000.,30000.,50.],ground)
    box('underpass_road',[30500.,1000.,.15],[1200.,18000.,.2],asphalt)
    for side in [-1,1]:
        box('underpass_edge',[30500.+side*550.,1000.,.5],[12.,18000.,.4],white)
        box('underpass_center',[30500.+side*8.,1000.,.5],[7.,18000.,.4],yellow)
    def strip(kind,x1,x2,y,width,thickness,top,color):
        z1,z2=height(x1),height(x2);angle=math.atan2(z2-z1,x2-x1)
        offset=top-thickness/2.
        box(kind,[(x1+x2)/2.-offset*math.sin(angle),CY+y,(z1+z2)/2.+offset*math.cos(angle)],
            [math.hypot(x2-x1,z2-z1)+.5,width,thickness],color,math.degrees(angle))
    # Each straight grade is one exact coplanar cuboid. Subdividing these
    # planes adds draw calls and repeated scene metadata without changing shape.
    stations=[RAMP_START,DECK_START,DECK_END,ROAD_END]
    for x1,x2 in zip(stations,stations[1:]):
        strip('bridge_structure',x1,x2,0.,1320.,70.,-6.,concrete)
        strip('bridge_asphalt',x1,x2,0.,1200.,6.,0.,asphalt)
        for side in [-1,1]:
            strip('bridge_edge',x1,x2,side*550.,12.,.5,.7,white)
            strip('bridge_curb',x1,x2,side*625.,28.,28.,28.,concrete)
            strip('bridge_guardrail',x1,x2,side*625.,15.,13.,90.,steel)
    for x in range(int(RAMP_START),int(ROAD_END)+1,1000):
        for side in [-1,1]:box('bridge_rail_post',[x,CY+side*625.,height(x)+55.],[10.,12.,70.],steel)
    for x in range(int(RAMP_START),int(ROAD_END)-300,1000):
        for offset in [-200.,200.]:strip('bridge_lane_dash',x,x+300.,offset,9.,.5,.7,white)
    for x in range(int(RAMP_START+4000),int(ROAD_END-4000)+1,3000):
        top=height(x)-77.
        if top<150.:continue
        box('bridge_crossbeam',[x,CY,top-35.],[150.,1180.,70.],concrete)
        for side in [-1,1]:box('bridge_pier',[x,CY+side*440.,(top-70.)/2.],[120.,120.,top-70.],concrete)
    length=sum(math.dist(a,b) for a,b in zip(points,points[1:]))/100.
    travel=distance_at(profile,100.*CRUISE_SPEED,VIDEO_SECONDS)/100.
    assert travel<length
    demo.MOTION.update(scenario='nonstop_acceleration_to_parallel_viaduct_30mps',speed_m_s=CRUISE_SPEED,speed_profile=profile,
        original_approach_and_turn_preserved=True,turn_exit_connector_m=12.,
        nonstop_bridge_entry=True,stop_sim_s=None,car_acceleration=acceleration,
        travel_by_28s_m=travel,route_forward_length_m=length,recommended_video_seconds=VIDEO_SECONDS,
        parallel_driving=dict(lane_centers_y_cm=lane_centers,lane_change='Smooth lateral ramp transition, complete by deck start.',
            lane_width_m=4.,alignment_sim_s=alignment_time,alignment_x_cm=DECK_START+100*common_deck_distance,
            cruise_speed_m_s=CRUISE_SPEED,leader_never_slows_for_alignment=True,
            description='Accelerate immediately after each turn exit. Followers briefly travel faster to catch up, then all3drive30m/s side-by-side from24s. No stops or scripted waits.'),
        bridge=dict(up_ramp_length_m=80.,deck_length_m=(DECK_END-DECK_START)/100.,down_ramp_length_m=80.,height_m=8.,
            road_width_m=12.,up_ramp_grade_percent=10.,ramp_start_x_cm=RAMP_START,deck_start_x_cm=DECK_START,
            deck_end_x_cm=DECK_END,road_end_x_cm=ROAD_END,underpass_x_cm=30500.,driving='onto bridge, not beneath bridge'),
        description='Original intersection,approach and rounded turn retained. Accelerate at turn exit,1.1s to30m/s, continue directly up the80m ramp after12m connector. No stops; parallel3lane30m/s from24s. Followers briefly travel faster to catch up; exact speeds in car_acceleration. Native3D travel, no pixel or pose replay.')
    cfg.update(name='user_crossroad_nonstop_parallel_viaduct_30mps',motion_schedule=demo.MOTION,
        features=[*cfg['features'],'80m_up_ramp','420m_elevated_deck','guardrails_and_piers','native3D_vehicle_elevation','30mps_nonstop_cruise'],
        scope_note='Scene-only3D route/road geometry. Native car heights follow the ramp and deck; existing flat scenes remain unchanged. Policy inputs remain RGB and own PTZ state; no route, actor identity, target position, speed or height input.')
    path.write_text(json.dumps(cfg,indent=2),encoding='utf-8')
    shutil.copy2(Path(__file__),folder/'frozen-sources'/Path(__file__).name)
    return path
