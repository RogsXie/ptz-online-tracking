"""Cross intersection, native rounded90-degree turn and real30fps capture."""
import speed_bootstrap
import json,math
from types import SimpleNamespace
import live_async_variable_speed as runner
from scenes import configuration

CX,CY=6000.,1000.
RADIUS=500.
SPEED_PROFILE=[[0.,.5],[4.,1.],[7.5,1./3.],[12.,2./3.],[15.,1.],[23.,0.]]

def materialize(folder,item):
    cfg=configuration(item,(180.,100.,100.))
    cfg['camera'].update(location_cm=[-2200.,-1800.,6500.],rotation_deg=[-42.,10.,0.],
        pitch_limits_deg=[-85.,-15.],yaw_limits_deg=[-65.,65.],fov_deg=66.,fov_min_deg=6.)
    start=[CX+300.,CY-6000.,8.]
    center=[CX+800.,CY-800.]
    points=[start]
    for i in range(37):
        a=math.pi-(math.pi/2)*i/36
        points.append([center[0]+RADIUS*math.cos(a),center[1]+RADIUS*math.sin(a),8.])
    points.append([CX+10800.,CY-300.,8.])
    length=sum(math.dist(a,b) for a,b in zip(points,points[1:]))/100.
    cfg['targets'][0]['path']=dict(type='waypoints',origin_cm=start,waypoints_cm=points,speed_cm_s=900.,
        speed_profile=SPEED_PROFILE,phase_s=0.,random_phase=False)
    parts=[]
    def rect(kind,x,y,sx,sy,color,z=.5):
        parts.append(dict(kind=kind,location_cm=[x,y,z],size_cm=[sx,sy,.2],color_rgb=color))
    asphalt=[.13,.14,.14];white=[.84,.84,.78];yellow=[.92,.68,.06]
    rect('road_surface',CX,CY,22000.,1200.,asphalt,.15)
    for sign in (-1,1):rect('road_surface',CX,CY+sign*5800.,1200.,10400.,asphalt,.15)
    # Both center markings and outer edges stop at the intersection.
    for sign in (-1,1):
        for side in (-1,1):
            rect('road_edge',CX+sign*5800.,CY+side*550.,10400.,12.,white)
            rect('road_edge',CX+side*550.,CY+sign*5800.,12.,10400.,white)
        for offset in (-8.,8.):
            rect('road_centerline',CX+sign*6050.,CY+offset,9900.,7.,yellow)
            rect('road_centerline',CX+offset,CY+sign*6050.,7.,9900.,yellow)
        for offset in range(-490,491,140):
            rect('crosswalk',CX+offset,CY+sign*730.,80.,230.,white)
            rect('crosswalk',CX+sign*730.,CY+offset,230.,80.,white)
        # Stop line only spans the approaching lane on each arm.
        rect('stop_line',CX-sign*300.,CY+sign*1000.,500.,36.,white)
        rect('stop_line',CX+sign*1000.,CY+sign*300.,36.,500.,white)
    for sx in (-1,1):
        for sy in (-1,1):rect('sidewalk',CX+sx*1575.,CY+sy*1575.,1850.,1850.,[.36,.37,.35],.13)
    cfg['occluders']=parts
    # Geometry describes the scene only, never a policy input or motor action.
    from unified_perception import project
    c=cfg['camera'];v=[start[i]-c['location_cm'][i] for i in range(3)]
    a=[math.degrees(math.atan2(v[1],v[0])),math.degrees(math.atan2(v[2],math.hypot(v[0],v[1])))]
    item.update(role='User requestedcrossroadturn',planned_center=list(project(a,dict(rotation=c['rotation_deg'],fov=66.))),
        fov_deg=66.,camera_height_cm=6500.,seed=40601001)
    cfg['experiment_coverage']=item
    runner.demo.MOTION=dict(speed_m_s=9.,speed_profile=SPEED_PROFILE,scenario='crossroad_rounded90degree_turn',
        intersection_center_cm=[CX,CY,0.],road_width_m=12.,road_arm_span_m=220.,turn_radius_m=5.,turn_direction='incoming+Y to outgoing+X',
        approach_length_m=52.,turn_arc_length_m=sum(math.dist(a,b) for a,b in zip(points[1:-1],points[2:-1]))/100.,
        route_forward_length_m=length,travel_before_stop_m=153.,stop_sim_s=23.,turn_speed_m_s=3.,
        description='4.5then9m/s approach;3m/s rounded90degturn;6then9m/s exit; stop23s. Native waypoint motion, no pose replay or video speedup.')
    assert 153.<length
    cfg.update(name='user_crossroad_native_rounded_turn',task='rgb_tracking_crossroad_turn',motion_schedule=runner.demo.MOTION,
        scope_note='Native predeclared scene path only; policy sees RGB/owncamera, never waypoints, targetposition, speed or turn timing. Frozen current model. No turning-training added. Flat road/markings below vehicle.')
    folder.mkdir(parents=True,exist_ok=True);p=folder/'scene.json';p.write_text(json.dumps(cfg,indent=2),encoding='utf-8');return p

if __name__=='__main__':
    runner.scenario=SimpleNamespace(materialize=materialize)
    runner.main()
