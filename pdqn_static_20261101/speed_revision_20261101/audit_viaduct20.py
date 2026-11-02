"""Post-run geometry, speed and actual online output diagnostics only."""
from pathlib import Path
import json,math
import numpy as np
from audit_continuous_online import read,summarize

ROOT=Path(__file__).resolve().parent

def main():
    folder=ROOT/'unified_motion_20261101/user-original-then-parallel-viaduct-20mps-v4-video'
    original=ROOT/'unified_motion_20261101/user-startup-active-zoom-v1-video/scene.json'
    old=json.loads(original.read_text());new=json.loads((folder/'scene.json').read_text())
    assert new['occluders'][:len(old['occluders'])]==old['occluders']
    assert {k:v for k,v in new['camera'].items() if k!='pitch_limits_deg'}=={k:v for k,v in old['camera'].items() if k!='pitch_limits_deg'}
    for a,b in zip(old['targets'],new['targets']):
        assert a['id']==b['id']
        if a['id'] not in {'car_0','car_red_close_follower','car_purple_following'}:
            assert a==b;continue
        x,y=a['path'],b['path'];assert x['origin_cm']==y['origin_cm']
        assert x['waypoints_cm']==y['waypoints_cm'][:len(x['waypoints_cm'])]
        assert len(y['speed_profile'])==len(x['speed_profile'])+4
        for (t,s),(t2,s2) in zip(x['speed_profile'],y['speed_profile']):
            assert t==t2 and abs(s*x['speed_cm_s']-s2*y['speed_cm_s'])<1e-8
    frames=read(folder/'offline-diagnostics.jsonl')
    positions=[(g['sim_time'],g['targets'][0]['location_cm'],g['targets'][0]['rotation_deg']) for g in frames]
    speeds=[];stopped=[]
    for (t,a,_),(u,b,_) in zip(positions,positions[1:]):
        if u<=t:continue
        v=math.dist(a,b)/(u-t)/100.
        if 30.1<t<u<45.9:speeds.append(v)
        if 46.1<t<u:stopped.append(v)
    ramp=next(t for t,p,_ in positions if p[2]>10.)
    deck=next(t for t,p,_ in positions if p[0]>=25000. and p[2]>790.)
    assert abs(float(np.median(speeds))-20.)<.02
    assert max(stopped,default=0.)<.001
    assert abs(positions[-1][1][2]-796.5)<.1
    quality=summarize(folder)
    parallel=[]
    for g in frames:
        if not 38.05<g['sim_time']<45.9:continue
        cars={t['id']:t for t in g['targets']}
        selected=[cars[k]['location_cm'] for k in ['car_0','car_red_close_follower','car_purple_following']]
        parallel.append(dict(time=g['sim_time'],longitudinal_spread_cm=max(p[0] for p in selected)-min(p[0] for p in selected),
            lane_y_cm=[p[1] for p in selected],body_z_cm=[p[2] for p in selected]))
    assert parallel and max(p['longitudinal_spread_cm'] for p in parallel)<.01
    assert all(p['lane_y_cm']==[600.,1000.,1400.] for p in parallel)
    result=dict(original_crossroad_geometry_and_vehicle_route_prefixes_preserved=True,
        original_actual_speed_profiles_and_stops_preserved_until_30s=True,
        original_blue_truck_and_initial_camera_unchanged=True,
        camera_extension_only='Pitch upper limit -15deg to-5deg to view distant elevated deck; original initial pose/FOV unchanged.',
        bridge=new['motion_schedule']['bridge'],
        parallel_driving=new['motion_schedule']['parallel_driving'],
        actual_parallel_longitudinal_spread_max_cm=max(p['longitudinal_spread_cm'] for p in parallel),
        actual_parallel_lane_centers_y_cm=parallel[0]['lane_y_cm'],
        actual_cruise_speed_median_m_s=float(np.median(speeds)),actual_ramp_entry_sim_s=ramp,
        actual_deck_entry_sim_s=deck,actual_max_body_pitch_deg=max(r[0] for _,_,r in positions),
        final_original_red_location_cm=positions[-1][1],stop_after_46s_verified=True,
        quality_and_online_proof=quality,
        scope='One focused52s native live scene. Geometry/positions/rotation and speed read only after run, never control or overlay inputs.')
    (folder/'viaduct20-result.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
