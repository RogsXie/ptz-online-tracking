"""Post-run speed/continuity/parallel-lane scoring; no control or repainting."""
from pathlib import Path
import json,math
import numpy as np
from audit_continuous_online import read,summarize

ROOT=Path(__file__).resolve().parent
FOLDER=ROOT/'unified_motion_20261101/user-nonstop-viaduct30-v1-video'

def main():
    scene=json.loads((FOLDER/'scene.json').read_text())
    cars=[c for c in scene['targets'] if c['id'] in ['car_0','car_red_close_follower','car_purple_following']]
    assert all(all(multiplier>0 for _,multiplier in car['path']['speed_profile']) for car in cars)
    gt=read(FOLDER/'offline-diagnostics.jsonl');speed=[];cruise=[];parallel=[];parallel_speeds=[]
    for a,b in zip(gt,gt[1:]):
        t,u=a['sim_time'],b['sim_time']
        if u<=t:continue
        x={v['id']:v for v in a['targets']};y={v['id']:v for v in b['targets']}
        v=math.dist(x['car_0']['location_cm'],y['car_0']['location_cm'])/(u-t)/100.
        speed.append(v)
        if 12.15<t<u<27.9:cruise.append(v)
        if 24.05<t<u<27.9:
            p=[x[k]['location_cm'] for k in ['car_0','car_red_close_follower','car_purple_following']]
            parallel.append(max(z[0] for z in p)-min(z[0] for z in p))
            assert [z[1] for z in p]==[600.,1000.,1400.]
            for key in ['car_0','car_red_close_follower','car_purple_following']:
                parallel_speeds.append(math.dist(x[key]['location_cm'],y[key]['location_cm'])/(u-t)/100.)
    assert min(speed)>2.9
    assert abs(float(np.median(cruise))-30.)<.02 and min(cruise)>25.
    assert parallel and max(parallel)<.01 and abs(float(np.median(parallel_speeds))-30.)<.02
    result=dict(no_scheduled_stops_or_waits=True,minimum_actual_lead_speed_m_s=min(speed),
        actual_cruise_speed_median_m_s=float(np.median(cruise)),
        actual_parallel_speed_median_m_s=float(np.median(parallel_speeds)),parallel_x_spread_max_cm=max(parallel),
        lead_acceleration=scene['motion_schedule']['car_acceleration']['car_0'],
        follower_catchup_speeds_m_s={c['id']:scene['motion_schedule']['car_acceleration'][c['id']]['catchup_speed_m_s'] for c in cars[1:]},
        bridge=scene['motion_schedule']['bridge'],parallel_alignment_sim_s=24.,
        quality_and_online_proof=summarize(FOLDER),
        scope='One28s actual live scene. Native target poses used only after recording for speed/identity diagnostics, never live control or drawing.')
    (FOLDER/'nonstop-30mps-result.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
