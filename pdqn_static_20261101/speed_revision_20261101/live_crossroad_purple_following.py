"""Purple follows the red route; blue foreground traffic and RGB distance line."""
import speed_bootstrap
import json,copy,shutil
from pathlib import Path
import live_crossroad_blue_traffic as previous

runner=previous.runner
DELAY=8./3.
original_write=runner.write

def materialize(folder,item):
    path=previous.materialize(folder,item);cfg=json.loads(path.read_text(encoding='utf-8'))
    purple=copy.deepcopy(cfg['targets'][0]);purple.update(id='car_purple_following',color_rgb=[.48,.025,.72])
    purple_start=list(purple['path']['waypoints_cm'][0]);purple_start[1]-=1200.
    purple['path'].update(origin_cm=purple_start,waypoints_cm=[purple_start,*purple['path']['waypoints_cm']],
        speed_profile=[[0.,.5],[4.+DELAY,1.],[7.5+DELAY,1./3.],[12.+DELAY,2./3.],[15.+DELAY,1.],[22.+DELAY,1./3.],[23.+DELAY,0.]])
    cfg['targets'].append(purple)
    runner.demo.MOTION.update(scenario='crossroad_redturn_bluepass_purplefollowing',following_traffic=dict(
        description='Purple starts12m behind, follows same route about2.67slater, slows to3m/s before stopping6m behind stoppedred. Native independentwaypoint actor.',
        initial_route_gap_m=12.,nominal_headway_s=DELAY,final_route_gap_m=6.,path=purple['path']))
    cfg.update(name='user_crossroad_three_vehicles_yellow_distance',motion_schedule=runner.demo.MOTION,
        features=['three_vehicles','cross_intersection','red_turn','blue_foreground_pass','purple_following','activePTZ'],
        scope_note='Adds only independent purplefollowing actor and video-only RGBcenter-distance overlay. Existing blue remains. Same frozen image-basedred-initialized selector/policy; no GT/IDs/colorcontrolrule. No depth/metre inference for screen distance.')
    path.write_text(json.dumps(cfg,indent=2),encoding='utf-8')
    shutil.copy2(Path(__file__),folder/'frozen-sources'/Path(__file__).name)
    return path

def write(path,value):
    if path.name=='configuration.json':
        value.update(overlay_target_distance=True,distance_overlay_units='pixels',
            distance_overlay='Yellow segment from same-frame RGB-selected bboxcenter toimagecenter; numericEuclideanpixel distance. Displayonly, no GT or policy effect.')
    return original_write(path,value)

if __name__=='__main__':
    runner.cross.materialize=materialize;runner.write=write;runner.main()
