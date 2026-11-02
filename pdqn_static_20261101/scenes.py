"""Offline layout only. No coordinates or camera intrinsics reach the policy."""
import json
import math
import random
from copy import deepcopy
from bootstrap import REPRO
from control_coverage_v8.scenes import ground_position

BASES = dict(calibration=31000000, train=32000000, development=33000000,
             validation=34000000, final_test=35000000)
SIGNS = [(-1,-1),(1,-1),(-1,1),(1,1)]
INIT = dict(camera_location_cm=[-2200.,0.,4500.], nominal_camera_pitch_deg=-60.,
            layout_reference_height_cm=90.)

def layout(split, index):
    seed = BASES[split]+index
    r = random.Random(seed)
    cell = index % 80 * 37 % 80
    q, layer, scale = cell%4, cell//4%5, cell//20
    ranges = [(56.,63.),(40.,44.),(27.,30.),(19.,21.)]
    if split in ('validation','final_test'):
        ranges = [(53.,55.),(37.,39.),(25.,26.),(22.,23.)]
    fov = r.uniform(*ranges[scale])
    maximum = [.38,.33,.275,.20][scale]
    edges = [(.001,.006),(.02,.07),(.08,.16),(.17,.25),(.26,.38)]
    low,high = edges[layer]
    low,high = low*maximum/.38,high*maximum/.38
    if layer == 0:
        ox,oy = r.uniform(low,high),r.uniform(low,high)
    else:
        radius = r.uniform(low,high)
        other = r.uniform(.015,radius)
        ox,oy = (radius,other) if r.random()<.5 else (other,radius)
    sx,sy = SIGNS[q]
    return dict(split=split,index=index,seed=seed,quadrant=q,layer=layer,scale_group=scale,
                planned_center=[.5+sx*ox,.5+sy*oy],fov_deg=fov)

def configuration(item, rates=(4.,2.,5.)):
    cfg = deepcopy(json.loads((REPRO/'focused_scenarios/focus_far.json').read_text(encoding='utf-8-sig')))
    cfg.update(name='stationary_author_pdqn_20261101',split=item['split'],task='stationary_rgb_center_and_active_zoom')
    cfg['camera'].update(location_cm=INIT['camera_location_cm'],rotation_deg=[-60.,0.,0.],
        rotation_jitter_deg=[0.,0.,0.],yaw_limits_deg=[-65.,65.],pitch_limits_deg=[-80.,-40.],
        pan_deg_s=rates[0],tilt_deg_s=rates[1],zoom_fov_deg_s=rates[2],
        fov_deg=item['fov_deg'],fov_min_deg=14.,fov_max_deg=66.,legacy_target_center_reset=False)
    target=deepcopy(cfg['targets'][0])
    target.update(id='car_0',scale=1.,color_rgb=[.75,.025,.012])
    target['path'] = dict(type='static',origin_cm=ground_position(*item['planned_center'],item['fov_deg'],INIT))
    cfg['targets']=[target]
    cfg['randomization']=dict(position_jitter_cm=0.,speed_scale_range=[1.,1.])
    cfg['occluders']=[]
    cfg['experiment_coverage']=item
    return cfg

def materialize(folder,item,rates=(4.,2.,5.)):
    folder.mkdir(parents=True,exist_ok=True)
    path=folder/f"scene-{item['split']}-{item['index']:05d}.json"
    value=json.dumps(configuration(item,rates),indent=2)
    if path.exists() and path.read_text()!=value:
        raise FileExistsError(path)
    path.write_text(value,encoding='utf-8')
    return path

