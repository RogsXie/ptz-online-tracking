"""Four quadrants x three offsets x five initial geometric magnifications."""
from copy import deepcopy
from pathlib import Path
import hashlib
import json
import math
import random
from control_curriculum.scenes import resolve as base_scene
from control_coverage_v8.scenes import ground_position

HERE = Path(__file__).parent
BASES = {'train':13100000, 'probe':13200000, 'calibration':13300000, 'validation':13400000}
CAPACITIES = {'train':1200, 'probe':60, 'calibration':60, 'validation':120}
CORNER_NAMES = ('top_left','top_right','bottom_left','bottom_right')
SIGNS = ((-1,-1),(1,-1),(-1,1),(1,1))
STAGE = 'static_zoom'


def load_protocol():
    return json.loads((HERE/'protocol.json').read_text(encoding='utf-8'))


def stage_spec(stage):
    if stage != STAGE:
        raise ValueError('v13 only covers stationary centering plus active zoom')
    return load_protocol()['stages'][0]


def layout(seed, split):
    if split not in BASES or not BASES[split] <= seed < BASES[split]+CAPACITIES[split]:
        raise ValueError('Outside declared partition; final tests remain sealed')
    index = seed-BASES[split]
    cell = index%60*17%60
    quadrant, layer, scale = cell%4, cell//4%3, cell//12
    rng = random.Random(seed)
    cfg = load_protocol()
    magnification = cfg['coverage']['initial_magnifications'][scale]
    # Calibration uses the exact representative levels. Train/probe positions
    # vary independently, while held-out validation has intermediate focal ratios.
    if split == 'validation':
        magnification *= rng.uniform(1.035,1.065)
    fov = math.degrees(2*math.atan(math.tan(math.radians(31))/magnification))
    low, high = cfg['coverage']['offset_layers'][layer]['max_axis_offset']
    if layer == 0:
        ox, oy = rng.uniform(low,high), rng.uniform(low,high)
    else:
        # Both axes occupy the selected corner, with independent jitter.
        ox, oy = rng.uniform(low,high), rng.uniform(low,high)
    sx,sy = SIGNS[quadrant]
    return dict(seed=seed, split=split, quadrant=quadrant, layer=layer, scale_group=scale,
                group=f'q{quadrant}-d{layer}-s{scale}', planned_center=[.5+sx*ox,.5+sy*oy],
                fov_deg=fov, initial_magnification=magnification, reference_fov_deg=62.,
                heldout_fov_style=split=='validation')


def cases(stage, split, count=None, offset=0, allow_final_test=False):
    stage_spec(stage)
    if split not in BASES:
        raise ValueError('Final tests remain sealed')
    count = CAPACITIES[split] if count is None else count
    if not 0<count<=CAPACITIES[split] or not 0<=offset<=CAPACITIES[split]-count:
        raise ValueError('Outside declared partition')
    plan = [dict(stage=stage,split=split,seed=BASES[split]+i,case_index=i) for i in range(offset,offset+count)]
    return plan


def probe_cases(full=False):
    plan = cases(STAGE,'probe')
    if full:
        return plan
    # Same balanced twenty seeds at both intermediate probes; one per corner x scale.
    return [c for c in plan if (lambda x:x['layer']==(x['quadrant']+x['scale_group'])%3)(layout(c['seed'],'probe'))]


def category(seed,split):
    item=layout(seed,split)
    return ('already_centered','inner_corner','outer_corner')[item['layer']],item['quadrant']


def resolve(stage,split,seed):
    stage_spec(stage)
    item=layout(seed,split); protocol=load_protocol();init=protocol['corner_initialization']
    cfg=deepcopy(base_scene('static_center','train',211000))
    cfg['targets'][0]['scale']=1.
    cfg['targets'][0]['path']['origin_cm']=ground_position(*item['planned_center'],item['fov_deg'],init)
    cfg.update(name='four_corner_model_zoom_v13',split=split,task='learned_rgb_model_planning_v13')
    cfg['camera'].update(location_cm=init['camera_location_cm'],rotation_deg=[-60.,0.,0.],
        rotation_jitter_deg=[0.,0.,0.],yaw_limits_deg=[-65.,65.],pitch_limits_deg=[-80.,-40.],
        fov_deg=item['fov_deg'],fov_min_deg=14.,fov_max_deg=66.,zoom_fov_deg_s=8.)
    cfg['control_protocol'].update(name=protocol['name'],seed=seed,stage=stage,
        protocol_sha256=hashlib.sha256((HERE/'protocol.json').read_bytes()).hexdigest(),
        scene_version='model-zoom-v13',coverage=item,
        initialization='fixed camera; offline placement only; no target-driven reset aiming',
        zoom_scope='active perspective FOV control; geometric magnification, not physical lens simulation')
    return cfg


def materialize(stage,split,seed,output_dir):
    cfg=resolve(stage,split,seed);out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    path=out/f'zoom-{split}-seed{seed}.json';value=json.dumps(cfg,ensure_ascii=False,indent=2)+'\n'
    if path.exists() and path.read_text(encoding='utf-8')!=value:
        raise FileExistsError('Different scene already exists')
    if not path.exists():path.write_text(value,encoding='utf-8')
    return path


def initial_zone(row,seed,split):
    item=layout(seed,split);box=row.get('selected_bbox')
    return bool(row.get('visible') and box and all(.005<=v<=.995 for v in box)
        and max(abs((row['x']+1)/2-item['planned_center'][0]),abs((row['y']+1)/2-item['planned_center'][1]))<=.025)


def training_cases():
    return [c for c in cases(STAGE,"train",60) if (lambda x:x["layer"]==(x["quadrant"]+x["scale_group"])%3)(layout(c["seed"],"train"))]
