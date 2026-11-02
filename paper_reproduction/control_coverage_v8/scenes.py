"""Offline scene layout, with disjoint train/probe/validation seed partitions."""
from copy import deepcopy
from pathlib import Path
import hashlib
import json
import math
import random
from control_curriculum.scenes import resolve as base_scene

HERE = Path(__file__).parent
CORNER_NAMES = ('top_left', 'top_right', 'bottom_left', 'bottom_right')
SIGNS = ((-1, -1), (1, -1), (-1, 1), (1, 1))
BASES = {'train': 2100000, 'probe': 2200000, 'validation': 2300000, 'calibration': 2500000}
CAPACITIES = {'train': 1200, 'probe': 60, 'validation': 120, 'calibration': 126}


def load_protocol():
    return json.loads((HERE / 'protocol.json').read_text(encoding='utf-8'))


def stage_spec(stage):
    if stage != 'static_center':
        raise ValueError('Static centering only; active zoom is a separate future stage')
    return load_protocol()['stages'][0]


def cases(stage, split, count=None, offset=0, allow_final_test=False):
    stage_spec(stage)
    if split not in BASES:
        raise ValueError('Final-test partition is reserved and inaccessible')
    count = CAPACITIES[split] if count is None else count
    if not 0 < count <= CAPACITIES[split] or not 0 <= offset <= CAPACITIES[split] - count:
        raise ValueError('Outside declared partition')
    return [dict(stage=stage, split=split, seed=BASES[split] + i, case_index=i)
            for i in range(offset, offset + count)]


def layout(seed, split):
    if split not in BASES or not BASES[split] <= seed < BASES[split] + CAPACITIES[split]:
        raise ValueError('Outside declared partition')
    index = seed - BASES[split]
    # A permutation covers every quadrant/layer/scale combination exactly once
    # in each block of60; it also spreads the combinations across training time.
    cell = (index % 60) * 17 % 60
    quadrant, layer, scale = cell % 4, cell // 4 % 5, cell // 20
    rng = random.Random(seed)
    cfg = load_protocol()['coverage']
    heldout_style = split == 'validation' or (split == 'calibration' and index >= 60)
    fov_range = cfg['validation_fov_ranges_deg' if heldout_style else 'train_fov_ranges_deg'][scale]
    fov = rng.uniform(*fov_range)
    low, high = cfg['offset_layers'][layer]['max_axis_offset']
    if layer == 0:
        ox, oy = rng.uniform(.001, high), rng.uniform(.001, high)
    else:
        radius = rng.uniform(low, high)
        # The other axis is independently varied rather than tied to a diagonal.
        other = rng.uniform(max(.015, radius * .15), radius)
        ox, oy = (radius, other) if rng.random() < .5 else (other, radius)
    sx, sy = SIGNS[quadrant]
    cx, cy = .5 + sx * ox, .5 + sy * oy
    exact_center = split == 'calibration' and index >= 120
    if exact_center:
        cx = cy = .5
        fov = [41.25, 42.75, 51.25, 52.75, 61.25, 66.][index - 120]
        layer = 0
    return dict(seed=seed, split=split, quadrant=quadrant, layer=layer, scale_group=scale,
                group=f'q{quadrant}-d{layer}-s{scale}', planned_center=[cx, cy], fov_deg=fov,
                heldout_fov_style=heldout_style, exact_center=exact_center)


def category(seed, split):
    case = layout(seed, split)
    return f'offset_layer_{case["layer"]}', case['quadrant']


def ground_position(cx, cy, fov, initialization):
    """Offline projection only; never fed into learner observations or actions."""
    px, py, pz = initialization['camera_location_cm']
    p = math.radians(initialization['nominal_camera_pitch_deg'])
    tangent = math.tan(math.radians(fov / 2))
    right, up = (2 * cx - 1) * tangent, (1 - 2 * cy) * tangent * 540 / 960
    dx, dz = math.cos(p) - up * math.sin(p), math.sin(p) + up * math.cos(p)
    t = (initialization['layout_reference_height_cm'] - pz) / dz
    return [px + t * dx, py + t * right, 8.]


def resolve(stage, split, seed):
    stage_spec(stage)
    item = layout(seed, split)
    protocol = load_protocol()
    init = protocol['corner_initialization']
    cfg = deepcopy(base_scene('static_center', 'train', 211000))
    cfg['targets'][0]['scale'] = 1.
    cfg['targets'][0]['path']['origin_cm'] = ground_position(*item['planned_center'], item['fov_deg'], init)
    cfg.update(name='static_2d_multiscale', split=split, task='coverage_static_pdqn_v8')
    cfg['camera'].update(location_cm=init['camera_location_cm'], rotation_deg=[-60., 0., 0.],
        rotation_jitter_deg=[0., 0., 0.], yaw_limits_deg=[-65., 65.], pitch_limits_deg=[-80., -40.],
        fov_deg=item['fov_deg'], fov_min_deg=item['fov_deg'], fov_max_deg=item['fov_deg'])
    cfg['control_protocol'].update(name=protocol['name'], seed=seed, stage=stage,
        protocol_sha256=hashlib.sha256((HERE/'protocol.json').read_bytes()).hexdigest(),
        scene_version='fixed-ground-2d-multiscale-v8', coverage=item,
        initialization='fixed camera installation and reset orientation; offline target placement; no live GT aiming',
        zoom_scope='initial FOV diversity only; zoom actions disabled and FOV fixed within every episode')
    return cfg


def materialize(stage, split, seed, output_dir):
    cfg = resolve(stage, split, seed)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f'coverage-{split}-seed{seed}.json'
    value = json.dumps(cfg, ensure_ascii=False, indent=2) + '\n'
    if path.exists() and path.read_text(encoding='utf-8') != value:
        raise FileExistsError('Different scene already exists')
    if not path.exists():
        path.write_text(value, encoding='utf-8')
    return path


def initial_zone(row, seed, split):
    item = layout(seed, split)
    box = row.get('selected_bbox')
    if not row.get('visible') or not box or not all(.005 <= v <= .995 for v in box):
        return False
    cx, cy = (row['x'] + 1) / 2, (row['y'] + 1) / 2
    # A projected 90cm reference and an image box center need not coincide
    # exactly. This tolerance is a start-layout check, never a success gate.
    tolerance = load_protocol()['coverage']['initial_center_layout_tolerance']
    return max(abs(cx-item['planned_center'][0]), abs(cy-item['planned_center'][1])) <= tolerance
