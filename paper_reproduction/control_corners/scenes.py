"""Fixed downward camera, predefined ground-board positions, RGB start checks."""
from copy import deepcopy
from pathlib import Path
import hashlib
import json
import random
import math
from control_curriculum.scenes import resolve as small_scene

HERE=Path(__file__).parent
CORNER_NAMES=('top_left','top_right','bottom_left','bottom_right')
SIGNS=((-1,-1),(1,-1),(-1,1),(1,1))


def load_protocol():
    return json.loads((HERE/'protocol.json').read_text(encoding='utf-8'))


def stage_spec(stage):
    if stage!='static_center':raise ValueError('Four-corner static stage only')
    return load_protocol()['stages'][0]


def cases(stage,split,count=None,offset=0,allow_final_test=False):
    stage_spec(stage)
    if split=='train':base,capacity=710000,10000
    elif split=='validation':base,capacity=810000,40
    else:raise ValueError('Only development and declared corner validation; final test reserved')
    count=count if count is not None else (32 if split=='train' else 40)
    if not 0<count<=capacity or not 0<=offset<=capacity-count:raise ValueError('Outside seed partition')
    return [{'stage':stage,'scenario_name':'control_four_corners','split':split,'seed':base+offset+i,'case_index':offset+i} for i in range(count)]


def category(seed,split):
    if split=='validation':return 'corner',(seed-810000)%4
    if 719100<=seed<719112:index=seed-719100
    elif 718950<=seed<718954:return 'corner',seed-718950
    elif 718900<=seed<718905:return 'corner',(seed-718900)%4
    else:index=seed-711000
    return ('corner','edge','near')[index%12//4],index%4


def ground_position(cx,cy,initialization):
    """Offline scene layout only. No live simulator/GT feedback or policy action."""
    px,py,pz=initialization['camera_location_cm']
    pitch=math.radians(initialization['nominal_camera_pitch_deg'])
    right=(2*cx-1)*math.tan(math.radians(30))
    up=(1-2*cy)*math.tan(math.radians(30))*540/960
    dx=math.cos(pitch)-up*math.sin(pitch)
    dz=math.sin(pitch)+up*math.cos(pitch)
    scale=(initialization['layout_reference_height_cm']-pz)/dz
    return [px+scale*dx,py+scale*right,8.]


def resolve(stage,split,seed):
    stage_spec(stage)
    allowed=710000<=seed<720000 if split=='train' else 810000<=seed<810040 if split=='validation' else False
    if not allowed:raise ValueError('Outside declared static partitions')
    # Scene generation places the target on the board; the installed camera and
    # initial rotation are identical across scenes. No target-driven reset aim.
    cfg=deepcopy(small_scene('static_center','train',211000))
    rng=random.Random(seed)
    kind,quadrant=category(seed,split);sx,sy=SIGNS[quadrant]
    protocol=load_protocol();init=protocol['corner_initialization']
    cfg['targets'][0]['scale']=init['vehicle_scale']
    offset=init['nominal_image_offset']
    if kind=='corner':ox,oy=offset+rng.uniform(-.008,.008),offset+rng.uniform(-.008,.008)
    elif kind=='edge':
        ox,oy=offset,offset
        sx,sy=[(-1,0),(1,0),(0,-1),(0,1)][quadrant]
    else:ox,oy=rng.uniform(.004,.035),rng.uniform(.004,.035)
    cfg['targets'][0]['path']['origin_cm']=ground_position(.5+sx*ox,.5+sy*oy,init)
    cfg.update(name='control_four_corners',split=split,task='corner_static_pdqn_v7')
    cfg['camera'].update(location_cm=init['camera_location_cm'],rotation_deg=[init['nominal_camera_pitch_deg'],0.,0.],
        rotation_jitter_deg=[0.,0.,0.],yaw_limits_deg=protocol['control_execution']['yaw_limits_deg'],
        pitch_limits_deg=protocol['control_execution']['pitch_limits_deg'])
    cfg['control_protocol'].update(name=protocol['name'],seed=seed,stage=stage,
        protocol_sha256=hashlib.sha256((HERE/'protocol.json').read_bytes()).hexdigest(),
        scene_version='fixed-ground-board-no-posts-v7',coverage_kind=kind,corner=CORNER_NAMES[quadrant],corner_index=quadrant,
        initialization='fixed downward installation; predefined ground-board target layout; no live GT aiming')
    return cfg


def materialize(stage,split,seed,output_dir):
    cfg=resolve(stage,split,seed);out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    p=out/f'control_four_corners-{split}-seed{seed}.json';value=json.dumps(cfg,ensure_ascii=False,indent=2)+'\n'
    if p.exists() and p.read_text()!=value:raise FileExistsError('Different scene already exists')
    if not p.exists():p.write_text(value,encoding='utf-8')
    return p


def initial_zone(row,seed,split):
    kind,index=category(seed,split);x,y=(row.get('x'),row.get('y'))
    if not row.get('visible') or x is None or y is None:return False
    box=row.get('selected_bbox');margin=load_protocol()['corner_initialization']['full_bbox_margin']
    if not box or not all(margin<=v<=1-margin for v in box):return False
    cx,cy=(x+1)/2,(y+1)/2
    if kind=='corner':
        zones=load_protocol()['corner_initialization']['zones'];sx,sy=SIGNS[index]
        left,right=zones['left' if sx<0 else 'right'];top,bottom=zones['top' if sy<0 else 'bottom']
        return left<=cx<=right and top<=cy<=bottom
    return True
