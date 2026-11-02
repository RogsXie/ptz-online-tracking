"""Prospective held-out corner emphasis; training layouts stay unchanged.

Declared before any validation/final-test rollout. Full-frame margins depend
only on the measured calibration size groups, never on a live policy action.
"""
import random
from scenes import layout as original_layout, SIGNS

MARGINS=[(.12,.12),(.15,.18),(.17,.23),(.18,.27)]

def corner_center(seed,quadrant,scale):
    r=random.Random(seed+7183);mx,my=MARGINS[scale]
    # Separate x and y jitter. A large box's center cannot reach (.1,.1)
    # while the full box remains visible; actual border gaps are measured.
    mx+=r.uniform(-.012,.012);my+=r.uniform(-.012,.012)
    sx,sy=SIGNS[quadrant]
    return [mx if sx<0 else 1-mx,my if sy<0 else 1-my]

def layout(split,index):
    item=original_layout(split,index)
    if split in ('validation','final_test') and item['layer']==4:
        item['planned_center']=corner_center(item['seed'],item['quadrant'],item['scale_group'])
        item['heldout_corner_rule']='independent xy jitter with group-specific complete-box margins; prospectively declared20261101'
    return item

def calibration_corner(index):
    item=original_layout('development',20000+index)
    q=index%4;s=index//4%4;r=random.Random(item['seed'])
    ranges=[(56.,63.),(40.,44.),(27.,30.),(19.,21.)]
    item.update(quadrant=q,scale_group=s,layer=4,fov_deg=r.uniform(*ranges[s]),
        planned_center=corner_center(item['seed'],q,s),
        role='development corner layout calibration only, never validation/final test')
    return item

