"""Offline layout; policy never receives scene coordinates or FOV."""
import speed_bootstrap
from speed_bootstrap import REV
import json,random
import scenes as original
from evaluation_scenes import corner_center

RATES=(25.,25.,5.)
def layout(split,index):
    item=original.layout(split,6000000+index)
    item.update(index=index,source_case_index=6000000+index,revision='speed_revision_20261101')
    if item['layer']==4 and split in ('train','validation','final_test'):
        center=corner_center(item['seed'],item['quadrant'],item['scale_group'])
        if split=='train' and item['scale_group']==0:
            center=[v+.02 if v<.5 else v-.02 for v in center]
        item['planned_center']=center
    return item

def materialize(folder,item):
    rates=(4.,2.,5.) if item.get('execution_profile')=='released' else RATES
    return original.materialize(folder,item,rates)
