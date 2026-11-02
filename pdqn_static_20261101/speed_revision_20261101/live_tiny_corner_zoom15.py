"""Requested tiny upper-left target demonstration, with actual zoom15slew.

Only offline initial scene placement changes. Coordinates and camera height
are never supplied to the learned policy; its RGB observation loop is reused.
"""
import speed_bootstrap
import json
import live_speed_zoom15 as zoom
from scenes import configuration,INIT,ground_position

runner=zoom.runner

def layout(split,index):
    return dict(split='development',index=0,seed=39400000,quadrant=0,layer=4,scale_group=0,
        planned_center=[.075,.09],fov_deg=66.,camera_height_cm=6500.,
        role='User requested tiny target near upper-left corner; fresh development video, not qualification')

def materialize(folder,item):
    folder.mkdir(parents=True,exist_ok=True)
    rates=(4.,2.,5.) if item.get('execution_profile')=='released' else (25.,25.,15.)
    cfg=configuration(item,rates)
    init=dict(INIT,camera_location_cm=[-2200.,0.,item['camera_height_cm']])
    cfg['camera']['location_cm']=init['camera_location_cm']
    cfg['targets'][0]['path']['origin_cm']=ground_position(*item['planned_center'],item['fov_deg'],init)
    cfg['scope_note']='Fresh tiny-corner demonstration: fixed65m-high overhead camera, static full-sized car, FOV66 initially. Offline placement only; no GT or camera coordinates in policy.'
    path=folder/'scene-user-tiny-upper-left.json'
    path.write_text(json.dumps(cfg,indent=2),encoding='utf-8')
    return path

if __name__=='__main__':
    runner.layout=layout
    runner.materialize=materialize
    runner.SECONDS_PER_RAW=(.1,.016,.016,.032,.032,1./15.,1./15.)
    runner.write=zoom.write
    runner.main()
