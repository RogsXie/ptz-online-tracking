"""One locked RGB target,3sforgetting,search zoomout,and an identical red follower."""
import speed_bootstrap
import json,copy,shutil
from pathlib import Path
import live_crossroad_area_cap as previous
from locked_target_selector import Selector
from image_area_constraint import filter_action as area_filter
from calibrate import write as plain_write

runner=previous.runner
TIMEOUT=3.
GAP=8.

class Observation(previous.Observation):
    def __init__(self,t,detector,extractor):
        super().__init__(t,detector,extractor)
        self.cfff=Selector(self.cfff.base,t,detector,timeout=TIMEOUT)
        self.cfff.on_target_change=self.clear_target_motion

def action_filter(env,agent,state,action,duration,params):
    row=env.latest_rgb_observation
    if not row['visible']:
        fov=float(env.t.diagnostics['fov']);maximum=float(env.camera_limits['fov_max_deg'])
        env.t.zoom_gain_override=None
        if fov<maximum-1e-4:
            action=6;duration=max(.001,min(.16,(maximum-fov)/env.camera_limits['zoom_fov_deg_s']))
            env.t.zoom_gain_override=1.
            reason='locked_missing_expand_view'
        else:
            action=0;duration=.04;reason='locked_missing_hold_camera'
        return action,duration,dict(reason=reason,executed_action=action,executed_duration_s=duration,
            input_fov_deg=fov,desired_fov_deg=maximum,locked_track_id=env.cfff.locked_id,
            missing_for_s=env.cfff.info.get('missing_for_s',0.),forget_after_s=TIMEOUT,
            input_area=None,selected_input_area=None,zoom_gain_override=env.t.zoom_gain_override)
    return area_filter(env,agent,state,action,duration,params)

def delayed_vehicle(red,gap,identity,color,stop_gap):
    car=copy.deepcopy(red);car.update(id=identity,color_rgb=color)
    delay=gap/4.5
    start=list(red['path']['waypoints_cm'][0]);start[1]-=100*gap
    slow_seconds=stop_gap/8.
    car['path'].update(origin_cm=start,waypoints_cm=[start,*red['path']['waypoints_cm']],
        speed_profile=[[0.,.5],[4.+delay,1.],[7.5+delay,1/3],[12.+delay,2/3],
            [15.+delay,1.],[23.+delay-slow_seconds,1/9],[23.+delay,0.]])
    return car

def materialize(folder,item):
    path=previous.materialize(folder,item);cfg=json.loads(path.read_text(encoding='utf-8'))
    red=cfg['targets'][0]
    follower=delayed_vehicle(red,GAP,'car_red_close_follower',list(red['color_rgb']),GAP)
    purple=delayed_vehicle(red,2*GAP,'car_purple_following',[.48,.025,.72],2*GAP)
    cfg['targets']=[purple if t['id']=='car_purple_following' else t for t in cfg['targets']]
    cfg['targets'].append(follower)
    cfg.update(name='user_locked_identity_identical_red_follower',
        scope_note='Initial maxconfidence RGB target locked through shortocclusion; forgetafter3s. Same-redcolour identical follower8m behind; purple moved16m behind to retain safe ordering. No actor IDs/colors/knownmotion incontrol.')
    runner.demo.MOTION.update(scenario='locked_identity_red_lookalike',red_following_gap_m=GAP,
        purple_following_gap_m=2*GAP,locked_missing_timeout_s=TIMEOUT)
    cfg['motion_schedule']=runner.demo.MOTION
    path.write_text(json.dumps(cfg,indent=2),encoding='utf-8')
    for n in ['live_crossroad_identity_memory.py','locked_target_selector.py']:
        shutil.copy2(Path(__file__).parent/n,folder/'frozen-sources'/n)
    return path

def write(path,value):
    result=previous.write(path,value)
    if path.name=='configuration.json':
        value.update(target_selection='locked_identity_timeout_v1',lost_memory_timeout_s=TIMEOUT,
            selection_rule='Lockinitialmaxconfidence; RGB/ReID multitrack association preservesidentity onshortloss. No other selectedbox duringloss. At3scontinuousloss deleteoldidentity and choosecurrentmaxconfidence.',
            action_semantics='Visible: frozen P-DQN plusRGB2%zoomguard. Missing: native-ratezoomout to66deg thenholdpan/tilt; no trackingotherobjects beforetimeout. Allactions remainexclusivepulses.',
            architecture='Frozen P-DQN+declared RGBmultitrack identity memory and explicitsearch/scale overrides; no training.',
            identical_red_follower=True,red_follow_gap_m=GAP)
        return plain_write(path,value)
    return result

if __name__=='__main__':
    runner.core.frontend=previous.previous.frontend
    runner.cross.materialize=materialize
    runner.Observation=Observation
    runner.write=write
    runner.ACTION_FILTER=action_filter
    runner.main()
