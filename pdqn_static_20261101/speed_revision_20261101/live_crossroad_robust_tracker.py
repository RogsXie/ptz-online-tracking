"""Same locked five-car scene with target-aware detection and robust association."""
import speed_bootstrap
import shutil
from pathlib import Path
import live_crossroad_stable_identity as previous
from robust_locked_tracker import Selector
from target_roi_detector import Detector
from stable_locked_control import filter_action as stable_filter
from calibrate import write as plain_write

runner=previous.runner

class Observation(previous.Observation):
    def __init__(self,t,detector,extractor):
        super().__init__(t,detector,extractor)
        self.cfff=Selector(self.cfff.base,t,detector,timeout=3.)
        self.cfff.feature_batch=self.batch_reid;self.cfff.on_target_change=self.clear_target_motion

def action_filter(env,agent,state,action,duration,params):
    row=env.latest_rgb_observation
    memory=getattr(env,'v3_last_visible',None)
    if getattr(agent,'precision_control_v3',False) and not row['visible'] and memory is not None:
        import numpy as np
        # Camera search only. Association, observations, boxes and3sidentity
        # memory are unchanged. Fast targets cannot afford the fixed250ms hold.
        if max(abs(np.asarray(memory['velocity'])))>=2.:
            return stable_filter(env,agent,state,action,duration,params)
    if not row['visible'] and env.cfff.info.get('locked_track_id') is not None and env.cfff.info.get('missing_for_s',0.)<.25:
        env.lock_searching=True;env.lock_recovering=False;env.t.zoom_gain_override=None
        env.t.pan_tilt_gain_override=None
        return 0,.025,dict(reason='brief_miss_recheck_before_zoom',phase='recheck',input_area=None,
            selected_input_area=None,input_fov_deg=env.t.diagnostics['fov'],executed_action=0,executed_duration_s=.025,
            missing_for_s=env.cfff.info.get('missing_for_s',0.),zoom_gain_override=None)
    return stable_filter(env,agent,state,action,duration,params)

def materialize(folder,item):
    path=previous.materialize(folder,item)
    for n in ['live_crossroad_robust_tracker.py','robust_locked_tracker.py','target_roi_detector.py','verified_visual_bridge.py','locked_target_selector.py']:
        shutil.copy2(Path(__file__).parent/n,folder/'frozen-sources'/n)
    return path

def write(path,value):
    result=previous.write(path,value)
    if path.name=='configuration.json':
        value.update(robust_rgb_tracker=True,brief_miss_grace_s=.25,visual_bridge_max_s=.35,
            tracker='Custom angularKalman high/lowconfidenceassociation, fragment-safe lockedrecovery, targetROI andRGBLK bridge. Not ByteTrack/BoT-SORT.',
            visual_bridge='Forward/backflow<1px, >=65%point/affineinliers, warpedNCC>=.72, ReID>=.60; max.35ssince trusted detector. No prediction-only boxes. Genuineocclusion stillblank.',
            action_semantics='Same stable lockedcontrol; first.25s of unconfirmedloss holds forrecheck before progressivesearch.3smemory unchanged.',
            architecture='FrozenP-DQN+declaredRGBtracking/scale governors; detector/ReIDweights unchanged, no training.')
        return plain_write(path,value)
    return result

if __name__=='__main__':
    runner.core.frontend=previous.previous.previous.previous.frontend
    runner.cross.materialize=materialize;runner.Observation=Observation;runner.Detector=Detector
    runner.write=write;runner.ACTION_FILTER=action_filter;runner.main()
