"""Same five-vehicle scene, unified lock-aware scale/search and bounded correction."""
import speed_bootstrap
import shutil
from pathlib import Path
import live_crossroad_identity_memory as previous
from stable_locked_control import filter_action
from batched_vehicle_reid import BatchFeatures
from calibrate import write as plain_write

runner=previous.runner

class Observation(previous.Observation):
    def __init__(self,t,detector,extractor):
        super().__init__(t,detector,extractor)
        self.batch_reid=BatchFeatures(extractor)
        self.cfff.feature_batch=self.batch_reid

def materialize(folder,item):
    path=previous.materialize(folder,item)
    for n in ['live_crossroad_stable_identity.py','stable_locked_control.py','batched_vehicle_reid.py']:
        shutil.copy2(Path(__file__).parent/n,folder/'frozen-sources'/n)
    return path

def write(path,value):
    result=previous.write(path,value)
    if path.name=='configuration.json':
        value.update(stable_locked_control=True,batched_fresh_reid=True,
            architecture='Same frozen P-DQN/detector/ReID; RGBremainingerror Qmask/durationlimit, progressive search and immediate locked recovery. Engineering overrides, no RLtraining.',
            action_semantics='Exclusive pulses; pan/tilt direction retained from allowed neuralQactions, bounded by85%predicted remaining pixelerror;4pxdeadband,6pxreversal margin. Search widens by6degrees perfeedback; recovery cancels search immediately and restores scale aftercentering.',
            area_constraint_scope='locked_target_only',wide_view_hold_s=0.,
            wide_view_hold='Disabled forsinglelockedtarget; visible recovery immediately permits pan/tilt and scale restoration.',
            zoom_out_max_pulse_s=.06,recovery_zoom_in_max_pulse_s=.06,
            area_constraint='Onlylocked wholeRGBbox controls2%scale; otheractors cannot forceoutwardzoom or blockrecovery.',
            reid_batch='Identical frozen FastReID model/preprocessing, up to8freshcrops perGPUcall. No cachedfeatureidentity substitution.')
        value['reid'].update(runtime_refresh_period_s=0.,refresh_rule='Fresh batch inference for every currentcandidate; no featurecache reuse.',batch_max_crops=8)
        return plain_write(path,value)
    return result

if __name__=='__main__':
    runner.core.frontend=previous.previous.previous.frontend
    runner.cross.materialize=materialize
    runner.Observation=Observation
    runner.write=write
    runner.ACTION_FILTER=filter_action
    runner.main()
