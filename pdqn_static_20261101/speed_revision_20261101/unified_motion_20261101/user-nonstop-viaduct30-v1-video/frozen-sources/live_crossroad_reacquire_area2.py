"""User-requested 2% box area and highest-confidence reacquisition."""
import speed_bootstrap
import shutil
from pathlib import Path
import live_crossroad_truck_occlusion as previous
from confidence_reacquire import Selector

runner = previous.runner
OriginalObservation = runner.Observation

class Observation(OriginalObservation):
    def __init__(self, t, detector, extractor):
        super().__init__(t, detector, extractor)
        self.cfff = Selector(self.cfff.base, t, detector)
        self.cfff.on_target_change = self.clear_target_motion
        extractor.force_real = True

    def clear_target_motion(self):
        self.forward_x = self.forward_y = -2.
        if hasattr(self, 'motion_features'):
            self.motion_features.samples.clear()

    def reset(self, *args, **kwargs):
        result = super().reset(*args, **kwargs)
        self.cfff.on_target_change = self.clear_target_motion
        return result

    def observe(self, *args, **kwargs):
        # reset() recreates association state, so restore the callback before selection.
        self.cfff.on_target_change = self.clear_target_motion
        state, row = super().observe(*args, **kwargs)
        if row['visible']:
            state = state.copy()
            state[2] *= 2.
        row.update(requested_area_fraction=.02, policy_area_feature=float(state[2]),
            area_mapping='Actual RGB box area *100 *(.04/.02); other state inputs and frozen model unchanged.')
        return state, row

def materialize(folder, item):
    path = previous.materialize(folder, item)
    for n in ['live_crossroad_reacquire_area2.py', 'confidence_reacquire.py', 'unified_core.py']:
        shutil.copy2(Path(__file__).parent/n, folder/'frozen-sources'/n)
    return path

def write(path, value):
    if path.name == 'configuration.json':
        value.update(target_selection='highest_confidence_on_loss_v1', target_area_fraction=.02,
            policy_area_mapping='Frozen model trained at4%; input area multiplied by2 to request2% actual box area. No training or claim of exact lock.',
            reid_force_fresh=True, target_change_clears_motion_history=True, control_reference_overlay=True,
            selection_rule='First target=max confidence. Continue motion+appearance match; if none, immediately reacquire max confidence of all current detections; empty frame waits for next detections.')
    return previous.previous.write(path, value)

if __name__ == '__main__':
    runner.cross.materialize = materialize
    runner.Observation = Observation
    runner.write = write
    runner.main()
