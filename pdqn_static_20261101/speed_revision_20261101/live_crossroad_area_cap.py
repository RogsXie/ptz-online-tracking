"""Whole vehicle tracking with user-requested immediate zoom-out above2% area."""
import speed_bootstrap
import shutil
from pathlib import Path
import live_crossroad_whole_vehicle as previous
from image_area_constraint import filter_action

runner=previous.runner

class Observation(previous.Observation):
    def observe(self,*args,**kwargs):
        state,row=super().observe(*args,**kwargs)
        row['area_guard_candidates']=[dict(area=d.area,confidence=d.confidence,label=d.label,bbox=list(d.bbox))
            for d in getattr(self.detector,'last_detections',[]) if d.confidence>=.5 and d.label in ('car','truck')
            and d.bbox[0]>.01 and d.bbox[1]>.01 and d.bbox[2]<.99 and d.bbox[3]<.99]
        self.latest_rgb_observation=row
        return state,row

def materialize(folder,item):
    path=previous.materialize(folder,item)
    for n in ['live_crossroad_area_cap.py','image_area_constraint.py']:
        shutil.copy2(Path(__file__).parent/n,folder/'frozen-sources'/n)
    return path

def write(path,value):
    if path.name=='configuration.json':
        value.update(explicit_image_area_constraint=True,
            architecture='Frozen15input P-DQN pan/tilt/normal actions plus explicit user-requested RGB zoom constraint. No retraining.',
            action_semantics='Exclusive pulses complete before next observation. RGB area>2% overrides to native-rate zoomout, <=160ms each. Neuralzoomin is clipped/blocked at1.85% settle margin. Others retain frozen P-DQN mapping.',
            area_constraint='Largest of selected whole RGB box and fullyinframe car/truckdetections withconfidence>=.5,1%border margin. Clipped off-target detections do not drive scale; selected box always counts. Perspective area ratio plus RGBfeedback, no GT/speed/schedule. Physical FOV limits apply.',
            area_constraint_scope='largest_reliable_visible_vehicle',
            wide_view_hold_s=2.,wide_view_hold='Afterzoomout preventneuralzoom-in below achievedFOV for2s; pan/tilt and extraoutwardzoomremain available. Avoids immediatelynarrowing the view on targetchange.',
            zoom_out_max_pulse_s=.16,zoom_out_native_rate=True,zoom_area_limit=.02,zoom_settle_area=.0185)
    return previous.write(path,value)

if __name__=='__main__':
    runner.core.frontend=previous.frontend
    runner.cross.materialize=materialize
    runner.Observation=Observation
    runner.write=write
    runner.ACTION_FILTER=filter_action
    runner.main()
