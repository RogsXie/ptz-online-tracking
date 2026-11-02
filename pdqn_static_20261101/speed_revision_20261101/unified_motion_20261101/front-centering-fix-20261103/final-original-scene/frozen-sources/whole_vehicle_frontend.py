"""Frozen learned car/truck whole-box RGB inference, shared by control and display."""
import speed_bootstrap
from speed_bootstrap import REV
from pathlib import Path
from control_model_v13.detector import MultiScaleDetector
from integration.pretrained_detector import YOLODetector
from unified_perception import FoveatedDetector
from integration.experiment import make_extractor

CHECKPOINT = REV/'unified_motion_20261101/whole_vehicle_detector_v1/whole_vehicle.pt'

def suppress_truck_parts(detections):
    """Keep original learned boxes; remove a largely covered duplicate truck part."""
    output = []
    for d in detections:
        duplicate = False
        if d.label=='truck':
            for whole in detections:
                if whole is d or whole.label!='truck' or whole.area<1.35*d.area or whole.confidence<max(.5,.8*d.confidence):
                    continue
                a,b,c,e = d.bbox
                x,y,z,w = whole.bbox
                covered = max(0.,min(c,z)-max(a,x))*max(0.,min(e,w)-max(b,y))/d.area
                if covered>=.7:
                    duplicate = True
                    break
        if not duplicate:
            output.append(d)
    return output

class WholeVehicleRaw(YOLODetector):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        if self.half:
            original=self.model.predict
            def predict(*args,**options):
                if options.pop('half',False):options['quantize']=16
                return original(*args,**options)
            self.model.predict=predict
    def __call__(self,rgb):
        return suppress_truck_parts(super().__call__(rgb))

class WholeVehicleMultiScale(MultiScaleDetector):
    def __init__(self, checkpoint,half=False):
        self.base = WholeVehicleRaw(checkpoint=checkpoint,threshold=.15,device='cuda',imgsz=960,
                                 target_labels=('car','truck'),half=half)
        self.checkpoint = self.base.checkpoint
        self.provenance = self.base.provenance | dict(
            preprocessing_profile='full_and_half_rgb_confidence_nms_v13',scales=[1.,.5],
            selection='confidence descending cross-scale NMS IoU>.5',no_gt_or_fov_input=True,
            whole_vehicle=True,truck_box_definition='Learned full cab+cargo+chassis RGB detection; no GT expansion',
            truck_duplicate_rule='Suppress lower-support truck parts covered>=70% by a >=1.35xarea truck box atconf>=max(.5,.8*partconfidence). Applied to raw fast path and multiscale fusion. Never enlarges or merges coordinates.')
        self.last_seconds = 0.
        self.last_fallback = False
        self.last_scales = []

    def __call__(self,rgb):
        return suppress_truck_parts(super().__call__(rgb))

def frontend(checkpoint=CHECKPOINT,half=False):
    return FoveatedDetector(WholeVehicleMultiScale(Path(checkpoint),half=half)),make_extractor('cuda')
