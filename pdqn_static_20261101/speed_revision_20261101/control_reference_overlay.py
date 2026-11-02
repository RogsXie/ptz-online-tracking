"""Display current-frame detections associated with recorded actual controller selections.

No independent offline tracker can choose a different identity. Reference is the
latest already-observed live RGB box plus own camera optics, never simulator GT.
"""
import bisect, json, math
import numpy as np
from unified_perception import Selector as PreviousSelector, bearing

class Selector(PreviousSelector):
    def __init__(self, base, t, detector, root):
        super().__init__(base, t, detector)
        frames = [json.loads(x) for x in (root/'capture-timestamps.jsonl').read_text().splitlines()]
        times = [f['sim_time'] for f in frames]
        initial = json.loads((root/'initial-observation.json').read_text())
        rows = [initial]+[json.loads(x) for x in (root/'decisions.jsonl').read_text().splitlines()]
        self.references = []
        self.reference_frames=[]
        self.root=root
        self.robust=json.loads((root/'configuration.json').read_text()).get('robust_rgb_tracker',False)
        self.bridge_reference=-1
        if self.robust:
            from verified_visual_bridge import VisualBridge
            self.visual_bridge=VisualBridge()
        for row in rows:
            i = bisect.bisect_left(times, row['sim_time'])
            i = min(range(max(0,i-1), min(len(frames),i+1)), key=lambda j: abs(times[j]-row['sim_time']))
            diag = frames[i]
            self.references.append((row, diag))
            self.reference_frames.append(i)
        self.times = [r['sim_time'] for r,d in self.references]
        self.generation = 1

    def choose(self, rgb, detections, count_step=False):
        diag = self.t.diagnostics
        now = diag['sim_time']
        index = max(0, bisect.bisect_right(self.times, now+1e-6)-1)
        ref, optics = self.references[index]
        association = ref.get('rgb_association', {})
        self.generation = association.get('target_generation', 1)
        self.info = dict(association,reference_sim_time=ref['sim_time'],overlay_observation_source='detector', overlay_reference='Latest actual live RGB selection, current-frame RGB evidence; no GT')
        if not ref['visible']:
            return None
        predicted = bearing(((ref['x']+1)/2,(ref['y']+1)/2), optics)
        velocity = np.asarray(ref.get('image_motion_bearing_rate_deg_s', [0.,0.]))
        predicted += velocity*max(0.,now-ref['sim_time'])
        angular_area = ref['area']*math.tan(math.radians(optics['fov']/2))**2
        expected = angular_area/max(1e-12,math.tan(math.radians(diag['fov']/2))**2)
        ranked = []
        gate=min(1.,.4+1.5*max(0.,now-ref['sim_time'])) if association.get('locked_track_id') is not None else 2.5
        for d in detections:
            if ref.get('target_class_id') is not None and d.class_id!=ref['target_class_id']:
                continue
            delta = bearing(d.center,diag)-predicted
            delta[0] = (delta[0]+180)%360-180
            distance = float(np.linalg.norm(delta))
            ratio = d.area/max(expected,1e-12)
            if .25<=ratio<=4. and distance<=gate:
                ranked.append((distance/.5+abs(math.log(ratio))+.2*(1-d.confidence), d))
        selected = min(ranked,key=lambda x:x[0])[1] if ranked else None
        if selected is None and self.robust:
            # Local RGB re-detection uses the recorded live selection only as a
            # search reference, never as a box to draw without image evidence.
            import cv2
            from hascam.detection import Detection
            from unified_perception import project
            h,w=rgb.shape[:2];cx,cy=project(predicted,diag)
            rw=min(w, max(180,int(math.sqrt(max(expected,1e-9)*w*h)*7)))
            rh=min(h,max(100,int(rw*h/w)))
            left=int(np.clip(cx*w-rw/2,0,w-rw));top=int(np.clip(cy*h-rh/2,0,h-rh))
            local=[]
            for d in self.detector.base.base(rgb[top:top+rh,left:left+rw]):
                if ref.get('target_class_id') is not None and d.class_id!=ref['target_class_id']:continue
                b=(np.asarray(d.bbox)*[rw,rh,rw,rh]+[left,top,left,top])/[w,h,w,h]
                z=Detection(tuple(np.clip(b,0,1)),d.confidence,d.class_id,d.label)
                delta=bearing(z.center,diag)-predicted;delta[0]=(delta[0]+180)%360-180
                distance=float(np.linalg.norm(delta));ratio=z.area/max(expected,1e-12)
                if distance<=gate and .3<=ratio<=3.3:local.append((distance+.2*abs(math.log(ratio)),z))
            if local:
                selected=min(local,key=lambda x:x[0])[1];self.info['overlay_observation_source']='detector_roi'
            else:
                if self.bridge_reference!=index:
                    reference=cv2.cvtColor(cv2.imread(str(self.root/f'frames/{self.reference_frames[index]:05d}.jpg')),cv2.COLOR_BGR2RGB)
                    self.visual_bridge.seed(reference,ref['bbox'],optics,ref.get('target_label') or 'car',ref.get('target_class_id') or 0)
                    self.bridge_reference=index
                flow=self.visual_bridge.propose(rgb,diag)
                if flow is not None:
                    delta=bearing(flow.center,diag)-predicted;delta[0]=(delta[0]+180)%360-180
                    if np.linalg.norm(delta)<=gate:
                        selected=flow;self.info['overlay_observation_source']='verified_optical_flow'
        self.detector.lost = 0 if selected else self.detector.lost+1
        if selected:
            self.detector.track = dict(time=now,bearing=bearing(selected.center,diag),velocity=velocity,area=selected.area)
        return selected
