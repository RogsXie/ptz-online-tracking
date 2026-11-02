"""Focused identity, duplicate-fragment, timeout and RGB-bridge regressions."""
import speed_bootstrap
from types import SimpleNamespace
import cv2,numpy as np
from hascam.cfff import CFFF
from hascam.detection import Detection
from robust_locked_tracker import Selector
from verified_visual_bridge import VisualBridge

def detection(x,confidence):return Detection((x,.4,x+.1,.6),confidence,0,'car')
def feature(rgb,d):return np.array([1.,0.])
def make():
    t=SimpleNamespace(diagnostics=dict(sim_time=0.,fov=40.,rotation=[-42.,0.,0.]))
    detector=SimpleNamespace(reset_tracking=lambda:None,lost=0,track=None)
    return t,Selector(CFFF(feature),t,detector,timeout=3.)

def main():
    t,s=make();s.choose(None,[detection(.3,.9),detection(.7,.8)]);locked=s.locked_id
    saved=s.base.current_feature.copy()
    for now in [.1,.5,1.,2.,3.05]:
        t.diagnostics['sim_time']=now
        assert s.choose(None,[detection(.7,.99)]) is None
        assert s.locked_id==locked and np.array_equal(saved,s.base.current_feature)
    t.diagnostics['sim_time']=3.101
    assert s.choose(None,[detection(.7,.99)]) is not None
    assert s.generation==2 and s.info['forgotten_track_id']==locked
    t,s=make();s.choose(None,[detection(.3,.9),detection(.7,.8)]);locked=s.locked_id
    for now in [.1,.2,.3]:
        t.diagnostics['sim_time']=now
        s.choose(None,[detection(.3,.9),detection(.7,.99),Detection((.32,.43,.38,.57),.85,0,'car')])
    t.diagnostics['sim_time']=.4
    assert s.choose(None,[detection(.7,.99),Detection((.32,.43,.38,.57),.85,0,'car')]) is not None
    assert s.locked_id==locked and s.generation==1
    t,s=make();s.choose(None,[detection(.3,.9),detection(.7,.8)]);locked=s.locked_id
    t.diagnostics['sim_time']=.1
    assert s.choose(None,[detection(.7,.99),detection(.3,.3)]) is not None
    assert s.locked_id==locked
    rng=np.random.default_rng(1);image=np.full((240,320,3),90,np.uint8)
    image[80:140,110:190]=rng.integers(20,240,(60,80,3),dtype=np.uint8)
    diag=dict(sim_time=0.,fov=40.,rotation=[-42.,0.,0.]);bridge=VisualBridge()
    bridge.seed(image,(110/320,80/240,190/320,140/240),diag)
    moved=cv2.warpAffine(image,np.float32([[1,0,3],[0,1,2]]),(320,240))
    result=bridge.propose(moved,dict(diag,sim_time=.1));assert result is not None
    assert abs(result.center[0]*320-153)<1 and abs(result.center[1]*240-112)<1
    covered=image.copy();covered[65:155,95:205]=210
    assert bridge.propose(covered,dict(diag,sim_time=.1)) is None
    assert bridge.propose(moved,dict(diag,sim_time=.5)) is None
    print('PASS: sameappearance distractor protection,3stimeout,fragmentreattachment,lowconfidence recovery,verifiedflow,occlusion/expiry rejection')

if __name__=='__main__':main()
