"""Causal turn recovery and same-appearance ambiguity/ownership regressions."""
import speed_bootstrap
import numpy as np
from hascam.detection import Detection
from unified_perception import project
from check_robust_tracker import make


def box(t,yaw,confidence=.95):
    x,y=project([yaw,-42.],t.diagnostics)
    return Detection((x-.03,y-.04,x+.03,y+.04),confidence,0,'car')


def settled(other=None):
    t,s=make()
    for i in range(21):
        t.diagnostics['sim_time']=i*.04
        detections=[box(t,0.)]
        if other is not None:detections.append(box(t,other,.85))
        assert s.choose(None,detections) is detections[0]
    return t,s


def main():
    t,s=settled();locked=s.locked_id;gallery=[x.copy() for x in s.tracks[locked]['gallery']]
    t.diagnostics['sim_time']=.84
    assert s.choose(None,[box(t,1.)]) is None
    assert s.info['turn_reassociation']['state']=='confirming'
    # A duplicate read of the same native time cannot confirm a measurement.
    assert s.choose(None,[box(t,1.)]) is None
    t.diagnostics['sim_time']=.88;d=box(t,1.4)
    assert s.choose(None,[d]) is d and s.locked_id==locked and s.generation==1
    assert s.info['turn_reassociation']['state']=='confirmed'
    assert all(np.array_equal(a,b) for a,b in zip(gallery,s.tracks[locked]['gallery']))
    assert np.linalg.eigvalsh(s.tracks[locked]['kf_cov']).min()>0.
    for i in range(1,7):
        t.diagnostics['sim_time']=.88+.04*i;d=box(t,1.4+.4*i)
        assert s.choose(None,[d]) is d and s.locked_id==locked
    # Equally plausible same-appearance boxes cannot be resolved by confidence.
    t,s=settled();locked=s.locked_id
    for now in [.84,.88]:
        t.diagnostics['sim_time']=now
        assert s.choose(None,[box(t,1.),box(t,-1.,.99)]) is None
        assert s.info['turn_reassociation']['state']=='no_unique_local_candidate'
    assert s.locked_id==locked
    # A nearby real car's established trajectory must not be reclaimed.
    t,s=settled(other=1.4);locked=s.locked_id
    for now in [.84,.88,.92]:
        t.diagnostics['sim_time']=now
        assert s.choose(None,[box(t,1.4,.99)]) is None
        assert s.locked_id==locked
    # Large missing-target covariance must not make its Mahalanobis cost
    # artificially more attractive than a nearby car's real trajectory.
    t,s=settled(other=1.4);locked=s.locked_id
    other=next(v for k,v in s.tracks.items() if k!=locked)
    s.tracks[locked]['kf_cov']*=10000.
    other['kf_mean'][0]-=.1
    t.diagnostics['sim_time']=.84
    assert s.choose(None,[box(t,1.4,.99)]) is None
    assert s.info['angular_ownership_rejections']
    # A nominally gated distractor must not suppress confirmation of the real
    # turning target after that distractor is rejected by its existing owner.
    t,s=settled(other=1.4);locked=s.locked_id
    s.tracks[locked]['kf_mean'][0]=.4
    s.tracks[locked]['kf_cov'][0,0]=.08
    t.diagnostics['sim_time']=.84
    assert s.choose(None,[box(t,-1.),box(t,1.4,.99)]) is None
    assert s.info['turn_reassociation']['state']=='confirming'
    t.diagnostics['sim_time']=.88;d=box(t,-1.3)
    assert s.choose(None,[d,box(t,1.4,.99)]) is d
    assert s.locked_id==locked and s.info['turn_reassociation']['state']=='confirmed'
    # View changes may lower gallery similarity while adjacent real crops
    # remain coherent; use THREE confirmations, retaining the nominal .70 gate.
    t,s=settled();locked=s.locked_id
    s.base.extractor=lambda rgb,d:np.array([.8,.6])
    for tm,angle in [(.84,1.),(.88,1.4)]:
        t.diagnostics['sim_time']=tm
        assert s.choose(None,[box(t,angle)]) is None
        assert s.info['turn_reassociation']['required_confirmations']==3
    t.diagnostics['sim_time']=.92;d=box(t,1.8)
    assert s.choose(None,[d]) is d and s.locked_id==locked
    # Joint matching must not force a much worse box to maximize cardinality.
    t,s=settled(other=1.4);locked=s.locked_id
    s.tracks[locked]['kf_cov'][0,0]=.16
    other=next(v for k,v in s.tracks.items() if k!=locked);other['kf_mean'][0]=.02
    t.diagnostics['sim_time']=.84;d=box(t,0.)
    assert s.choose(None,[d,box(t,1.4,.99)]) is d
    assert s.info['spatial_assignment_rejections']
    # A partially duplicated box of a distractor is not a new physical owner.
    t,s=settled(other=4.);t.diagnostics['sim_time']=.84
    whole=box(t,4.,.9);x,y=whole.center
    partial=Detection((x-.02,y-.035,x+.02,y-.01),.7,0,'car')
    s.choose(None,[box(t,0.),whole,partial])
    assert s.tracks[s.next_id-1]['fragment']
    # No local rescue after a stale gap, or for incompatible appearance.
    t,s=settled();t.diagnostics['sim_time']=1.2
    assert s.choose(None,[box(t,2.9)]) is None
    t,s=settled();s.base.extractor=lambda rgb,d:np.array([0.,1.])
    for now in [.84,.88]:
        t.diagnostics['sim_time']=now
        assert s.choose(None,[box(t,1.)]) is None
    print('PASS: two fresh RGB confirmations, same ID, manoeuvre velocity, no template rewrite, ambiguous/owned/stale/appearance rejection')


if __name__=='__main__':main()
