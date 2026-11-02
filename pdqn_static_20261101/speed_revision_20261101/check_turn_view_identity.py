"""View-change links retain stable identity memory and reject stale/weak links."""
import numpy as np
from check_turn_reassociation import settled,box


def main():
    t,s=settled();locked=s.locked_id
    gallery=[f.copy() for f in s.tracks[locked]['gallery']]
    for now,feature in [(.84,[.8,.6]),(.88,[.65,.76])]:
        s.base.extractor=lambda rgb,d,f=feature:np.array(f)
        t.diagnostics['sim_time']=now;d=box(t,0.)
        assert s.choose(None,[d]) is d and s.locked_id==locked
    evidence=s.info['active_candidate_debug'][0]
    assert evidence['similarity']<.7 and evidence['recent_observed_similarity']>.9
    assert evidence['short_term_identity']
    assert len(gallery)==len(s.tracks[locked]['gallery'])
    assert all(np.array_equal(a,b) for a,b in zip(gallery,s.tracks[locked]['gallery']))
    # A gradual short-term link cannot bypass the .60 stable-bank floor.
    s.base.extractor=lambda rgb,d:np.array([.4,.916])
    t.diagnostics['sim_time']=.92
    assert s.choose(None,[box(t,0.)]) is None
    # Once stale, even a near-identical short-term crop cannot reacquire alone.
    s.base.extractor=lambda rgb,d:np.array([.65,.76])
    t.diagnostics['sim_time']=1.4
    assert s.choose(None,[box(t,0.)]) is None
    # Accepted view-change measurements must update geometric scale without
    # changing the stable appearance bank as the same car moves away.
    t,s=settled();locked=s.locked_id;gallery=[f.copy() for f in s.tracks[locked]['gallery']]
    for i in range(32):
        feature=[.8,.6] if i==0 else [.65,.76]
        s.base.extractor=lambda rgb,d,f=feature:np.array(f)
        t.diagnostics['sim_time']=.84+.04*i;d=box(t,0.);x,y=d.center
        scale=.95**i
        d=type(d)((x-.03*scale,y-.04*scale,x+.03*scale,y+.04*scale),.95,0,'car')
        assert s.choose(None,[d]) is d
    assert all(np.array_equal(a,b) for a,b in zip(gallery,s.tracks[locked]['gallery']))
    print('PASS: fresh geometric short-term view link, bank unchanged, hard appearance floor, stale-link rejection')


if __name__=='__main__':main()
