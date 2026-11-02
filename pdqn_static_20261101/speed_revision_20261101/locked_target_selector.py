"""RGB multi-object association with one locked identity and timed forgetting."""
import math
from collections import deque
import numpy as np
from scipy.optimize import linear_sum_assignment
from unified_perception import bearing
from hascam.cfff import unit,cosine

class Selector:
    def __init__(self,base,t,detector,timeout=3.):
        self.base,self.t,self.detector,self.timeout=base,t,detector,float(timeout)
        detector.transport=t
        self.reset()

    def reset(self):
        self.base.reset();self.detector.reset_tracking()
        self.tracks={};self.next_id=1;self.locked_id=None;self.missing_since=None
        self.generation=0;self.reacquisitions=0;self.forgotten_count=0
        self.on_target_change=None;self.info={};self.history=deque()

    def velocity(self,track):
        samples=list(track['samples'])
        if len(samples)<3 or samples[-1][0]-samples[0][0]<.12:return np.zeros(2)
        times=np.array([s[0] for s in samples]);angles=np.stack([s[1] for s in samples])
        dt=times-times.mean()
        return np.clip((dt[:,None]*(angles-angles.mean(0))).sum(0)/max(1e-12,float(dt@dt)),-30.,30.)

    def pair_cost(self,track,d,angle,feature,now,tangent):
        age=max(0.,now-track['last_seen']);predicted=track['bearing']+self.velocity(track)*age
        gate=min(4.,.45+2.*age);delta=angle-predicted;delta[0]=(delta[0]+180)%360-180
        distance=float(np.linalg.norm(delta));ratio=d.area*tangent/max(1e-12,track['angular_area'])
        similarity=max(cosine(feature,f) for f in track['gallery'])
        if distance<=gate and .35<=ratio<=2.85 and similarity>=self.base.threshold:
            return distance/max(.25,gate)+.35*abs(math.log(ratio))+.5*(1-similarity),(distance,similarity,gate)
        return None

    def protected_other(self,track):return True
    def track_created(self,track,d,angle,feature,now):pass
    def track_observed(self,track,d,angle,feature,now):pass
    def allow_gallery_update(self,d):return True
    def assignments(self,cost,candidates):
        return list(zip(*linear_sum_assignment(cost))) if cost.size else []

    def choose(self,rgb,detections,count_step=False):
        diag=self.t.diagnostics;now=float(diag['sim_time']);fov=float(diag['fov'])
        candidates=sorted(detections,key=lambda d:d.confidence,reverse=True)[:16]
        forgotten=None
        if self.locked_id is not None and self.missing_since is not None and now-self.missing_since>=self.timeout:
            forgotten=self.locked_id
            self.tracks.pop(self.locked_id,None);self.locked_id=None;self.missing_since=None
            self.base.reset();self.forgotten_count+=1;self.history.clear()
        self.tracks={k:v for k,v in self.tracks.items() if k==self.locked_id or now-v['last_seen']<=self.timeout+1.}
        batch=getattr(self,'feature_batch',None)
        features=[unit(f) for f in batch(rgb,candidates)] if batch is not None else [unit(self.base.extractor(rgb,d)) for d in candidates]
        angles=[bearing(d.center,diag) for d in candidates]
        tangent=math.tan(math.radians(fov/2))**2
        ids=list(self.tracks);cost=np.full((len(ids),len(candidates)),1e6)
        details={}
        for r,tid in enumerate(ids):
            track=self.tracks[tid]
            for j,d in enumerate(candidates):
                if d.class_id!=track['class_id']:continue
                pair=self.pair_cost(track,d,angles[j],features[j],now,tangent)
                if pair is not None:cost[r,j],details[r,j]=pair
        validate=getattr(self,'filter_costs',None)
        if validate is not None:validate(cost,ids,candidates,details,now,tangent)
        # A visible distractor's established trajectory must not be stolen by a
        # missing locked target, even when both cars have identical appearance.
        ownership_rejections=[]
        if self.locked_id in ids:
            r=ids.index(self.locked_id)
            for j in range(len(candidates)):
                if cost[r,j]>=1e6:continue
                for other,tid in enumerate(ids):
                    if tid==self.locked_id or now-self.tracks[tid]['last_seen']>.35 or not self.protected_other(self.tracks[tid]):continue
                    if cost[other,j]<.65 and cost[other,j]+.12<cost[r,j]:
                        ownership_rejections.append(dict(bbox=list(candidates[j].bbox),protected_track_id=tid,
                            locked_cost=float(cost[r,j]),owner_cost=float(cost[other,j])))
                        cost[r,j]=1e6;break
        assignment={};similarities={}
        if cost.size:
            for r,j in self.assignments(cost,candidates):
                if cost[r,j]<1e6:
                    assignment[int(j)]=ids[r];similarities[int(j)]=details[r,int(j)][1]
        for j,d in enumerate(candidates):
            if j not in assignment:
                tid=self.next_id;self.next_id+=1;assignment[j]=tid
                self.tracks[tid]=dict(class_id=d.class_id,label=d.label,last_seen=now,bearing=angles[j],
                    angular_area=d.area*tangent,samples=deque(),gallery=deque([features[j].copy()],maxlen=6),last_gallery_time=now)
                self.track_created(self.tracks[tid],d,angles[j],features[j],now)
            track=self.tracks[assignment[j]]
            track['last_seen']=now;track['bearing']=angles[j];track['angular_area']=d.area*tangent
            track['samples'].append((now,angles[j]))
            while len(track['samples'])>1 and now-track['samples'][0][0]>.5:track['samples'].popleft()
            # Only clear, accepted crops can update identity memory. Missing
            # detections never clear or overwrite the locked appearance bank.
            if self.allow_gallery_update(d) and d.confidence>=.8 and now-track['last_gallery_time']>=.25 and similarities.get(j,1.)>=.65:
                track['gallery'].append(features[j].copy());track['last_gallery_time']=now
            self.track_observed(track,d,angles[j],features[j],now)
        acquired=False
        if self.locked_id is None and candidates:
            self.locked_id=assignment[0];self.generation+=1;self.reacquisitions=max(0,self.generation-1)
            acquired=True
            if self.on_target_change:self.on_target_change()
        selected_index=next((j for j,tid in assignment.items() if tid==self.locked_id),None)
        chosen=candidates[selected_index] if selected_index is not None else None
        if chosen is not None:
            track=self.tracks[self.locked_id];self.missing_since=None
            self.base.current_feature=unit(np.mean(list(track['gallery']),axis=0))
            self.detector.track=dict(time=now,bearing=track['bearing'],velocity=self.velocity(track),area=chosen.area)
            self.detector.lost=0
            state='tracking';reason='initial_highest_confidence' if acquired and self.generation==1 else 'timeout_then_highest_confidence' if acquired else 'same_identity'
        else:
            if self.locked_id is not None and self.missing_since is None:self.missing_since=now
            self.detector.lost+=1
            state='occluded_search' if self.locked_id is not None else 'search_new_target'
            reason='keep_identity_no_other_box' if self.locked_id is not None else 'no_detections'
        self.info=dict(raw_candidates=len(candidates),association_candidates=len(assignment),lost=self.detector.lost,
            selection_reason=reason,target_generation=self.generation,locked_track_id=self.locked_id,
            memory_state=state,missing_for_s=now-self.missing_since if self.missing_since is not None else 0.,
            forget_after_s=self.timeout,forgotten_track_id=forgotten,forgotten_count=self.forgotten_count,
            reacquisitions=self.reacquisitions,reacquired=acquired,
            selected_label=chosen.label if chosen else None,selected_class_id=chosen.class_id if chosen else None,
            reid_similarity=similarities.get(selected_index),highest_candidate_confidence=candidates[0].confidence if candidates else None,
            candidate_tracks=[dict(track_id=assignment[j],label=d.label,confidence=d.confidence,bbox=list(d.bbox)) for j,d in enumerate(candidates)],
            ownership_rejections=ownership_rejections,
            scope='RGB detections+fresh ReID+own-camera compensated multi-object trajectories; one locked identity, no other box until3scontinuousmissing. No GT/colorfilter/scene IDs.')
        return chosen
