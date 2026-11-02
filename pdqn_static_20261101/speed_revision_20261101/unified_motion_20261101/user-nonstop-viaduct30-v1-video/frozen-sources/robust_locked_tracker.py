"""Angular Kalman association, two-stage matching and verified RGB short bridges."""
import math
from collections import deque
import numpy as np
from scipy.optimize import linear_sum_assignment
from hascam.cfff import cosine,unit
from locked_target_selector import Selector as PreviousSelector
from verified_visual_bridge import VisualBridge
from unified_perception import bearing,project

class Selector(PreviousSelector):
    def reset(self):
        super().reset();self.bridge=VisualBridge();self.flow_candidate=None;self.active_debug=[]

    def predict(self,track,now):
        dt=max(0.,now-track['kf_time']);F=np.eye(4);F[0,2]=F[1,3]=dt
        G=np.array([[dt*dt/2,0],[0,dt*dt/2],[dt,0],[0,dt]])
        return F@track['kf_mean'],F@track['kf_cov']@F.T+64.*G@G.T

    def velocity(self,track):
        return np.clip(track['kf_mean'][2:],-30.,30.) if 'kf_mean' in track else super().velocity(track)

    def track_created(self,track,d,angle,feature,now):
        track.update(kf_mean=np.r_[angle,[0.,0.]],kf_cov=np.diag([.04,.04,9.,9.]),kf_time=now,
            created_at=now,hits=0,fragment=False,recent_feature=feature.copy(),sizes=deque(maxlen=3))
        active=self.tracks.get(self.locked_id)
        if active is not None and active is not track and d.class_id==active['class_id']:
            predicted,_=self.predict(active,now);delta=angle-predicted[:2];delta[0]=(delta[0]+180)%360-180
            if np.linalg.norm(delta)<min(3.,1.+3.*max(0.,now-active['last_seen'])):
                track['fragment']=True

    def protected_other(self,track):return track.get('hits',0)>=3 and not track.get('fragment',False)
    def allow_gallery_update(self,d):
        return d is not self.flow_candidate and id(d) in getattr(self,'trusted_gallery_candidates',set())

    def filter_costs(self,cost,ids,candidates,details,now,tangent):
        self.trusted_gallery_candidates=set();self.merge_rejections=[]
        if self.locked_id not in ids:
            self.trusted_gallery_candidates={id(d) for d in candidates};return
        active=self.tracks[self.locked_id];r=ids.index(self.locked_id)
        for j,d in enumerate(candidates):
            ranked=sorted((cost[k,j],k) for k in range(len(ids)) if cost[k,j]<1e6)
            if ranked and details[ranked[0][1],j][1]>=.85 and (len(ranked)==1 or ranked[1][0]>ranked[0][0]+.15):
                self.trusted_gallery_candidates.add(id(d))
        for j,d in enumerate(candidates):
            if cost[r,j]>=1e6:continue
            # A detector box covering multiple established tracks cannot be
            # treated as one trusted identity or used to refresh its template.
            ratio=d.area*tangent/max(1e-12,active['angular_area'])
            covered=[]
            if ratio>1.65:
                for tid in ids:
                    t=self.tracks[tid]
                    if t['class_id']!=d.class_id or now-t['last_seen']>.35:continue
                    if tid!=self.locked_id and not self.protected_other(t):continue
                    mean,_=self.predict(t,now);x,y=project(mean[:2],self.t.diagnostics)
                    if d.bbox[0]<=x<=d.bbox[2] and d.bbox[1]<=y<=d.bbox[3]:covered.append(tid)
            if self.locked_id in covered and len(covered)>1:
                self.merge_rejections.append(dict(bbox=list(d.bbox),covered_track_ids=covered,area_ratio=ratio))
                self.trusted_gallery_candidates.discard(id(d))
                cost[r,j]=1e6;continue
            others=[cost[k,j] for k,tid in enumerate(ids) if tid!=self.locked_id and self.protected_other(self.tracks[tid])]
            if details[r,j][1]>=.85 and (not others or min(others)>cost[r,j]+.15):
                self.trusted_gallery_candidates.add(id(d))

    def pair_cost(self,track,d,angle,feature,now,tangent):
        mean,P=self.predict(track,now);delta=angle-mean[:2];delta[0]=(delta[0]+180)%360-180
        extent=math.degrees(math.sqrt(max(1e-12,d.area*tangent)))
        sigma=max(.07,.06+.10*extent)*(1.5 if d.confidence<.35 else 1.)
        S=P[:2,:2]+np.eye(2)*sigma*sigma
        mahal=float(delta@np.linalg.solve(S,delta));distance=float(np.linalg.norm(delta))
        gate=min(4.,max(.6,3.7*math.sqrt(float(np.linalg.eigvalsh(S).max()))))
        ratio=d.area*tangent/max(1e-12,track['angular_area'])
        similarity=max(cosine(feature,f) for f in [*track['gallery'],track['recent_feature']])
        flow=d is self.flow_candidate
        threshold=max(self.base.threshold,.70 if track is self.tracks.get(self.locked_id) else .60 if flow or d.confidence<.35 else self.base.threshold)
        valid=distance<=gate and mahal<=(5.99 if d.confidence<.35 else 13.8) and .3<=ratio<=3.3 and similarity>=threshold
        if track is self.tracks.get(self.locked_id):
            self.active_debug.append(dict(bbox=list(d.bbox),confidence=d.confidence,distance_deg=distance,mahalanobis2=mahal,similarity=similarity,area_ratio=ratio,accepted_gate=bool(valid),source='flow' if flow else 'detector'))
        if not valid:return None
        penalty=.7 if track.get('fragment',False) and track is not self.tracks.get(self.locked_id) else 0.
        value=mahal/13.8+.3*abs(math.log(ratio))+.5*(1-similarity)-.1*d.confidence+penalty+(.3 if flow else 0.)
        return value,(distance,similarity,gate)

    def assignments(self,cost,candidates):
        remaining_rows=list(range(cost.shape[0]));remaining_cols=list(range(cost.shape[1]));result=[]
        for high in [True,False]:
            cols=[j for j in remaining_cols if (candidates[j].confidence>=.5)==high]
            if not cols or not remaining_rows:continue
            # Solve all trajectories jointly; a locked ID has no right to
            # preempt another visible vehicle's better-supported measurement.
            rr,cc=linear_sum_assignment(cost[np.ix_(remaining_rows,cols)])
            pairs=[(remaining_rows[r],cols[c]) for r,c in zip(rr,cc) if cost[remaining_rows[r],cols[c]]<1e6]
            result.extend(pairs);used_r={r for r,c in pairs};used_c={c for r,c in pairs}
            remaining_rows=[r for r in remaining_rows if r not in used_r];remaining_cols=[j for j in remaining_cols if j not in used_c]
        return result

    def track_observed(self,track,d,angle,feature,now):
        mean,P=self.predict(track,now);extent=math.degrees(math.sqrt(max(1e-12,track['angular_area'])))
        sigma=max(.07,.06+.10*extent)*(1.5 if d.confidence<.35 else 1.)
        S=P[:2,:2]+np.eye(2)*sigma*sigma;K=np.linalg.solve(S,P[:2,:]).T
        innovation=angle-mean[:2];innovation[0]=(innovation[0]+180)%360-180
        mean+=K@innovation;mean[2:]=np.clip(mean[2:],-30,30)
        H=np.zeros((2,4));H[:2,:2]=np.eye(2);I=np.eye(4)-K@H
        track['kf_mean']=mean;track['kf_cov']=I@P@I.T+K@(np.eye(2)*sigma*sigma)@K.T;track['kf_time']=now;track['hits']+=1
        if self.allow_gallery_update(d) and d.confidence>=.35:
            track['recent_feature']=unit(.8*track['recent_feature']+.2*feature)
            track['sizes'].append(track['angular_area'])
        if track['sizes']:track['angular_area']=float(np.median(track['sizes']))

    def choose(self,rgb,detections,count_step=False):
        self.active_debug=[];self.flow_candidate=None;ds=list(detections)
        active=self.tracks.get(self.locked_id);supported=False
        if active is not None:
            now=self.t.diagnostics['sim_time'];predicted,_=self.predict(active,now)
            tangent=math.tan(math.radians(self.t.diagnostics['fov']/2))**2
            for d in ds:
                if d.class_id!=active['class_id'] or d.confidence<.55:continue
                delta=bearing(d.center,self.t.diagnostics)-predicted[:2];delta[0]=(delta[0]+180)%360-180
                ratio=d.area*tangent/max(1e-12,active['angular_area'])
                if np.linalg.norm(delta)<min(1.2,.5+2*max(0.,now-active['last_seen'])) and .35<=ratio<=2.85:
                    supported=True;break
        if not supported:self.flow_candidate=self.bridge.propose(rgb,self.t.diagnostics)
        else:self.bridge.reason='detector_supported';self.bridge.proposal=None
        if self.flow_candidate is not None:
            # A genuine detector result is preferred over a redundant flow box.
            from integration.metrics import iou
            if any(d.class_id==self.flow_candidate.class_id and d.confidence>=.3 and iou(d.bbox,self.flow_candidate.bbox)>.3 for d in ds):
                self.flow_candidate=None
            else:ds.append(self.flow_candidate)
        chosen=super().choose(rgb,ds,count_step)
        source='missing'
        if chosen is not None:
            if chosen is self.flow_candidate:
                source='verified_optical_flow';self.bridge.accept()
            else:
                source='detector_roi' if id(chosen) in getattr(self.detector,'roi_ids',set()) else 'detector_full'
                if chosen.confidence>=.35:self.bridge.seed(rgb,chosen.bbox,self.t.diagnostics,chosen.label,chosen.class_id)
            track=self.tracks[self.locked_id]
            w,h=chosen.bbox[2]-chosen.bbox[0],chosen.bbox[3]-chosen.bbox[1]
            self.detector.track.update(angular_area=track['angular_area'],class_id=chosen.class_id,
                box_aspect=w*960/max(1e-9,h*540),bbox=list(chosen.bbox),fov=self.t.diagnostics['fov'])
        self.info.update(observation_source=source,visual_bridge_reason=self.bridge.reason,
            locked_bearing_velocity_deg_s=self.velocity(self.tracks[self.locked_id]).tolist() if chosen is not None else None,
            visual_bridge_quality=self.bridge.proposal['quality'] if self.bridge.proposal else None,
            active_candidate_debug=self.active_debug,
            merged_candidate_rejections=getattr(self,'merge_rejections',[]),
            tracker_revision='Joint high/low assignment+always-on trajectory ownership+lockedReID>=.70+trusted-only gallery+merged-box rejection')
        return chosen
