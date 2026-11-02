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
        self.turn_pending=None;self.turn_evidence={};self.turn_accepted={};self.turn_info={};self.short_term_only_candidates=set()

    def predict(self,track,now):
        dt=max(0.,now-track['kf_time']);F=np.eye(4);F[0,2]=F[1,3]=dt
        G=np.array([[dt*dt/2,0],[0,dt*dt/2],[dt,0],[0,dt]])
        covariance=F@track['kf_cov']@F.T+64.*G@G.T
        if now<track.get('maneuver_until',-1.):
            # White angular acceleration during a confirmed manoeuvre. Unlike
            # the nominal per-frame acceleration, this scales with elapsed time.
            Q=np.zeros((4,4))
            for pos,vel in [(0,2),(1,3)]:
                Q[pos,pos]=dt**3/3;Q[pos,vel]=Q[vel,pos]=dt*dt/2;Q[vel,vel]=dt
            covariance+=1600.*Q
        return F@track['kf_mean'],covariance

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
            pending=self.turn_pending
            if pending is not None and pending['locked_id']==self.locked_id and pending['time']==now:
                # A tentative recovery's temporary track is a duplicate
                # hypothesis, not an established competing vehicle.
                delta=angle-pending['angle'];delta[0]=(delta[0]+180)%360-180
                if np.linalg.norm(delta)<1e-6:track['fragment']=True
        # Partial duplicate boxes of ANY established vehicle are hypotheses,
        # not independent owners. Compensate the saved RGB box for own-camera
        # rotation/zoom and the other track's measured bearing motion.
        for other in self.tracks.values():
            if other is track or other['class_id']!=d.class_id or other.get('hits',0)<3:continue
            if now-other['last_seen']>.15 or 'last_rgb_box' not in other:continue
            old=other['last_rgb_box'];delta_time=now-old['time'];velocity=self.velocity(other)
            corners=[project(bearing(c,old['camera'])+velocity*delta_time,self.t.diagnostics)
                for c in [(old['box'][0],old['box'][1]),(old['box'][2],old['box'][3])]]
            x1,y1=np.minimum(*corners);x2,y2=np.maximum(*corners)
            intersection=max(0.,min(x2,d.bbox[2])-max(x1,d.bbox[0]))*max(0.,min(y2,d.bbox[3])-max(y1,d.bbox[1]))
            if (x2-x1)*(y2-y1)>=1.35*d.area and intersection/max(1e-12,d.area)>=.7:
                track['fragment']=True;break

    def protected_other(self,track):return track.get('hits',0)>=3 and not track.get('fragment',False)
    def allow_gallery_update(self,d):
        return (d is not self.flow_candidate and id(d) in getattr(self,'trusted_gallery_candidates',set())
            and id(d) not in self.short_term_only_candidates)

    def filter_costs(self,cost,ids,candidates,details,now,tangent):
        self.trusted_gallery_candidates=set();self.merge_rejections=[];self.angular_ownership_rejections=[];self.spatial_assignment_rejections=[]
        if self.locked_id not in ids:
            self.trusted_gallery_candidates={id(d) for d in candidates};return
        r=ids.index(self.locked_id)
        self.guard_locked_costs(cost,ids,candidates,details,now,tangent)
        self.recover_turn(cost,ids,candidates,details,now)
        self.guard_locked_costs(cost,ids,candidates,details,now,tangent)
        for j,d in enumerate(candidates):
            ranked=sorted((cost[k,j],k) for k in range(len(ids)) if cost[k,j]<1e6)
            if ranked and details[ranked[0][1],j][1]>=.85 and (len(ranked)==1 or ranked[1][0]>ranked[0][0]+.15):
                self.trusted_gallery_candidates.add(id(d))
        for j,d in enumerate(candidates):
            if cost[r,j]>=1e6:continue
            others=[cost[k,j] for k,tid in enumerate(ids) if tid!=self.locked_id and self.protected_other(self.tracks[tid])]
            if details[r,j][1]>=.85 and (not others or min(others)>cost[r,j]+.15):
                self.trusted_gallery_candidates.add(id(d))
        # A recovery must not immediately rewrite appearance memory.
        self.trusted_gallery_candidates.difference_update(self.turn_accepted)

    def guard_locked_costs(self,cost,ids,candidates,details,now,tangent):
        active=self.tracks[self.locked_id];r=ids.index(self.locked_id)
        for j,d in enumerate(candidates):
            if cost[r,j]>=1e6:continue
            # Missing tracks accumulate covariance, which lowers normalized
            # Mahalanobis cost even far from the actual box. Ownership must
            # also compare angular residuals, not just those normalized costs.
            owner=next((tid for k,tid in enumerate(ids)
                if tid!=self.locked_id and self.protected_other(self.tracks[tid])
                and now-self.tracks[tid]['last_seen']<=.35 and cost[k,j]<1e6
                and details[k,j][0]+.1<details[r,j][0]
                and details[k,j][0]<=.5*details[r,j][0]),None)
            if owner is not None:
                self.angular_ownership_rejections.append(dict(bbox=list(d.bbox),protected_track_id=owner,
                    locked_distance_deg=details[r,j][0],owner_distance_deg=details[ids.index(owner),j][0]))
                self.trusted_gallery_candidates.discard(id(d));cost[r,j]=1e6;continue
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
        # Do not let a maximum-cardinality joint assignment move a freshly
        # observed locked track to a much worse box just to fit another track.
        # Ambiguous crossings still have no preferred measurement.
        valid=[j for j in range(len(candidates)) if cost[r,j]<1e6]
        if valid and now-active['last_seen']<=.15:
            best=min(valid,key=lambda j:details[r,j][0]);distance=details[r,best][0]
            if distance<.25 and details[r,best][1]>=.85:
                for j in valid:
                    if details[r,j][0]>max(.2,3.*distance+.1):
                        self.spatial_assignment_rejections.append(dict(bbox=list(candidates[j].bbox),
                            preferred_distance_deg=distance,rejected_distance_deg=details[r,j][0]))
                        cost[r,j]=1e6

    def recover_turn(self,cost,ids,candidates,details,now):
        """Reattach a locally coherent multi-frame RGB detection after a turn.

        The nominal gate, global assignment, established-track ownership and
        merged-box checks remain in force. This fallback never proposes a box,
        accepts a lone frame, or searches the whole image by appearance alone.
        """
        r=ids.index(self.locked_id);active=self.tracks[self.locked_id]
        self.turn_info=dict(state='nominal')
        if any(cost[r,j]<1e6 for j in range(len(candidates))):
            self.turn_pending=None;return
        eligible=[]
        for j,d in enumerate(candidates):
            evidence=self.turn_evidence.get(id(d))
            if evidence is None or not evidence['local_recovery_eligible']:continue
            # A real, established other car keeps its measurement even if the
            # locked car has nearly identical ReID. New duplicate fragments do
            # not acquire that ownership merely by surviving the first miss.
            owned=any(tid!=self.locked_id and self.protected_other(self.tracks[tid])
                and now-self.tracks[tid]['last_seen']<=.35 and cost[k,j]<1e6
                for k,tid in enumerate(ids))
            if owned:continue
            score=evidence['recovery_distance']/evidence['local_gate']+.5*(1-evidence['similarity'])
            eligible.append((score,j,evidence))
        eligible.sort(key=lambda v:v[0])
        if not eligible or (len(eligible)>1 and eligible[1][0]-eligible[0][0]<.2):
            self.turn_pending=None
            self.turn_info=dict(state='no_unique_local_candidate',eligible=len(eligible));return
        score,j,e=eligible[0];d=candidates[j];previous=self.turn_pending
        confirmed=False;velocity=None;count=1;required=e['required_confirmations']
        if previous is not None and previous['locked_id']==self.locked_id:
            dt=now-previous['time'];delta=e['angle']-previous['angle'];delta[0]=(delta[0]+180)%360-180
            coherent=(0.<dt<=.15 and np.linalg.norm(delta)<=30.*dt+max(.15,.25*e['extent'])
                and .5<=e['angular_area']/previous['angular_area']<=2.
                and cosine(e['feature'],previous['feature'])>=.90)
            if coherent:
                count=previous['count']+1;required=max(required,previous['required'])
                confirmed=count>=required
            if confirmed:velocity=np.clip(delta/dt,-30.,30.)
        self.turn_pending=dict(locked_id=self.locked_id,time=now,angle=e['angle'].copy(),
            feature=e['feature'].copy(),angular_area=e['angular_area'],count=count,required=required)
        self.turn_info=dict(state='confirmed' if confirmed else 'confirming',
            distance_deg=e['distance'],reid_similarity=e['similarity'],confidence=d.confidence,
            confirmations=count,required_confirmations=required)
        if not confirmed:return
        cost[r,j]=.3+.3*e['recovery_distance']/e['local_gate']+.5*(1-e['similarity'])
        details[r,j]=(e['distance'],e['similarity'],e['local_gate'])
        self.turn_accepted[id(d)]=dict(velocity=velocity)
        for debug in self.active_debug:
            if debug['bbox']==list(d.bbox):debug['turn_recovery_confirmed']=True

    def pair_cost(self,track,d,angle,feature,now,tangent):
        mean,P=self.predict(track,now);delta=angle-mean[:2];delta[0]=(delta[0]+180)%360-180
        extent=math.degrees(math.sqrt(max(1e-12,d.area*tangent)))
        sigma=max(.07,.06+.10*extent)*(1.5 if d.confidence<.35 else 1.)
        S=P[:2,:2]+np.eye(2)*sigma*sigma
        mahal=float(delta@np.linalg.solve(S,delta));distance=float(np.linalg.norm(delta))
        gate=min(4.,max(.6,3.7*math.sqrt(float(np.linalg.eigvalsh(S).max()))))
        ratio=d.area*tangent/max(1e-12,track['angular_area'])
        similarity=max(cosine(feature,f) for f in [*track['gallery'],track['recent_feature']])
        # Short-term view changes concern the selected identity. Avoid extra
        # embedding comparisons for every unrelated track/candidate pair.
        observed=track.get('last_observed_feature') if track is self.tracks.get(self.locked_id) else None
        recent_similarity=cosine(feature,observed) if observed is not None else 0.
        flow=d is self.flow_candidate
        threshold=max(self.base.threshold,.70 if track is self.tracks.get(self.locked_id) else .60 if flow or d.confidence<.35 else self.base.threshold)
        short_term_identity=(not flow and now-track['last_seen']<=.15 and d.confidence>=.65
            and similarity>=.60 and recent_similarity>=.90 and mahal<=5.99)
        valid=distance<=gate and mahal<=(5.99 if d.confidence<.35 else 13.8) and .3<=ratio<=3.3 and (similarity>=threshold or short_term_identity)
        if track is self.tracks.get(self.locked_id):
            if short_term_identity and similarity<.85:self.short_term_only_candidates.add(id(d))
            age=max(0.,now-track['last_seen']);local_gate=min(3.,max(.6,gate,1.25*extent))
            recovery_distance=distance
            pending=self.turn_pending
            adjacent_similarity=recent_similarity
            if pending is not None and pending['locked_id']==self.locked_id and 0.<now-pending['time']<=.15:
                # The second confirmation follows the provisional real RGB
                # bearing, rather than requiring it to fit the failed CV model
                # once again. No provisional box is output to the controller.
                delta_pending=angle-pending['angle'];delta_pending[0]=(delta_pending[0]+180)%360-180
                recovery_distance=float(np.linalg.norm(delta_pending))
                local_gate=min(3.,30.*(now-pending['time'])+max(.15,.25*extent))
                adjacent_similarity=cosine(feature,pending['feature'])
            displacement=angle-track['bearing'];displacement[0]=(displacement[0]+180)%360-180
            strong=similarity>=max(.88,threshold) and d.confidence>=.65
            view_change=(similarity>=threshold or (similarity>=.60 and adjacent_similarity>=.90)) and d.confidence>=.85
            eligible=(not valid and not flow and 0.<age<=.35 and (strong or view_change)
                and .3<=ratio<=3.3 and recovery_distance<=local_gate
                and np.linalg.norm(displacement)<=max(.6,30.*age+.3*extent))
            self.turn_evidence[id(d)]=dict(angle=angle.copy(),feature=feature.copy(),extent=extent,
                angular_area=d.area*tangent,distance=distance,recovery_distance=recovery_distance,local_gate=local_gate,
                similarity=similarity,required_confirmations=2 if strong else 3,local_recovery_eligible=bool(eligible))
            self.active_debug.append(dict(bbox=list(d.bbox),confidence=d.confidence,distance_deg=distance,mahalanobis2=mahal,similarity=similarity,recent_observed_similarity=recent_similarity,short_term_identity=bool(short_term_identity),area_ratio=ratio,accepted_gate=bool(valid),source='flow' if flow else 'detector',local_recovery_eligible=bool(eligible)))
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
        recovery=self.turn_accepted.get(id(d)) if track is self.tracks.get(self.locked_id) else None
        if recovery is not None:
            # The old straight-line velocity caused the rejection; retaining
            # it would reject the very next observation again. Use measured
            # RGB bearings only and temporarily admit angular acceleration.
            samples=list(track['samples']);velocity=recovery['velocity']
            if len(samples)>=2:
                dt=now-samples[-2][0];delta=angle-samples[-2][1];delta[0]=(delta[0]+180)%360-180
                if dt>0.:velocity=.5*velocity+.5*np.clip(delta/dt,-30.,30.)
            track.update(kf_mean=np.r_[angle,velocity],kf_cov=np.diag([.04,.04,100.,100.]),maneuver_until=now+.4)
            self.turn_pending=None
        if self.allow_gallery_update(d) and d.confidence>=.35:
            track['recent_feature']=unit(.8*track['recent_feature']+.2*feature)
            track['sizes'].append(track['angular_area'])
        elif id(d) in self.short_term_only_candidates and d.confidence>=.65:
            # Geometric size follows accepted fresh measurements, even while
            # a changed view is deliberately barred from the appearance bank.
            # Freezing both together eventually rejects a receding real car.
            track['sizes'].append(track['angular_area'])
        if track['sizes']:track['angular_area']=float(np.median(track['sizes']))
        track['last_rgb_box']=dict(box=list(d.bbox),time=now,
            camera={k:self.t.diagnostics[k] for k in ['fov','rotation']})
        track['last_observed_feature']=feature.copy()

    def choose(self,rgb,detections,count_step=False):
        self.active_debug=[];self.flow_candidate=None;ds=list(detections)
        self.turn_evidence={};self.turn_accepted={};self.turn_info={};self.short_term_only_candidates=set()
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
            angular_ownership_rejections=getattr(self,'angular_ownership_rejections',[]),
            spatial_assignment_rejections=getattr(self,'spatial_assignment_rejections',[]),
            turn_reassociation=self.turn_info,
            tracker_revision='Joint high/low assignment+angular ownership+lockedReID>=.70+trusted gallery+merged rejection+multi-frame local turn recovery+duplicate fragment protection')
        return chosen
