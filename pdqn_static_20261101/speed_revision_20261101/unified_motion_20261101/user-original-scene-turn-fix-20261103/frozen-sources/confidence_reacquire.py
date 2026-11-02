"""Keep an RGB-associated target; immediately select highest confidence on loss."""
import math
import numpy as np
from unified_perception import Selector as PreviousSelector, bearing
from hascam.cfff import unit, cosine

class Selector(PreviousSelector):
    def reset(self):
        super().reset()
        self.generation = 0
        self.reacquisitions = 0
        self.on_target_change = None
        self.last_observation_lost = True
        self.last_class_id = None

    def choose(self, rgb, detections, count_step=False):
        candidates = sorted(detections, key=lambda d: d.confidence, reverse=True)
        diag = self.t.diagnostics
        now, fov = float(diag['sim_time']), float(diag['fov'])
        chosen = None
        feature = None
        similarity = None
        velocity = np.zeros(2)
        matched = []
        feature_cache = {}
        def extract(d):
            key = id(d)
            if key not in feature_cache:
                feature_cache[key] = unit(self.base.extractor(rgb, d))
            return feature_cache[key]
        continuing = self.history and self.base.current_feature is not None and not self.last_observation_lost
        if continuing:
            while len(self.history) > 1 and now-self.history[0]['time'] > .7:
                self.history.popleft()
            last = self.history[-1]
            times = np.array([v['time'] for v in self.history])
            angles = np.stack([v['bearing'] for v in self.history])
            if len(times) >= 3 and times[-1]-times[0] >= .12:
                dt = times-times.mean()
                velocity = np.clip((dt[:,None]*(angles-angles.mean(0))).sum(0)/max(1e-12,float(dt@dt)), -30., 30.)
            age = max(0., now-last['time'])
            predicted = last['bearing']+velocity*age
            expected = np.median([v['angular_area'] for v in self.history])/max(1e-12, math.tan(math.radians(fov/2))**2)
            for d in candidates:
                if getattr(self,'match_same_class',False) and d.class_id != self.last_class_id:
                    continue
                delta = bearing(d.center, diag)-predicted
                delta[0] = (delta[0]+180)%360-180
                distance = float(np.linalg.norm(delta))
                ratio = d.area/max(expected, 1e-12)
                if not (.25 <= ratio <= 4. and distance <= max(2.5, 1.+15*age)):
                    continue
                f = extract(d)
                sim = cosine(f, self.base.current_feature)
                if sim >= self.base.threshold:
                    score = distance/.5+abs(math.log(ratio))+2*(1-sim)
                    matched.append((score, d, f, sim))
            if matched:
                _, chosen, feature, similarity = min(matched, key=lambda x: x[0])
        reacquired = chosen is None and bool(candidates)
        reason = 'continued'
        if reacquired:
            reason = 'initial_highest_confidence' if not self.generation else 'lost_then_highest_confidence'
            if self.generation:
                self.reacquisitions += 1
            self.generation += 1
            chosen = candidates[0]
            feature = extract(chosen)
            self.history.clear()
            velocity = np.zeros(2)
            self.base.reset()
            if self.on_target_change:
                self.on_target_change()
        if chosen is not None:
            old = self.base.current_feature
            self.base.current_feature = feature.copy() if old is None else unit(.85*feature+.15*old)
            angle = bearing(chosen.center, diag)
            self.history.append(dict(time=now, bearing=angle, angular_area=chosen.area*math.tan(math.radians(fov/2))**2))
            self.detector.track = dict(time=now, bearing=angle, velocity=velocity, area=chosen.area)
            self.detector.lost = 0
            self.last_class_id = chosen.class_id
        else:
            reason = 'no_detections'
            self.detector.lost += 1
        self.last_observation_lost = chosen is None
        self.info = dict(raw_candidates=len(candidates), association_candidates=len(matched),
            lost=self.detector.lost, selection_reason=reason, target_generation=self.generation,
            reacquisitions=self.reacquisitions, reacquired=reacquired, reid_similarity=similarity,
            selected_label=chosen.label if chosen else None, selected_class_id=chosen.class_id if chosen else None,
            highest_candidate_confidence=candidates[0].confidence if candidates else None,
            scope='RGB only; highest confidence at initialization/on loss, motion+fresh ReID association otherwise. No persistent geometry gate on reacquisition.')
        return chosen
