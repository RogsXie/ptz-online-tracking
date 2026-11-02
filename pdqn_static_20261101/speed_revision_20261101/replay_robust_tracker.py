"""Bounded replay of the previous live RGB observations, GT only in final audit."""
import speed_bootstrap
import json,time,bisect,collections,argparse
from pathlib import Path
from types import SimpleNamespace
import cv2,numpy as np,torch
from whole_vehicle_frontend import frontend
from target_roi_detector import Detector
from fast_rgb_perception import PeriodicReID
from batched_vehicle_reid import BatchFeatures
from hascam.cfff import CFFF
from robust_locked_tracker import Selector
from integration.metrics import iou

def main():
    torch.set_num_threads(4)
    root=Path('unified_motion_20261101');source=root/'user-crossroad-stable-identity-video'
    parser=argparse.ArgumentParser();parser.add_argument('--name',default='robust-tracker-rgb-replay-aligned');args=parser.parse_args()
    output=root/args.name;output.mkdir(exist_ok=False)
    rows=[json.loads((source/'initial-observation.json').read_text())]+[json.loads(x) for x in (source/'decisions.jsonl').read_text().splitlines()]
    frames=[json.loads(x) for x in (source/'capture-timestamps.jsonl').read_text().splitlines()];times=[x['sim_time'] for x in frames]
    detector,extractor=frontend();detector=Detector(detector.base);extractor=PeriodicReID(extractor)
    t=SimpleNamespace(diagnostics=frames[0]);extractor.transport=t
    selector=Selector(CFFF(extractor),t,detector);selector.feature_batch=BatchFeatures(extractor)
    extractor.association=selector;extractor.detector=detector
    results=[]
    with (output/'predictions.jsonl').open('w',encoding='utf-8') as log:
        for k,row in enumerate(rows):
            j=bisect.bisect_left(times,row['sim_time']);i=min(range(max(0,j-1),min(len(frames),j+1)),key=lambda i:abs(times[i]-row['sim_time']))
            frame=frames[i];t.diagnostics=frame;rgb=cv2.cvtColor(cv2.imread(str(source/f'frames/{i:05d}.jpg')),cv2.COLOR_BGR2RGB)
            start=time.perf_counter();d=detector(rgb)
            if k==0:
                from hascam.detection import Detection
                # Replay starts from the actual live RGB-selected identity.
                # JPEG compression can change cold confidence ordering; do not
                # accidentally compare another car against the original lock.
                selector.choose(rgb,[Detection(tuple(row['bbox']),row['confidence'],row.get('target_class_id') or 0,row.get('target_label') or 'car')])
            chosen=selector.choose(rgb,d);elapsed=time.perf_counter()-start
            result=dict(frame=i,sim_time=frame['sim_time'],visible=chosen is not None,bbox=list(chosen.bbox) if chosen else None,confidence=chosen.confidence if chosen else None,association=selector.info.copy(),perception_s=elapsed)
            results.append(result);log.write(json.dumps(result)+'\n')
            if (k+1)%60==0:print('RGB REPLAY',k+1,len(rows),flush=True)
    # Simulator diagnostics are first read after all replay predictions finish.
    gt=[json.loads(x) for x in (source/'offline-diagnostics.jsonl').read_text().splitlines()]
    metrics={};wrong=[];identity_by_generation={}
    for r in results:
        if not r['visible']:continue
        score,identity=max((iou(r['bbox'],t.get('gt_bbox_clipped_normalized')),t['id']) for t in gt[r['frame']]['targets'])
        if score>=.3:
            generation=r['association']['target_generation']
            identity_by_generation.setdefault(generation,identity)
            if identity!=identity_by_generation[generation]:wrong.append((r['sim_time'],generation,identity,score))
    for label,lo,hi in [('all',0,30),('turn_after_occlusion',9,12)]:
        subset=[x for x in results if lo<=x['sim_time']<hi];miss=[];counts=collections.Counter()
        for r in subset:
            g=gt[r['frame']];target=g['targets'][0]
            if not r['visible']:
                if not target['gt_center_occluded']:miss.append(round(r['sim_time'],3))
                counts['miss']+=1;continue
            score,identity=max((iou(r['bbox'],t.get('gt_bbox_clipped_normalized')),t['id']) for t in g['targets'])
            counts[identity if score>=.3 else 'partial_or_unknown']+=1
        metrics[label]=dict(observations=len(subset),counts=dict(counts),unoccluded_miss_times=miss)
    summary=dict(source=str(source.resolve()),metrics=metrics,wrong_identities=wrong,identities_by_generation=identity_by_generation,
        source_counts=dict(collections.Counter(r['association']['observation_source'] for r in results)),
        generations=max(r['association']['target_generation'] for r in results),roi_calls=detector.roi_calls,
        median_perception_ms=float(np.median([r['perception_s'] for r in results]))*1000,
        scope='One prerecorded RGB observation sequence; GT read only after prediction fordiagnostic scoring, never control.')
    (output/'result.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
