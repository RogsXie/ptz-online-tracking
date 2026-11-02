"""Bounded heldout RGB detector check; GT accessed only for post-prediction scoring."""
import speed_bootstrap
import json
from pathlib import Path
import cv2, numpy as np, torch
from whole_vehicle_frontend import WholeVehicleMultiScale, CHECKPOINT
from integration.metrics import iou

def main():
    torch.set_num_threads(4)
    root = CHECKPOINT.parent
    source = root.parent/'user-crossroad-reacquire-area2-video'
    detector = WholeVehicleMultiScale(CHECKPOINT)
    frames = [json.loads(s) for s in (source/'capture-timestamps.jsonl').read_text().splitlines()]
    indices = [i for i,f in enumerate(frames) if i%15==0 and 3.<=f['sim_time']<=8.5]
    example = min(range(len(frames)),key=lambda i:abs(frames[i]['sim_time']-7.))
    predictions = {}
    for i in sorted(set(indices+[example])):
        bgr = cv2.imread(str(source/f'frames/{i:05d}.jpg'))
        detections = detector(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB))
        predictions[i] = [d.as_dict() for d in detections]
        if i==example:
            for d in detections:
                a,b,c,e = [int(v*z) for v,z in zip(d.bbox,[960,540,960,540])]
                color = (0,235,255) if d.label=='truck' else (60,235,60)
                cv2.rectangle(bgr,(a,b),(c,e),color,2,cv2.LINE_AA)
                cv2.putText(bgr,f'{d.label} {d.confidence:.2f}',(a,max(42,b-8)),cv2.FONT_HERSHEY_SIMPLEX,.55,color,2,cv2.LINE_AA)
            cv2.imwrite(str(root/'whole-truck-detection-example.jpg'),bgr)
    # Diagnostic visibility/boxes never participate in predictions above.
    gt = [json.loads(s) for s in (source/'offline-diagnostics.jsonl').read_text().splitlines()]
    scores = []
    for i in indices:
        t = gt[i]['targets'][-1]
        if not t['gt_in_frame'] or t['gt_center_occluded']:
            continue
        candidates = [d for d in predictions[i] if d['label']=='truck']
        best = max([iou(d['bbox'],t['gt_bbox_clipped_normalized']) for d in candidates],default=0.)
        scores.append(dict(frame=i,sim_time=frames[i]['sim_time'],whole_truck_iou=best))
    result = dict(heldout_recording=str(source),scored_frames=len(scores),
        whole_truck_iou_mean=float(np.mean([s['whole_truck_iou'] for s in scores])),
        whole_truck_iou_ge_05_fraction=float(np.mean([s['whole_truck_iou']>=.5 for s in scores])),
        detector_sha256=detector.provenance['sha256'],scores=scores,
        scope='Focused heldout check at3..8.5s; predictions use RGB only. No train/val frame overlap. Not broad qualification.')
    (root/'focused-whole-truck-check.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result))

if __name__=='__main__':main()
