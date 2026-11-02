"""Post-run diagnostics only; never used by live perception or control."""
import collections, hashlib, json
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent

def read(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line]

def iou(a,b):
    if not a or not b:return 0.
    intersection=max(0.,min(a[2],b[2])-max(a[0],b[0]))*max(0.,min(a[3],b[3])-max(a[1],b[1]))
    return intersection/max(1e-12,(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-intersection)

def summarize(folder):
    outputs=read(folder/'online-output/online-results.jsonl');normal=outputs[1:]
    decisions=read(folder/'decisions.jsonl');gt=read(folder/'offline-diagnostics.jsonl')
    config=json.loads((folder/'configuration.json').read_text())
    audit=json.loads((folder/'online-output/online-recording-audit.json').read_text())
    image=cv2.imread(str(folder/'online-output/frames/00000.jpg'))
    h,w=image.shape[:2]
    matches=collections.Counter();miss=[];distances=[];unknown=[]
    for output in normal:
        r=output['result'];frame=output['source_frame'];diagnostic=gt[frame]
        assert abs(diagnostic['sim_time']-r['sim_time'])<1e-6
        assert hashlib.sha256((folder/f'online-output/frames/{output["frame"]:05d}.jpg').read_bytes()).hexdigest()==output['rendered_jpeg_sha256']
        if not r['visible']:
            miss.append(r['sim_time']);continue
        overlap,identity=max((iou(r['bbox'],target.get('gt_bbox_clipped_normalized')),target['id']) for target in diagnostic['targets'])
        if overlap>=.3:matches[identity]+=1
        else:unknown.append(dict(sim_time=r['sim_time'],iou=overlap,best_identity=identity))
        if 2<r['sim_time']<23:
            box=r['bbox'];distances.append(float(np.hypot(((box[0]+box[2])/2-.5)*w,((box[1]+box[3])/2-.5)*h)))
    ids=[o['source_frame'] for o in normal]
    assert all(a<b for a,b in zip(ids,ids[1:]))
    handoff=True
    if (folder/'perception-online.jsonl').exists():
        perception=read(folder/'perception-online.jsonl')
        assert len(perception)==len(normal)
        for p,o in zip(perception,normal):
            assert p['source_frame']==o['source_frame']
            for key in ['bbox','confidence','visible','sim_time']:assert p[key]==o['result'][key]
        for decision in decisions:
            p=perception[decision['perception_result_index']]
            assert decision['observation_frame']==p['source_frame']
            for key in ['bbox','confidence','visible','sim_time']:assert decision[key]==p[key]
            event=decision['command_execution']
            assert p['sim_time']>=event['native_sim_s']+event['duration_s']-1e-5
            assert p['source_action_remaining_s']<=1e-6
    else:handoff=None
    return dict(folder=str(folder),duration_s=normal[-1]['wall_s'],online_updates=len(normal),
        actual_online_output_fps=len(normal)/normal[-1]['wall_s'],
        source_to_draw_median_ms=float(np.median([o['latency_at_draw_ms'] for o in normal])),
        source_to_published_median_ms=float(np.median([o['latency_at_draw_ms']+o['draw_and_write_ms'] for o in normal])),
        control_decisions=len(decisions),control_decisions_per_s=len(decisions)/normal[-1]['wall_s'],
        visible_identity_matches=dict(matches),unmatched_visible=unknown,missing_outputs=len(miss),
        turn_9_to_12s_missing_outputs=sum(9<=t<=12 for t in miss),
        maximum_target_generation=max(o['result']['target_generation'] for o in normal),
        moving_center_distance_median_p95_px=np.percentile(distances,[50,95]).tolist(),
        actual_result_handoff_and_completed_action_gates_verified=handoff,
        strictly_increasing_unique_source_frames=True,saved_jpeg_hashes_verified=True,
        all_pixels_painted_before_live_stopped=audit.get('all_pixels_written_before_live_stop'),
        all_learning_state_frozen=config['all_learning_state_frozen'],
        checkpoint_sha256=config['checkpoint_sha256'],
        no_postrun_inference_or_repainting=True,GT_usage='Post-run diagnostic scoring only')

if __name__=='__main__':
    base=ROOT/'unified_motion_20261101'
    old=summarize(base/'user-genuine-online-results-video')
    new=summarize(base/'user-fast-genuine-online-v2-video')
    assert old['checkpoint_sha256']==new['checkpoint_sha256']
    attempts={name:summarize(base/name) for name in ['user-fast-genuine-online-v1-video','user-fast-genuine-online-v3-video','user-fast-genuine-online-v4-video','user-fast-genuine-online-v5-video']}
    result=dict(baseline=old,optimized=new,fps_speedup=new['actual_online_output_fps']/old['actual_online_output_fps'],
        preserved_other_attempts=attempts,
        selection_reason='Select v2 FP32 capture30 continuous perception: original identity retained in this focused live scene. Faster v3/v4/v5 lost the original identity long enough to trigger the configured 3s timeout. These attempts are preserved and rejected.',
        limitation='One scene only. Throughput improved; source-to-published latency and exclusive control decision rate did not improve. No broad reliability claim.')
    target=base/'online-throughput-improvement.json'
    target.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))
