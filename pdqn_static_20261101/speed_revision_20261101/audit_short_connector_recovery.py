"""Read-only post-run comparison; no inference, control or repainting."""
import argparse,json
from pathlib import Path
import numpy as np
from audit_continuous_online import read,summarize,iou

ROOT=Path(__file__).resolve().parent
BASE=ROOT/'unified_motion_20261101'

def phases(folder):
    rows=read(folder/'perception-online.jsonl');result={}
    for name,a,b in [('turn',8.,12.),('approach',0.,7.5),('bridge',15.,23.),('parallel',38.,42.)]:
        selected=[r for r in rows if a<=r['sim_time']<b]
        result[name]=dict(actual_outputs=len(selected),output_hz=len(selected)/(b-a),
            missing_outputs=sum(not r['visible'] for r in selected),
            detector_median_ms=float(np.median([r['detector_s'] for r in selected]))*1000,
            reid_association_median_ms=float(np.median([r['reid_s'] for r in selected]))*1000)
    return result

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--folder',default='user-short-connector-owned-identity-v9-video');args=parser.parse_args()
    old=BASE/'user-original-then-parallel-viaduct-20mps-v4-video'
    new=BASE/args.folder
    quality=summarize(new);rows=read(new/'perception-online.jsonl')
    gt=read(new/'offline-diagnostics.jsonl');scene=json.loads((new/'scene.json').read_text())
    switches=[];previous={};identity_by_generation={}
    for r in rows:
        if not r['visible']:continue
        match,identity=max((iou(r['bbox'],t.get('gt_bbox_clipped_normalized')),t['id']) for t in gt[r['source_frame']]['targets'])
        if match<.3:continue
        generation=r['rgb_association']['target_generation']
        identity_by_generation.setdefault(str(generation),{}).setdefault(identity,0)
        identity_by_generation[str(generation)][identity]+=1
        if generation in previous and previous[generation]!=identity:
            switches.append(dict(sim_time=r['sim_time'],generation=generation,from_identity=previous[generation],to_identity=identity,
                detector_confidence=r['confidence'],reid_similarity=r['rgb_association']['reid_similarity']))
        previous[generation]=identity
    first_old=len((old/'offline-diagnostics.jsonl').open().readline().encode())
    first_new=len((new/'offline-diagnostics.jsonl').open().readline().encode())
    parallel=[];speeds=[]
    for a,b in zip(gt,gt[1:]):
        if 38.05<a['sim_time']<b['sim_time']<41.95:
            cars={t['id']:t['location_cm'] for t in a['targets']}
            xs=[cars[k][0] for k in ['car_0','car_red_close_follower','car_purple_following']]
            parallel.append(max(xs)-min(xs))
            p=a['targets'][0]['location_cm'];q=b['targets'][0]['location_cm']
            speeds.append(float(np.linalg.norm(np.array(q)-p)/(b['sim_time']-a['sim_time'])/100.))
    assert parallel and max(parallel)<.01
    assert abs(float(np.median(speeds))-20.)<.02
    original=json.loads((BASE/'user-startup-active-zoom-v1-video/scene.json').read_text())
    for old_car,new_car in zip(original['targets'],scene['targets']):
        if old_car['id'] in ['car_0','car_red_close_follower','car_purple_following']:
            assert old_car['path']['waypoints_cm'][:-1]==new_car['path']['waypoints_cm'][:len(old_car['path']['waypoints_cm'])-1]
        else:assert old_car==new_car
    report=dict(turn_exit_connector_m=scene['motion_schedule']['turn_exit_connector_m'],original_connector_m=100.,
        original_approach_rounded_turn_and_other_traffic_preserved=True,
        actual_parallel_speed_median_m_s=float(np.median(speeds)),parallel_x_spread_max_cm=max(parallel),
        before_phases=phases(old),after_phases=phases(new),quality_and_online_proof=quality,
        physical_identity_counts_by_generation=identity_by_generation,same_generation_identity_transitions=switches,
        native_metadata_bytes_before=first_old,native_metadata_bytes_after=first_new,
        configuration_preserved_at_reset=(new/'initial-native-state.json').is_file(),
        resource_context='Concurrent PBZ game was observed at98% aggregate GPU utilization after our UE exited; user paused the game and the pre-v9 GPU measurement dropped to1%. Earlier busy-GPU runs are not a controlled throughput comparison.',
        provenance='Fresh actual online processing only. Ground truth used by this post-run scoring script, never live control or output drawing.',
        preserved_attempts=['v5: improved ownership but motor lag/over-wide search lost target at35s.','v6: readiness RPC timeout before live run.','v7/v8: concurrent external GPU load invalidated throughput comparison. Static-batch variant reverted; v8 fixed within-generation switches but had a legitimate timeout/reselection under low FPS.'])
    (new/'short-connector-and-identity-result.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
