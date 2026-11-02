"""Independent file-only audit of captured real-time control and geometry."""
import argparse,json,math
from pathlib import Path
from independent_validation import iou,weighted_quantile

def lines(path):
    return [json.loads(x) for x in path.read_text(encoding='utf-8').splitlines()]

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('folder',type=Path);args=p.parse_args();root=args.folder
    frames=lines(root/'capture-timestamps.jsonl');diagnostics=lines(root/'offline-diagnostics.jsonl');rows=lines(root/'decisions.jsonl')
    assert len(frames)==len(diagnostics)
    assert all(abs(f['sim_time']-d['sim_time'])<1e-8 for f,d in zip(frames,diagnostics))
    end=rows[-1]['sim_time'];errors=[];scales=[];areas=[];hold=[];matched=visible=total=0.;before=frames[0]['sim_time']
    matched_decisions=[];valid_geometry=True
    for row in rows:
        duration=row['sim_time']-before
        assert duration>0
        frame_index=min(range(len(frames)),key=lambda i:abs(frames[i]['sim_time']-row['sim_time']))
        assert abs(frames[frame_index]['sim_time']-row['sim_time'])<1e-8
        gt=diagnostics[frame_index]['gt_bbox_normalized'];box=row['bbox']
        hit=bool(box and gt and iou(box,gt)>=.3)
        matched_decisions.append(dict(sim_time=row['sim_time'],frame=frame_index,correct_target=hit,iou=iou(box,gt) if box and gt else None))
        total+=duration;visible+=duration*row['visible'];matched+=duration*hit
        if row['visible']:
            x,y=box[0]+box[2]-1,box[1]+box[3]-1
            valid_geometry &= abs(row['center_error']-math.hypot(x,y))<1e-5
            valid_geometry &= abs(row['area']-(box[2]-box[0])*(box[3]-box[1]))<1e-5
        overlap=max(0.,min(end,row['sim_time'])-max(end-5.,before));before=row['sim_time']
        if overlap:
            hold.append((row,overlap));areas.append((row['area'],overlap))
            if row['visible']:
                errors.append((row['center_error'],overlap));scales.append((abs(math.log(row['area']/.04)),overlap))
    avg=lambda a:sum(v*w for v,w in a)/sum(w for v,w in a) if a else None
    mean,p95,scale=avg(errors),weighted_quantile(errors,.95),avg(scales)
    target_positions=[tuple(d['target_location']) for d in diagnostics]
    camera_positions=[tuple(d['camera_location']) for d in diagnostics]
    checks=dict(timestamp_pairing=True,geometry=valid_geometry,
        stationary_target=len(set(target_positions))==1,
        fixed_camera_installation=len(set(camera_positions))==1,
        actual_ue_realtime=all(d['realtime_control'] and not d['demo_geometry_follow'] and not d['free_preview'] for d in diagnostics),
        five_second_hold=sum(w for r,w in hold)>=5.-1e-5,
        all_stay=all(r['action']==0 for r,w in hold),all_visible=all(r['visible'] for r,w in hold),
        center_mean=mean is not None and mean<=.025,center_p95=p95 is not None and p95<=.05,
        logscale=scale is not None and scale<=.25,
        area_band=all(r['area'] is not None and .025<=r['area']<=.055 for r,w in hold),
        selected_availability=visible/total>=.99,correct_target_availability=matched/total>=.98,
        no_gt_camera_reset=all(not d['legacy_target_center_reset'] for d in diagnostics))
    result=dict(passed=all(checks.values()),checks=checks,center_mean=mean,center_p95=p95,scale_log_mean=scale,
        area_mean=avg(areas),hold_start_s=end-5.,selected_availability=visible/total,matched_availability=matched/total,
        captured_frames=len(frames),decisions=len(rows),simulation_duration_s=end-frames[0]['sim_time'],
        wall_duration_s=frames[-1]['wall_s']-frames[0]['wall_s'],
        raw_frame_gt_visible_fraction=sum(bool(d['gt_in_frame']) for d in diagnostics)/len(diagnostics),
        matching=matched_decisions,
        scope='One known development case, freshly captured real-time. No claim of independent multi-case qualification; perception measured at action ends, not every video frame.')
    (root/'independent-live-results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='matching'},indent=2))

if __name__=='__main__':main()
