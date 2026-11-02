"""Post-run camera scoring, including frames with no selected detection.

Reads saved native diagnostics only AFTER controls/learning have ended. These
target positions/bounds never enter the policy or training reward. Native source
frames, rather than only successful detections, define camera coverage/error.
"""
import argparse,json,math
from pathlib import Path
import numpy as np


def read(path):return [json.loads(x) for x in path.read_text(encoding='utf-8').splitlines() if x]


def camera(folder,target='car_0'):
    diagnostics=read(folder/'offline-diagnostics.jsonl')
    measurements=[];outside=undefined=0
    for d in diagnostics:
        if d['sim_time']<2.:continue
        obj=next(x for x in d['targets'] if x['id']==target)
        outside+=int(not obj['gt_in_frame']);box=obj.get('gt_bbox_normalized')
        if box is None:undefined+=1;continue
        x,y=(box[0]+box[2])/2-.5,(box[1]+box[3])/2-.5
        tangent=math.tan(math.radians(d['fov']/2))
        measurements.append(dict(sim_time=d['sim_time'],pixel_error=float(math.hypot(x*960.,y*540.)),
            angular_error_deg=math.degrees(math.atan(math.hypot(2*x*tangent,2*y*tangent*540/960))),
            x_px=x*960.,y_px=y*540.,fov=d['fov'],geometric_area=(box[2]-box[0])*(box[3]-box[1])))
    outputs=read(folder/'online-output/online-results.jsonl')[1:];missing=[]
    for o in outputs:
        if o['result']['visible']:continue
        d=diagnostics[o['source_frame']];obj=next(x for x in d['targets'] if x['id']==target)
        missing.append(dict(sim_time=d['sim_time'],target_in_view=obj['gt_in_frame'],
            physically_occluded=obj['gt_center_occluded'],occluder=obj.get('gt_occluder')))
    segments={}
    for name,start,end in [('motion_2_to_end',2.,1e9),('turn_9_to_12',9.,12.),('fast_12_to_end',12.,1e9),('parallel_24_to_end',24.,1e9)]:
        rows=[r for r in measurements if start<=r['sim_time']<end]
        if not rows:
            segments[name]=dict(source_frames=0,pixel_median_p95=None,angular_deg_median_p95=None)
            continue
        segments[name]=dict(source_frames=len(rows),pixel_median_p95=np.percentile([r['pixel_error'] for r in rows],[50,95]).tolist(),
            angular_deg_median_p95=np.percentile([r['angular_error_deg'] for r in rows],[50,95]).tolist(),
            image_x_y_std_px=np.std([[r['x_px'],r['y_px']] for r in rows],axis=0).tolist(),
            median_fov_deg=float(np.median([r['fov'] for r in rows])),
            median_geometric_area_percent=float(np.median([r['geometric_area'] for r in rows])*100.))
    decisions=read(folder/'decisions.jsonl');last={};reversals=0;duration_errors=[];agree=total=0
    for row in decisions:
        action=row['action'];guard=row['area_constraint']
        if row['visible'] and guard.get('phase')=='track' and 'pdqn_action' in guard:
            total+=1;agree+=int(action==guard['pdqn_action'])
            # Compare the parameter for the EXECUTED action, not a duration
            # belonging to a different rejected discrete action.
            seconds=(.01,.004,.004,.8/180.,.8/180.,.01,.01)[action]
            predicted=float(row['all_parameters'][action])*seconds
            duration_errors.append(abs(predicted-row['duration'])*1000.)
        if action not in (1,2,3,4):continue
        axis=0 if action in (3,4) else 1;sign=1 if action in (2,3) else -1
        now=row['command_wall_s']
        if axis in last and sign!=last[axis][0] and now-last[axis][1]<.35:reversals+=1
        last[axis]=(sign,now)
    return dict(folder=str(folder),target=target,evaluation_only_GT=True,
        physical_camera_target_out_of_view_frames=outside,undefined_projected_target_frames=undefined,
        camera_segments=segments,no_box_outputs=len(missing),
        no_box_target_still_in_view=sum(r['target_in_view'] for r in missing),
        no_box_physically_occluded=sum(r['physically_occluded'] for r in missing),
        no_box_out_of_view=sum(not r['target_in_view'] for r in missing),no_box_events=missing,
        rapid_opposite_axis_commands_within_point35_s=reversals,
        normal_tracking_raw_action_agreement=[agree,total],
        executed_action_parameter_mean_absolute_correction_ms=float(np.mean(duration_errors)) if duration_errors else None,
        qualification='Native geometric target center scored over ALL source frames after2s, including occlusion/no-box. GT is post-run scoring only, never policy or reward. One focused scene, not general qualification.')


def main():
    p=argparse.ArgumentParser();p.add_argument('--runs',nargs='+',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();result={x.name:camera(x) for x in args.runs}
    args.output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    for name,r in result.items():print(name,json.dumps({k:r[k] for k in ['physical_camera_target_out_of_view_frames','camera_segments','no_box_outputs','no_box_target_still_in_view','no_box_physically_occluded','normal_tracking_raw_action_agreement','executed_action_parameter_mean_absolute_correction_ms']}))


if __name__=='__main__':main()
