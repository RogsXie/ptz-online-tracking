"""Post-run segment scoring; includes native images with no selected box."""
import argparse,json
from pathlib import Path
import numpy as np
from audit_continuous_online import summarize
from evaluate_camera_control_only import camera


def score(folder):
    frames=[json.loads(x) for x in (folder/'offline-diagnostics.jsonl').read_text().splitlines()]
    values=[]
    for frame in frames:
        target=next(t for t in frame['targets'] if t['id']=='car_0');box=target.get('gt_bbox_normalized')
        if box is None:continue
        values.append((frame['sim_time'],float(np.hypot((box[0]+box[2]-1)*480.,(box[1]+box[3]-1)*270.))))
    segments={}
    for name,lo,hi in [('initial_acquisition',0.,.75),('early_075_to_2',.75,2.),('early_2_to_4',2.,4.),
        ('early_4_to_6',4.,6.),('early_075_to_6',.75,6.),('occlusion_6_to_9',6.,9.),
        ('turn_9_to_12',9.,12.),('fast_12_to_end',12.,28.)]:
        subset=[e for t,e in values if lo<=t<hi]
        segments[name]=dict(native_frames=len(subset),median_p95_px=np.percentile(subset,[50,95]).tolist(),
            within_15px_fraction=float(np.mean(np.asarray(subset)<=15.)))
    first_settle=next((values[i][0] for i in range(len(values)-5) if all(e<=20. for _,e in values[i:i+6])),None)
    online=summarize(folder);control=camera(folder)
    (folder/'online-audit.json').write_text(json.dumps(online,indent=2),encoding='utf-8')
    (folder/'camera-audit.json').write_text(json.dumps(control,indent=2),encoding='utf-8')
    return dict(folder=str(folder.resolve()),segments=segments,first_six_source_frames_within20px_at_s=first_settle,
        online=online,camera=control,scope='GT read post-run only. All source images including missing/occlusion; same original car and identical scene.')


def main():
    p=argparse.ArgumentParser();p.add_argument('--baseline',type=Path,required=True);p.add_argument('--candidate',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    result=dict(baseline=score(args.baseline),candidate=score(args.candidate))
    assert json.loads((args.baseline/'scene.json').read_text())==json.loads((args.candidate/'scene.json').read_text())
    result['scene_configuration_identical']=True
    args.output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    for name in result['baseline']['segments']:
        print(name,result['baseline']['segments'][name]['median_p95_px'],'->',result['candidate']['segments'][name]['median_p95_px'])
    print('FIRST SETTLE',result['baseline']['first_six_source_frames_within20px_at_s'],'->',result['candidate']['first_six_source_frames_within20px_at_s'])
    print('ONLINE',result['candidate']['online'])


if __name__=='__main__':main()
