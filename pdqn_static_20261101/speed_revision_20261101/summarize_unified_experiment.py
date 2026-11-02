"""Post-run evidence only. Never imported by control or training."""
import speed_bootstrap
import json,math,hashlib
from pathlib import Path
from collections import Counter
import numpy as np
from unified_core import ROOT
from integration.metrics import iou

def lines(p):return [json.loads(s) for s in p.read_text(encoding='utf-8').splitlines()]

def summarize(folder):
    cfg=json.loads((folder/'configuration.json').read_text());ds=lines(folder/'detection-overlay/detections.jsonl');gs=lines(folder/'offline-diagnostics.jsonl');frames=lines(folder/'capture-timestamps.jsonl')
    groups={};first=None;speeds=[]
    for n,(d,g,f) in enumerate(zip(ds,gs,frames)):
        match=bool(d['bbox'] and g.get('gt_bbox_normalized') and iou(d['bbox'],g['gt_bbox_normalized'])>=.3)
        mult=1.
        for boundary,value in cfg['motion']['speed_profile']:
            if g['sim_time']>=boundary:mult=value
        moving=cfg['motion']['speed_m_s']*mult>0
        names=['all','moving' if moving else 'stopped']
        if 10<=g['sim_time']<18:names.append('10to18s')
        if g['sim_time']>=25:names.append('last5s')
        if d['bbox']:
            b=d['bbox'];area=(b[2]-b[0])*(b[3]-b[1]);px=math.hypot(((b[0]+b[2])/2-.5)*960,((b[1]+b[3])/2-.5)*540)
            if first is None and match and d['centered'] and .025<=area<=.055 and abs(math.log(area/.04))<=.25:first=f['wall_s']
        for name in names:
            v=groups.setdefault(name,dict(frames=0,matched=0,gt_in_frame=0,yellow=0,pixels=[]))
            v['frames']+=1;v['gt_in_frame']+=bool(g.get('gt_in_frame'));v['matched']+=match;v['yellow']+=bool(match and d['centered'])
            if match:v['pixels'].append(px)
        if n:
            prev=gs[n-1];dt=g['sim_time']-prev['sim_time']
            if dt>1e-6:speeds.append((g['sim_time'],float(np.linalg.norm(np.asarray(g['target_location'])-prev['target_location']))/dt/100))
    for v in groups.values():
        p=v.pop('pixels');v.update(retention_fraction=v['matched']/v['frames'],yellow_fraction=v['yellow']/v['frames'],matched_mean_center_px=float(np.mean(p)) if p else None,matched_p95_center_px=float(np.percentile(p,95)) if p else None)
    decisions=lines(folder/'decisions.jsonl')
    result=dict(folder=str(folder),video=str(folder/'detection-overlay/live-detection-vfr.mp4'),architecture=cfg['architecture'],scenario=cfg['motion'],groups=groups,
        first_joint_geometry_wall_s=first,first_joint_scope='Instantaneous RGB center<=.025, area.025..055 and abslog(area/.04)<=.25, independently GT-matched. Not a5sStay success.',
        mean_measured_speed_2to16_m_s=float(np.mean([v for t,v in speeds if 2<t<16])) if speeds else None,
        mean_measured_speed_after25_m_s=float(np.mean([v for t,v in speeds if t>25])) if speeds else None,
        decisions=len(decisions),action_counts=dict(Counter(d['action'] for d in decisions)),frozen=cfg['all_learning_state_frozen'],checkpoint_sha256=cfg['checkpoint_sha256'])
    (folder/'focused-results.json').write_text(json.dumps(result,indent=2),encoding='utf-8');return result

def main():
    results=[]
    for folder in sorted(ROOT.glob('live-*')):
        if (folder/'detection-overlay/provenance-and-audit.json').exists():results.append(summarize(folder))
    train=ROOT/'pdqn_training_v1'
    report=dict(training=json.loads((train/'status.json').read_text()),configuration=str(train/'configuration.json'),detector_training=dict(train_images=930,val_images=81,epochs=5,GT_used_only_for_detector_labels=True),
        single_unified_policy=True,original_update_math_preserved=True,no_PID_MPC_teacher_or_GT_control=True,focused_runs=results,
        qualification='Bounded mixed-training experiment and focused demonstrations, not statistical qualification. Old releases and failed experiments preserved.')
    (ROOT/'experiment-report.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
