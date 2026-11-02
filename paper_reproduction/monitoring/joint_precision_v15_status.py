"""Shared read-only v15 monitor; never labels model fitting as Q learning."""
import datetime,json,math,hashlib
from pathlib import Path
import psutil
from center_v12a_status import shared,read,rows,STATE,ROOT

def snapshot():
    st=read(STATE);run=Path(st['active_run']);assert run.name=='joint-precision-zoom-v15-attempt1'
    cfg=read(run/'configuration.json');episodes=rows(run/'episodes.jsonl')
    result=dict(checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),run=str(run),cursor=st['last_broadcast_round'],
        status=read(run/'status.json') if (run/'status.json').exists() else {},complete_rounds=len(episodes),
        complete_training_actions=sum(x['steps'] for x in episodes),
        new_rounds=[{k:x[k] for k in ['round','fit_mse_xyz_mean','reward_sum','epsilon','steps','model_updates','updates_total','action_counts','final_area','final_center_error']} for x in episodes if x['round']>st['last_broadcast_round']],processes=[])
    for p in psutil.process_iter(['pid','name','cmdline']):
        try:
            if any(Path(a).name=='train_joint_precision_zoom_50mamba.py' for a in (p.info['cmdline'] or [])) or 'UnrealEditor' in (p.info['name'] or ''):result['processes'].append(p.info)
        except (psutil.NoSuchProcess,psutil.AccessDenied):pass
    result['reports']={}
    for pat in ['failure.json','integrity-after.json','preflight-progress.json','preflight-report.json','development-report.json','probe-*/report.json','probe-*/progress.json','probe-*/freeze-receipt.json','**/incident.json']:
        for p in run.glob(pat):
            v=read(p);result['reports'][str(p.relative_to(run))]={k:x for k,x in v.items() if k not in ['episodes','initials','last_case']}
    result['live_cases']=[]
    for pat in ['training/round-*/steps.jsonl','probe-*/seed*/steps.jsonl']:
        for p in sorted(run.glob(pat),key=lambda x:x.stat().st_mtime)[-1:]:
            rr=rows(p);actions=rr[1:]
            if not actions:continue
            values=[v for x in actions for v in [x['reward'],*(x['model_state'] or [])]]
            result['live_cases'].append(dict(path=str(p.relative_to(run)),actions=len(actions),last_time=actions[-1]['sim_time_s'],
                file_age_seconds=datetime.datetime.now().timestamp()-p.stat().st_mtime,all_finite=all(math.isfinite(v) for v in values),
                visible=sum(x['visible'] for x in actions),matched=sum(x['matched'] for x in actions),
                max_actuator_error=max(x['actuator_error_deg'] for x in actions),saturated=sum(x['saturated_nonstay'] for x in actions),
                action_counts={str(a):sum(x['action_type']==a for x in actions) for a in range(7)},
                last_center=actions[-1]['center_error'],last_area=actions[-1]['area_ratio'],last_fov=actions[-1]['fov']))
    before=read(run/'frozen-before.json');result['frontend_match']=all(hashlib.sha256(shared(Path(p))).hexdigest()==h for p,h in before.items())
    result['source_match']=all(hashlib.sha256(shared(ROOT/'paper_reproduction'/p)).hexdigest()==h for p,h in cfg['source_sha256'].items()) and all(hashlib.sha256(shared(ROOT/p)).hexdigest()==h for p,h in cfg['root_source_sha256'].items())
    result['protocol_match']=hashlib.sha256(shared(ROOT/'paper_reproduction/control_precision_v15/protocol.json')).hexdigest()==cfg['protocol_sha256']
    return result
if __name__=='__main__':print(json.dumps(snapshot(),ensure_ascii=True))
