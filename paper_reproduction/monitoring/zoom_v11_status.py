"""Read-only live snapshot with Windows read/write/delete sharing; complete JSONL only."""
from pathlib import Path
import ctypes
from ctypes import wintypes
import hashlib
import json
import math
import msvcrt
import os
import psutil
import datetime

ROOT=Path(__file__).resolve().parents[2]
STATE=Path(__file__).parent/'improved-control-broadcast-state.json'


def shared(path):
    api=ctypes.WinDLL('kernel32',use_last_error=True)
    fn=api.CreateFileW
    fn.argtypes=[wintypes.LPCWSTR,wintypes.DWORD,wintypes.DWORD,wintypes.LPVOID,wintypes.DWORD,wintypes.DWORD,wintypes.HANDLE]
    fn.restype=wintypes.HANDLE
    handle=fn(str(path),0x80000000,7,None,3,0x80,None)
    if handle==ctypes.c_void_p(-1).value:raise ctypes.WinError(ctypes.get_last_error())
    fd=msvcrt.open_osfhandle(handle,os.O_RDONLY|os.O_BINARY)
    with os.fdopen(fd,'rb') as stream:return stream.read()


def read(path):return json.loads(shared(path).decode('utf-8-sig'))
def rows(path):
    if not path.exists():return []
    b=shared(path)
    return [json.loads(s) for s in b[:b.rfind(b'\n')+1].splitlines() if s.strip()]


def snapshot():
    state=read(STATE);run=Path(state['active_run'])
    assert run.name=='active-zoom-v11-attempt1'
    config=read(run/'configuration.json');episodes=rows(run/'episodes.jsonl')
    result=dict(checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),run=str(run),
        cursor=state['last_broadcast_round'],status=read(run/'status.json'),
        new_rounds=[{k:r.get(k) for k in ('round','mean_q_loss','mean_parameter_objective','reward_sum','epsilon','steps',
            'network_updates','updates_total','final_area','final_fov','zoom_in_actions','zoom_out_actions')} for r in episodes if r['round']>state['last_broadcast_round']],
        complete_rounds=len(episodes),complete_training_actions=sum(r['steps'] for r in episodes),
        round_metrics_tail=shared(run/'round-metrics.log').decode('utf-8').splitlines()[-2:])
    result['processes']=[]
    for process in psutil.process_iter(['pid','cmdline','name']):
        try:
            args=process.info['cmdline'] or []
            if any(Path(a).name=='train_zoom_control_50mamba.py' for a in args) or 'UnrealEditor' in (process.info['name'] or ''):
                result['processes'].append(process.info)
        except (psutil.NoSuchProcess,psutil.AccessDenied):pass
    result['reports']={}
    for pattern in ('failure.json','value-review.json','policy-review.json','review*.json','integrity-failure.json',
                    'probe-*/report.json','probe-*/progress.json','probe-*/freeze-receipt.json','validation-*/report.json','**/incident.json','**/initial-zone-failure.json'):
        for path in run.glob(pattern):
            content=read(path)
            if 'episodes' in content:content={k:v for k,v in content.items() if k!='episodes'}
            result['reports'][str(path.relative_to(run))]=content
    latest=sorted(run.glob('training/round-*/steps.jsonl'))[-1:]
    latest+=sorted(run.glob('probe-*/seed*/steps.jsonl'),key=lambda p:p.stat().st_mtime)[-1:]
    result['live_cases']=[]
    for path in latest:
        actions=[r for r in rows(path) if not r.get('initial')]
        if not actions:continue
        recent=actions[-200:]
        values=[v for r in recent for v in [r['reward'],*r['input_state'],*r['next_state'],*r['policy_selection']['q_values']]]
        result['live_cases'].append(dict(file=str(path.relative_to(run)),steps=len(actions),
            last_sim_time=actions[-1]['sim_time_s'],updates_before=actions[-1]['updates_before'],
            age_seconds=datetime.datetime.now().timestamp()-path.stat().st_mtime,
            nonfinite=not all(math.isfinite(v) for v in values),
            recent_q_excess_gt1=sum(r['policy_selection']['greedy_q_bound_excess']>1 for r in recent),
            max_actuator_error=max(r['actuator_error_deg'] for r in actions),
            max_duration_error=max(abs(r['duration_s']-.04) for r in actions),
            saturated_actions=sum(r['saturated_nonstay'] for r in actions),
            visible_actions=sum(r['visible'] for r in actions),
            rgb_fallback_actions=sum(r.get('detector_rgb_scale_fallback',False) for r in actions),
            last_area=actions[-1]['area_ratio'],last_fov=actions[-1]['fov'],last_center=actions[-1]['center_error']))
    before=read(run/'frozen-before.json')
    result['frontend_weights_match']=all(hashlib.sha256(shared(Path(p))).hexdigest()==d for p,d in before.items())
    result['sources_match']=all(hashlib.sha256(shared(ROOT/'paper_reproduction'/p)).hexdigest()==d for p,d in config['source_sha256'].items())
    result['sources_match'] &= all(hashlib.sha256(shared(ROOT/p)).hexdigest()==d for p,d in config['root_source_sha256'].items())
    result['protocol_match']=hashlib.sha256(shared(ROOT/'paper_reproduction/control_zoom_v11/protocol.json')).hexdigest()==config['protocol_sha256']
    if (run/'progress.json').exists():
        p=read(run/'progress.json');result['progress']={k:v for k,v in p.items() if k!='last_round'}
    return result


if __name__=='__main__':print(json.dumps(snapshot(),ensure_ascii=True))
