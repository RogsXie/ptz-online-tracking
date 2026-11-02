"""Gated serial speed comparison; never promotes an initialization-only model."""
from pathlib import Path
import argparse,hashlib,json,subprocess,time,traceback
import numpy as np,psutil

ROOT=Path(__file__).resolve().parent
PYTHON=Path('C:/Users/12879/.conda/envs/50mamba/python.exe')
STATUS=ROOT/'qualification-status.json'

def load(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(phase,**detail):STATUS.write_text(json.dumps(dict(phase=phase,utc_seconds=time.time(),**detail),indent=2),encoding='utf-8')
def run(entry,*args):subprocess.run([str(PYTHON),'-B',str(ROOT/entry),*map(str,args)],check=True,creationflags=subprocess.CREATE_NO_WINDOW)
def freeze_ok():
    for p,sha in load(ROOT/'source-freeze-training-v1/manifest.json').items():
        assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==sha,f'Frozen source changed: {p}'
    if (ROOT/'QUEUE_STOP_REQUEST').exists():raise InterruptedError('Requested process-boundary stop')

def speed_metrics(report,baseline):
    rows=report['episodes'];old=baseline['episodes'];assert {r['seed'] for r in rows}=={r['seed'] for r in old}
    times=lambda a:[r['settling_time'] if r['passed'] else 50. for r in a]
    t0,t1=times(old),times(rows);med=float(np.median(t1));p95=float(np.quantile(t1,.95))
    zoom=[r['first_zoom_s'] for r in rows if r['needs_zoom'] and r['first_zoom_s'] is not None]
    outer=[r['first_zoom_s'] for r in rows if r['needs_zoom'] and r['layer']>=3 and r['first_zoom_s'] is not None]
    checks=dict(quality=report['qualification']['qualified'],median=med<=.60*np.median(t0),
        p95=p95<=.75*np.quantile(t0,.95),first_zoom=bool(zoom) and np.median(zoom)<=5.,
        outer_first_zoom=bool(outer) and np.median(outer)<=5.)
    return dict(qualified=all(checks.values()),checks={k:bool(v) for k,v in checks.items()},
        count=len(rows),median_settling_sim_s=med,p95_settling_sim_s=p95,
        baseline_median_settling_sim_s=float(np.median(t0)),baseline_p95_settling_sim_s=float(np.quantile(t0,.95)),
        median_ratio=float(med/np.median(t0)),p95_ratio=float(p95/np.quantile(t0,.95)),
        first_zoom_median_sim_s=float(np.median(zoom)) if zoom else None,
        outer_first_zoom_median_sim_s=float(np.median(outer)) if outer else None,
        failures_penalized_at_s=50,clock_scope='Synchronous UE action time; real wall speed still requires new realtime clips')

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--owner-pid',type=int,required=True);args=p.parse_args()
    write('waiting_for_owned_training',owned_python_pid=args.owner_pid)
    while psutil.pid_exists(args.owner_pid):
        try:
            if 'train_speed.py' not in ' '.join(psutil.Process(args.owner_pid).cmdline()):raise RuntimeError('Owner PID reused')
        except psutil.NoSuchProcess:break
        if (ROOT/'QUEUE_STOP_REQUEST').exists():raise InterruptedError('Stop requested')
        time.sleep(1)
    status=load(ROOT/'training-v1/status.json');assert status['completed']==160
    freeze_ok();old_checkpoint=ROOT.parent/'exploration-support-original-updates/updates-1024.pt'
    baseline=ROOT/'released-baseline-development';write('paired_released_baseline_development')
    run('evaluate_speed.py','--checkpoint',old_checkpoint,'--output',baseline,'--old-baseline','--count',80)
    freeze_ok();baseline_report=load(baseline/'cases/independent-results.json')
    candidates=[]
    for number in (40,80,160):
        folder=ROOT/f'training-v1/development-{number:04d}';r=load(folder/'independent-results.json')
        metrics=speed_metrics(r,baseline_report)
        candidates.append(dict(episodes=number,checkpoint=str(ROOT/f'training-v1/checkpoint-{number:04d}.pt'),
            development_report=str(folder/'independent-results.json'),speed=metrics))
    (ROOT/'development-speed-comparison.json').write_text(json.dumps(candidates,indent=2),encoding='utf-8')
    choices=sorted([c for c in candidates if c['speed']['qualified']],key=lambda c:c['speed']['median_settling_sim_s'])
    selected=None
    for candidate in choices:
        n=candidate['episodes'];out=ROOT/f'corners-trained-{n:04d}'
        write('known_corner_quality_audit',candidate=candidate)
        run('evaluate_speed_corners.py','--checkpoint',candidate['checkpoint'],'--output',out,'--count',32)
        freeze_ok();r=load(out/'cases/independent-results.json')
        if r['qualification']['qualified']:
            selected=candidate|dict(corner_report=str(out/'cases/independent-results.json'));break
    if selected is None:
        write('development_not_qualified',candidates=candidates,new_validation_executed=False,new_final_executed=False);return
    oldval=ROOT/'released-baseline-validation';write('paired_released_baseline_validation',selected=selected)
    run('evaluate_speed.py','--checkpoint',old_checkpoint,'--output',oldval,'--old-baseline','--split','validation','--count',160)
    validation=ROOT/'validation-v1';write('new_independent_validation',selected=selected)
    run('evaluate_speed.py','--checkpoint',selected['checkpoint'],'--output',validation,'--split','validation','--count',160)
    freeze_ok();vpath=validation/'cases/independent-results.json';v=load(vpath);vspeed=speed_metrics(v,load(oldval/'cases/independent-results.json'))
    (ROOT/'validation-speed-comparison.json').write_text(json.dumps(vspeed,indent=2),encoding='utf-8')
    if not vspeed['qualified']:
        write('new_validation_not_qualified',selected=selected,validation_report=str(vpath),speed=vspeed,new_final_executed=False);return
    oldfinal=ROOT/'released-baseline-final';write('paired_released_baseline_final',selected=selected)
    run('evaluate_speed.py','--checkpoint',old_checkpoint,'--output',oldfinal,'--old-baseline','--split','final_test','--count',160)
    final=ROOT/'final-test-v1';write('new_independent_final',selected=selected)
    run('evaluate_speed.py','--checkpoint',selected['checkpoint'],'--output',final,'--split','final_test','--count',160,'--validation-report',vpath)
    freeze_ok();fpath=final/'cases/independent-results.json';fspeed=speed_metrics(load(fpath),load(oldfinal/'cases/independent-results.json'))
    (ROOT/'final-speed-comparison.json').write_text(json.dumps(fspeed,indent=2),encoding='utf-8')
    write('independent_tests_complete',selected=selected,validation_report=str(vpath),final_report=str(fpath),
        validation_speed=vspeed,final_speed=fspeed,qualified=fspeed['qualified'],
        remaining='Fresh paired realtime captures, source-protocol contrast, boxed videos, new PyCharm release; not complete')

if __name__=='__main__':
    try:main()
    except BaseException as exc:
        write('error_requires_inspection',error=str(exc),traceback=traceback.format_exc());raise
