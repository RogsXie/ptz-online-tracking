"""Capture the eight declared realtime examples after all independent tests."""
import argparse,hashlib,json,subprocess,time,traceback
from pathlib import Path
import psutil

ROOT=Path(__file__).resolve().parent
PYTHON=Path('C:/Users/12879/.conda/envs/50mamba/python.exe')
STATUS=ROOT/'realtime-evidence-status.json'

def load(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def write(phase,**detail):STATUS.write_text(json.dumps(dict(phase=phase,utc_seconds=time.time(),**detail),indent=2),encoding='utf-8')
def run(entry,*args,check=True):
    return subprocess.run([str(PYTHON),'-B',str(ROOT/entry),*map(str,args)],check=check,creationflags=subprocess.CREATE_NO_WINDOW)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--owner-pid',type=int,required=True);args=p.parse_args()
    write('waiting_for_qualification_coordinator',owned_python_pid=args.owner_pid)
    while psutil.pid_exists(args.owner_pid):
        try:
            if 'qualify_after_exploration.py' not in ' '.join(psutil.Process(args.owner_pid).cmdline()):raise RuntimeError('Owner PID reused')
        except psutil.NoSuchProcess:break
        if (ROOT/'LIVE_QUEUE_STOP').exists():raise InterruptedError('Stop requested at process boundary')
        time.sleep(1)
    status=load(ROOT/'exploration-qualification-status.json')
    if status['phase']!='independent_tests_complete' or not status['final_test_qualified']:
        write('qualification_not_ready',qualification_status=status,no_capture_launched=True);return
    checkpoint=Path(status['selected']['checkpoint']);digest=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    for key in ('validation_report','final_test_report'):
        path=Path(status[key]);r=load(path)
        assert r['qualification']['qualified'] and r['qualification']['count']==160
        assert r['configuration']['checkpoint_sha256']==digest
        assert load(path.parent/'frozen-verification.json')['networks_targets_optimizers_replay_rng_unchanged']
    frozen=load(ROOT/'exploration-qualification-source-freeze.json')
    for path,sha in frozen.items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==sha
    plan=load(ROOT/'realtime-evidence-plan.json');assert len(plan['cases'])==8
    assert all(sum(c['quadrant']==i for c in plan['cases'])==2 for i in range(4))
    assert {c['scale_group'] for c in plan['cases']}=={0,1,2,3}
    output=ROOT/'release-live-evidence';output.mkdir(exist_ok=False)
    manifest={str(ROOT/n):hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in
        ('live_demo.py','audit_live.py','audit_release_live.py','plot_release_live.py','realtime-evidence-plan.json')}
    manifest.update(frozen)
    (output/'source-freeze.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    (output/'plan.json').write_text(json.dumps(plan,indent=2),encoding='utf-8')
    results=[]
    for i,case in enumerate(plan['cases']):
        if (ROOT/'LIVE_QUEUE_STOP').exists():raise InterruptedError('Stop requested at process boundary')
        folder=output/case['name'];write('capturing_realtime',case_number=i+1,total=8,case=case,completed=results)
        run('live_demo.py','--checkpoint',checkpoint,'--output',folder,'--index',case['index'],
            '--split',case['split'],'--seconds',plan['seconds_each'])
        run('audit_live.py',folder)
        audited=run('audit_release_live.py',folder,check=False)
        audit_path=folder/'independent-release-live-results.json'
        if not audit_path.exists():raise RuntimeError('Realtime auditor did not finish; preserve files and inspect')
        audit=load(audit_path)
        assert audit['case']['quadrant']==case['quadrant'] and audit['case']['scale_group']==case['scale_group']
        assert audited.returncode==0 or not audit['passed']
        run('plot_release_live.py',folder)
        results.append(dict(case=case,passed=audit['passed'],folder=str(folder),audit=str(audit_path),video=str(folder/'live-sensor-vfr.mp4')))
        for path,sha in manifest.items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==sha
        (output/'results.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
    zoom_in=sum(load(r['audit'])['action_counts']['5'] for r in results)
    zoom_out=sum(load(r['audit'])['action_counts']['6'] for r in results)
    qualified=all(r['passed'] for r in results) and zoom_in>0 and zoom_out>0
    write('realtime_examples_complete',qualified=qualified,results=results,checkpoint=str(checkpoint),
        checkpoint_sha256=digest,validation_report=status['validation_report'],final_test_report=status['final_test_report'],
        actual_zoom_in_decisions=zoom_in,actual_zoom_out_decisions=zoom_out,goal_complete=False,
        remaining='Review captured figures, complete frozen-decision audits and report, create release marker and test visible PyCharm entry')

if __name__=='__main__':
    try:main()
    except BaseException as exc:
        write('error_requires_inspection',error=str(exc),traceback=traceback.format_exc());raise
