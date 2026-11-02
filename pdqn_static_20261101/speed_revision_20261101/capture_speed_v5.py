"""Serial source-protocol contrast and predeclared paired real-wall evidence."""
import argparse,hashlib,json,subprocess,time,traceback
from pathlib import Path
import psutil,numpy as np
ROOT=Path(__file__).resolve().parent;OLD=ROOT.parent;PY=Path('C:/Users/12879/.conda/envs/50mamba/python.exe');STATUS=ROOT/'speed-live-v5-status.json'
def load(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(phase,**v):STATUS.write_text(json.dumps(dict(phase=phase,utc=time.time(),**v),indent=2),encoding='utf-8')
def run(p,*args,check=True):return subprocess.run([str(PY),'-B',str(p),*map(str,args)],check=check,creationflags=subprocess.CREATE_NO_WINDOW)
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--owner-pid',type=int,required=True);args=p.parse_args()
    write('waiting_for_qualification',owner_pid=args.owner_pid)
    while psutil.pid_exists(args.owner_pid):
        try:assert 'qualify_v5.py' in ' '.join(psutil.Process(args.owner_pid).cmdline())
        except psutil.NoSuchProcess:break
        if (ROOT/'LIVE_V5_QUEUE_STOP_REQUEST').exists():raise InterruptedError('Requested queue stop')
        time.sleep(1)
    write('original_protocol_behavior_diagnostic')
    if not (ROOT/'source-protocol-paper25').exists():run(ROOT/'source_protocol_contrast.py')
    assert load(ROOT/'source-protocol-paper25/protocol-diagnostic-note.json')['completed']
    status=load(ROOT/'qualification-v5-status.json')
    if status['phase']!='independent_tests_complete' or not status['qualified']:
        write('qualification_not_ready',status=status,live_executed=False);return
    checkpoint=Path(status['selected']['checkpoint']);digest=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    for name in ('validation_report','final_report'):
        path=Path(status[name]);s=load(path)
        assert s['qualification']['qualified'] and s['qualification']['count']==160 and s['configuration']['checkpoint_sha256']==digest
    for p,h in load(ROOT/'source-freeze-training-v5/manifest.json').items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,p
    out=ROOT/'v5-paired-realtime';out.mkdir(exist_ok=False);plan=load(ROOT/'speed-realtime-plan.json');assert len(plan['cases'])==8
    freeze={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [*ROOT.glob('*.py'),ROOT/'speed-realtime-plan.json']}
    (out/'source-freeze.json').write_text(json.dumps(freeze,indent=2));results=[]
    oldcp=OLD/'exploration-support-original-updates/updates-1024.pt'
    for i,case in enumerate(plan['cases']):
        pair=dict(case=case)
        for mode,cp in (('released',oldcp),('speed',checkpoint)):
            if (ROOT/'LIVE_V5_QUEUE_STOP_REQUEST').exists():raise InterruptedError('Requested capture-boundary stop')
            folder=out/case['name']/mode;folder.parent.mkdir(exist_ok=True);write('fresh_realtime_control',case=i+1,total=8,mode=mode,completed=results)
            extra=['--old-baseline'] if mode=='released' else []
            run(ROOT/'live_speed_v5.py','--checkpoint',cp,'--output',folder,'--index',case['index'],'--seconds',plan['seconds_each'],*extra)
            run(OLD/'audit_live.py',folder);run(ROOT/'audit_speed_live.py',folder,check=False)
            report=load(folder/'independent-speed-live-results.json')
            assert report['case']['quadrant']==case['quadrant'] and report['case']['scale_group']==case['scale_group']
            pair[mode]=dict(folder=str(folder),audit=str(folder/'independent-speed-live-results.json'),**{k:report[k] for k in ('passed','settling_time_s','first_zoom_wall_s','center_mean','center_p95','area_mean','max_loss_s','reversals')})
            for p,h in freeze.items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,p
        results.append(pair);(out/'results.json').write_text(json.dumps(results,indent=2))
    t=lambda mode,rows:np.array([x[mode]['settling_time_s'] if x[mode]['passed'] else 50. for x in rows])
    new,old=t('speed',results),t('released',results);outer=[x for x in results if 'outer' in x['case']['name']]
    quality=all(x[k]['passed'] for x in results for k in ('speed','released'))
    comparison=dict(qualified=quality and np.median(new)<=.75*np.median(old) and np.median(t('speed',outer))<=.75*np.median(t('released',outer)),all16_quality_pass=quality,
        median_wall_settle_new_s=float(np.median(new)),median_wall_settle_old_s=float(np.median(old)),median_ratio=float(np.median(new)/np.median(old)),
        outer_median_new_s=float(np.median(t('speed',outer))),outer_median_old_s=float(np.median(t('released',outer))),
        scope='Eight fixed paired known-development scenes, fresh real UE /play and wall timestamps. Not additional heldout statistical qualification.',results=results)
    (out/'paired-speed-comparison.json').write_text(json.dumps(comparison,indent=2))
    if not comparison['qualified']:write('realtime_not_qualified',comparison=comparison);return
    write('cpu_frozen_decision_and_motion_audit')
    run(ROOT/'audit_speed_decisions_v5.py','--checkpoint',checkpoint,'--folders',ROOT/'v5-validation/cases',ROOT/'v5-final_test/cases','--output',out/'cpu-decision-motion-audit.json')
    for mode in ('released','speed'):
        folder=out/plan['cases'][3]['name']/mode;write('same_frame_rgb_detection_overlay',mode=mode)
        run(OLD/'render_detection_video.py',folder)
    write('paired_realtime_complete',qualified=True,checkpoint=str(checkpoint),checkpoint_sha256=digest,comparison=str(out/'paired-speed-comparison.json'),cpu_audit=str(out/'cpu-decision-motion-audit.json'),remaining='Review boxed videos, build speed report/release and test visible PyCharm entry')
if __name__=='__main__':
    try:main()
    except BaseException as exc:write('error_requires_inspection',error=str(exc),traceback=traceback.format_exc());raise
