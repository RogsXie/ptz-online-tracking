"""Serial qualification; complete real quality evidence and paired speed gains."""
import argparse,hashlib,json,subprocess,time,traceback
from pathlib import Path
import psutil,numpy as np

ROOT=Path(__file__).resolve().parent;PY=Path('C:/Users/12879/.conda/envs/50mamba/python.exe');STATUS=ROOT/'qualification-v3-status.json'
def load(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(phase,**v):STATUS.write_text(json.dumps(dict(phase=phase,utc=time.time(),**v),indent=2),encoding='utf-8')
def run(*args):subprocess.run([str(PY),'-B',str(ROOT/'evaluate_speed_v3.py'),*map(str,args)],check=True,creationflags=subprocess.CREATE_NO_WINDOW)
def guard():
    for p,h in load(ROOT/'source-freeze-training-v3/manifest.json').items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,p
    if (ROOT/'V3_QUEUE_STOP_REQUEST').exists():raise InterruptedError('Requested v3process-boundary stop')

def speed(new,old):
    assert len(new['episodes'])==len(old['episodes']) and {r['seed'] for r in new['episodes']}=={r['seed'] for r in old['episodes']}
    t=lambda r:[x['settling_time'] if x['passed'] else 50. for x in r['episodes']]
    a,b=t(new),t(old);median=float(np.median(a));p95=float(np.quantile(a,.95))
    zoom=[x['first_zoom_s'] if x['first_zoom_s'] is not None else 50. for x in new['episodes'] if x['needs_zoom']]
    outer=[x['first_zoom_s'] if x['first_zoom_s'] is not None else 50. for x in new['episodes'] if x['needs_zoom'] and x['layer']>=3]
    checks=dict(new_quality=new['qualification']['qualified'],baseline_quality=old['qualification']['qualified'],
        median=median<=.6*np.median(b),p95=p95<=.75*np.quantile(b,.95),
        first_zoom=bool(zoom) and np.median(zoom)<=5.,outer_first_zoom=bool(outer) and np.median(outer)<=5.)
    return dict(qualified=all(checks.values()),checks={k:bool(v) for k,v in checks.items()},
        median_settle_sim_s=median,old_median_settle_sim_s=float(np.median(b)),median_ratio=float(median/np.median(b)),
        p95_settle_sim_s=p95,old_p95_settle_sim_s=float(np.quantile(b,.95)),p95_ratio=float(p95/np.quantile(b,.95)),
        need_zoom_first_median_sim_s=float(np.median(zoom)) if zoom else None,
        outer_need_zoom_first_median_sim_s=float(np.median(outer)) if outer else None,
        failure_settle_and_missing_zoom_penalty_s=50.,scope='Paired synchronous UE-time metrics; real wall comparison still required')

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--owner-pid',type=int,required=True);args=p.parse_args()
    write('waiting_for_owned_training',owner_pid=args.owner_pid)
    while psutil.pid_exists(args.owner_pid):
        try:assert 'train_speed_v3.py' in ' '.join(psutil.Process(args.owner_pid).cmdline())
        except psutil.NoSuchProcess:break
        if (ROOT/'V3_QUEUE_STOP_REQUEST').exists():raise InterruptedError('Requested stop')
        time.sleep(1)
    assert load(ROOT/'training-v3/status.json')['completed']==160;guard()
    oldcp=ROOT.parent/'exploration-support-original-updates/updates-1024.pt';oldout=ROOT/'v3-released-baseline-development'
    write('paired_old_baseline_development');run('--checkpoint',oldcp,'--output',oldout,'--count',80,'--old-baseline')
    old=load(oldout/'cases/independent-results.json');candidates=[]
    for n in (20,40,80,160):
        path=ROOT/f'training-v3/development-{n:04d}/independent-results.json';r=load(path)
        if len(r['episodes'])==80:m=speed(r,old)
        else:m=dict(qualified=False,reason='Incomplete development futility stop',completed=len(r['episodes']),passed=r['qualification']['passed'])
        candidates.append(dict(episodes=n,checkpoint=str(ROOT/f'training-v3/checkpoint-{n:04d}.pt'),development_report=str(path),speed=m))
    (ROOT/'v3-development-speed-comparison.json').write_text(json.dumps(candidates,indent=2),encoding='utf-8')
    selected=None
    for candidate in sorted([c for c in candidates if c['speed']['qualified']],key=lambda c:c['speed']['median_settle_sim_s']):
        out=ROOT/f'v3-corners-{candidate["episodes"]:04d}';write('known_corner_quality',candidate=candidate)
        run('--checkpoint',candidate['checkpoint'],'--output',out,'--count',32,'--corners');guard()
        r=load(out/'cases/independent-results.json')
        if r['qualification']['qualified']:selected=candidate|dict(corner_report=str(out/'cases/independent-results.json'));break
    if selected is None:write('development_not_qualified',candidates=candidates,new_validation_executed=False,new_final_executed=False);return
    for split in ('validation','final_test'):
        oldout=ROOT/f'v3-released-baseline-{split}';out=ROOT/f'v3-{split}';write('paired_old_baseline_'+split,selected=selected)
        run('--checkpoint',oldcp,'--output',oldout,'--count',160,'--split',split,'--old-baseline')
        write('new_independent_'+split,selected=selected)
        extra=['--validation-report',ROOT/'v3-validation/cases/independent-results.json'] if split=='final_test' else []
        run('--checkpoint',selected['checkpoint'],'--output',out,'--count',160,'--split',split,*extra);guard()
        m=speed(load(out/'cases/independent-results.json'),load(oldout/'cases/independent-results.json'))
        (ROOT/f'v3-{split}-speed-comparison.json').write_text(json.dumps(m,indent=2),encoding='utf-8')
        if not m['qualified']:write(split+'_not_qualified',selected=selected,speed=m,final_executed=split=='final_test');return
    write('independent_tests_complete',selected=selected,qualified=True,
        validation_report=str(ROOT/'v3-validation/cases/independent-results.json'),final_report=str(ROOT/'v3-final_test/cases/independent-results.json'),
        remaining='Fresh paired realtime control with boxes, source-protocol contrast and new PyCharm release; not complete')

if __name__=='__main__':
    try:main()
    except BaseException as exc:write('error_requires_inspection',error=str(exc),traceback=traceback.format_exc());raise
