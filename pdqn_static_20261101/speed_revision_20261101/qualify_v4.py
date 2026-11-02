"""Serial v4 qualification; reuse completed identical old development baseline."""
import argparse,hashlib,json,subprocess,time,traceback
from pathlib import Path
import psutil
from qualify_v3 import speed

ROOT=Path(__file__).resolve().parent
PY=Path('C:/Users/12879/.conda/envs/50mamba/python.exe')
STATUS=ROOT/'qualification-v4-status.json'
def load(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(phase,**v):STATUS.write_text(json.dumps(dict(phase=phase,utc=time.time(),**v),indent=2),encoding='utf-8')
def run(*args):subprocess.run([str(PY),'-B',str(ROOT/'evaluate_speed_v4.py'),*map(str,args)],check=True,creationflags=subprocess.CREATE_NO_WINDOW)
def guard():
    for p,h in load(ROOT/'source-freeze-training-v4/manifest.json').items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,p
    if (ROOT/'V4_QUEUE_STOP_REQUEST').exists():raise InterruptedError('Requested v4 process-boundary stop')
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--owner-pid',type=int,required=True);args=p.parse_args()
    write('waiting_for_owned_training',owner_pid=args.owner_pid)
    while psutil.pid_exists(args.owner_pid):
        try:assert 'train_speed_v4.py' in ' '.join(psutil.Process(args.owner_pid).cmdline())
        except psutil.NoSuchProcess:break
        if (ROOT/'V4_QUEUE_STOP_REQUEST').exists():raise InterruptedError('Requested stop')
        time.sleep(1)
    assert load(ROOT/'training-v4/status.json')['completed']==80;guard()
    oldcp=ROOT.parent/'exploration-support-original-updates/updates-1024.pt'
    oldpath=ROOT/'v3-released-baseline-development/cases/independent-results.json'
    old=load(oldpath)
    assert len(old['episodes'])==80 and old['qualification']['qualified']
    assert old['configuration']['checkpoint_sha256']==hashlib.sha256(oldcp.read_bytes()).hexdigest()
    assert load(oldpath.parent/'frozen-verification.json')['networks_targets_optimizers_replay_rng_unchanged']
    candidates=[]
    for n in (20,40,80):
        path=ROOT/f'training-v4/development-{n:04d}/independent-results.json';r=load(path)
        m=speed(r,old) if len(r['episodes'])==80 else dict(qualified=False,reason='Incomplete development futility stop',completed=len(r['episodes']),passed=r['qualification']['passed'])
        candidates.append(dict(episodes=n,checkpoint=str(ROOT/f'training-v4/checkpoint-{n:04d}.pt'),development_report=str(path),speed=m))
    (ROOT/'v4-development-speed-comparison.json').write_text(json.dumps(dict(old_report=str(oldpath),old_report_sha256=hashlib.sha256(oldpath.read_bytes()).hexdigest(),candidates=candidates),indent=2),encoding='utf-8')
    selected=None
    for c in sorted([c for c in candidates if c['speed']['qualified']],key=lambda c:c['speed']['median_settle_sim_s']):
        out=ROOT/f'v4-corners-{c["episodes"]:04d}';write('known_corner_quality',candidate=c)
        run('--checkpoint',c['checkpoint'],'--output',out,'--count',32,'--corners');guard()
        if load(out/'cases/independent-results.json')['qualification']['qualified']:
            selected=c|dict(corner_report=str(out/'cases/independent-results.json'));break
    if selected is None:write('development_not_qualified',candidates=candidates,new_validation_executed=False,new_final_executed=False);return
    for split in ('validation','final_test'):
        oldout=ROOT/f'v4-released-baseline-{split}';out=ROOT/f'v4-{split}';write('paired_old_baseline_'+split,selected=selected)
        run('--checkpoint',oldcp,'--output',oldout,'--count',160,'--split',split,'--old-baseline')
        write('new_independent_'+split,selected=selected)
        extra=['--validation-report',ROOT/'v4-validation/cases/independent-results.json'] if split=='final_test' else []
        run('--checkpoint',selected['checkpoint'],'--output',out,'--count',160,'--split',split,*extra);guard()
        m=speed(load(out/'cases/independent-results.json'),load(oldout/'cases/independent-results.json'))
        (ROOT/f'v4-{split}-speed-comparison.json').write_text(json.dumps(m,indent=2),encoding='utf-8')
        if not m['qualified']:write(split+'_not_qualified',selected=selected,speed=m,final_executed=split=='final_test');return
    write('independent_tests_complete',selected=selected,qualified=True,validation_report=str(ROOT/'v4-validation/cases/independent-results.json'),final_report=str(ROOT/'v4-final_test/cases/independent-results.json'),remaining='Fresh paired realtime control with boxes, source-protocol contrast and new PyCharm release')
if __name__=='__main__':
    try:main()
    except BaseException as exc:write('error_requires_inspection',error=str(exc),traceback=traceback.format_exc());raise
