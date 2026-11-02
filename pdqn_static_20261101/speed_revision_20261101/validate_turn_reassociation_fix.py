"""Serial real-online and unseen-layout checks for the turn association fix."""
import argparse,hashlib,json,shutil,subprocess,sys
from pathlib import Path

HERE=Path(__file__).resolve().parent


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    root=args.output.resolve();root.mkdir(exist_ok=False);sources=root/'frozen-sources';sources.mkdir()
    names=['robust_locked_tracker.py','locked_target_selector.py','check_turn_reassociation.py',
        'live_fast_online_results.py','stable_locked_control.py','camera_control_v3.py',
        'precision_policy_v2.py','evaluate_precision_policy.py','validate_turn_reassociation_fix.py']
    hashes={}
    for name in names:
        shutil.copy2(HERE/name,sources/name);hashes[name]=hashlib.sha256((HERE/name).read_bytes()).hexdigest()
    checkpoint=HERE/'unified_motion_20261101/control-v3-turn-training-20261102/deploy.pt'
    config=dict(source_sha256=hashes,checkpoint=str(checkpoint),
        checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        detector_reid_camera_weights_unchanged=True,
        scope='Association fix; true live RGB outputs; all GT scoring after controls end.')
    (root/'manifest.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    def run(label,script,*arguments):
        print('START',label,flush=True)
        (root/'progress.json').write_text(json.dumps(dict(phase=label)),encoding='utf-8')
        with (root/f'{label}.log').open('w',encoding='utf-8') as out:
            subprocess.run([sys.executable,str(HERE/script),*map(str,arguments)],cwd=HERE,stdout=out,stderr=subprocess.STDOUT,check=True)
        print('COMPLETE',label,flush=True)
    run('synthetic-regressions','check_turn_reassociation.py')
    run('existing-regressions','check_robust_tracker.py')
    for label,scene,seconds in [('online-turn','precision-turn',12),('online-viaduct30','viaduct30',28)]:
        run(label,'live_fast_online_results.py','--scene',scene,'--seconds',seconds,'--checkpoint',checkpoint,'--output',root/label)
    run('unseen-check','evaluate_precision_policy.py','--checkpoint',checkpoint,'--output',root/'unseen-check',
        '--split','check','--cases',4,5,6,'--steps',64)
    from audit_continuous_online import summarize
    from evaluate_camera_control_only import camera
    result=dict(manifest=config,online={})
    for label in ['online-turn','online-viaduct30']:
        folder=root/label;result['online'][label]=dict(online=summarize(folder),camera=camera(folder))
        for key,value in result['online'][label].items():
            (folder/f'{key}-audit.json').write_text(json.dumps(value,indent=2),encoding='utf-8')
    result['unseen_check']=json.loads((root/'unseen-check/summary.json').read_text())
    assert all(hashlib.sha256((HERE/name).read_bytes()).hexdigest()==sha for name,sha in hashes.items())
    result['source_unchanged_during_validation']=True
    (root/'result.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    (root/'progress.json').write_text(json.dumps(dict(phase='complete')),encoding='utf-8')
    print('VALIDATION COMPLETE',root,flush=True)


if __name__=='__main__':main()
