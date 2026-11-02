"""Serial continuation after the owned, real exploration collection process.

This coordinator changes no controller and never marks the project complete.
Checkpoint selection uses development data only. Heldout tests remain gated.
"""
import argparse,hashlib,json,subprocess,sys,time,traceback
from pathlib import Path
import psutil

ROOT=Path(__file__).resolve().parent
PYTHON=Path('C:/Users/12879/.conda/envs/50mamba/python.exe')
STATUS=ROOT/'exploration-qualification-status.json'

def write(phase,**detail):
    STATUS.write_text(json.dumps(dict(phase=phase,utc_seconds=time.time(),**detail),indent=2),encoding='utf-8')

def load(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def sources():
    local=['bootstrap.py','train.py','environment.py','scenes.py','evaluation_scenes.py',
        'evaluate_audited.py','evaluate_corner_development.py','independent_validation.py',
        'protocol.json','heldout_coverage_declaration.json','replay_update.py']
    upstream=['author_aligned/source.py','author_aligned/environment.py','author_aligned/run.py',
        'integration/runtime.py','integration/transport.py','integration/focused_environment.py',
        'integration/pretrained_detector.py','control_model_v13/detector.py','hascam/reid.py','hascam/cfff.py',
        'reference/HAS-CAM/agent.py','reference/HAS-CAM/model.py','reference/HAS-CAM/memory.py']
    paths=[ROOT/n for n in local]+[ROOT.parent/'paper_reproduction'/n for n in upstream]
    return {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}

def unchanged(manifest):
    assert sources()==manifest,'A frozen control/evaluation source changed; inspect before continuing'
    if (ROOT/'EXPLORATION_QUEUE_STOP').exists():raise InterruptedError('Requested stop at process boundary')

def run(entry,*arguments):
    subprocess.run([str(PYTHON),'-B',str(ROOT/entry),*map(str,arguments)],check=True,
        creationflags=subprocess.CREATE_NO_WINDOW)

def verified_report(folder,checkpoint):
    report=load(folder/'independent-results.json')
    assert load(folder/'frozen-verification.json')['networks_targets_optimizers_replay_rng_unchanged']
    assert report['configuration']['checkpoint_sha256']==hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    return report

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--owner-pid',type=int,required=True)
    args=parser.parse_args();data=ROOT/'real-exploration-support-after0480'
    write('waiting_for_owned_collection',owned_python_pid=args.owner_pid)
    while psutil.pid_exists(args.owner_pid):
        try:
            if 'collect_exploration_support.py' not in ' '.join(psutil.Process(args.owner_pid).cmdline()):
                raise RuntimeError('Owner PID was reused by another process')
        except psutil.NoSuchProcess:break
        if (ROOT/'EXPLORATION_QUEUE_STOP').exists():raise InterruptedError('Requested collection queue stop')
        time.sleep(1)
    summary=load(data/'summary.json')
    assert summary['status']=='real_data_collection_complete'
    assert summary['unique_initial_rollout_fragments']==640 and summary['new_real_transitions']==5120
    assert summary['completed_control_episodes_claimed']==0 and summary['parent_completed_episodes']==480
    assert summary['weights_targets_optimizers_unchanged'] and summary['updates']==11238
    manifest=sources()
    (ROOT/'exploration-qualification-source-freeze.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    update_out=ROOT/'exploration-support-original-updates'
    write('original_source_updates',collection_summary=summary,additional_updates=2048)
    run('replay_update.py','--checkpoint',data/'data.pt','--output',update_out,'--updates','2048')
    unchanged(manifest)
    assert load(update_out/'summary.json')['additional_updates']==2048
    selected=None;development_results=[]
    for budget in (2048,1024,512):
        checkpoint=update_out/f'updates-{budget:04d}.pt'
        corners=ROOT/f'audited-corner-development-exploration-u{budget:04d}'
        write('auditing_development_corners',additional_updates=budget,previous_candidates=development_results)
        run('evaluate_corner_development.py','--checkpoint',checkpoint,'--output',corners,'--count',32)
        unchanged(manifest);corner=verified_report(corners,checkpoint)
        record=dict(additional_updates=budget,corner_passed=corner['qualification']['passed'],corner_count=32)
        development_results.append(record)
        if not corner['qualification']['qualified']:continue
        development=ROOT/f'audited-development-exploration-u{budget:04d}'
        write('auditing_full_development',additional_updates=budget,previous_candidates=development_results)
        run('evaluate_audited.py','--checkpoint',checkpoint,'--output',development,'--split','development','--count',80)
        unchanged(manifest);dev=verified_report(development,checkpoint)
        record.update(development_passed=dev['qualification']['passed'],development_count=80)
        if not dev['qualification']['qualified']:continue
        selected=dict(checkpoint=str(checkpoint),additional_updates=budget,
            corner_report=str(corners/'independent-results.json'),development_report=str(development/'independent-results.json'))
        break
    if selected is None:
        write('development_not_qualified',candidates=development_results,validation_executed=False,final_test_executed=False)
        return
    checkpoint=Path(selected['checkpoint']);budget=selected['additional_updates']
    validation=ROOT/f'validation-exploration-u{budget:04d}'
    write('independent_validation',selected=selected)
    run('evaluate_audited.py','--checkpoint',checkpoint,'--output',validation,'--split','validation','--count',160)
    unchanged(manifest);report=verified_report(validation,checkpoint)
    validation_report=validation/'independent-results.json'
    if not report['qualification']['qualified']:
        write('validation_not_qualified',selected=selected,validation_report=str(validation_report),final_test_executed=False)
        return
    final=ROOT/f'final-test-exploration-u{budget:04d}'
    write('independent_final_test',selected=selected,validation_report=str(validation_report))
    run('evaluate_audited.py','--checkpoint',checkpoint,'--output',final,'--split','final_test',
        '--count',160,'--validation-report',validation_report)
    unchanged(manifest);report=verified_report(final,checkpoint)
    write('independent_tests_complete',selected=selected,validation_report=str(validation_report),
        final_test_report=str(final/'independent-results.json'),final_test_qualified=report['qualification']['qualified'],
        goal_complete=False,remaining='Fresh realtime representative videos, independent video audits, report and PyCharm release')

if __name__=='__main__':
    try:main()
    except BaseException as exc:
        write('error_requires_inspection',error=str(exc),traceback=traceback.format_exc())
        raise
