"""One20-round Phase A; new60 and legacy60 gates, no automatic retry."""
import argparse,json,time,traceback
from dataclasses import asdict
from pathlib import Path
import psutil,torch
from .agent import CenterAgent,SOURCE,SOURCE_SHA
from .scenes import load_protocol,training_cases,probe_cases,regression_cases,STAGE
from .rollout import episode,evaluate,ValueReviewRequired,PolicyReviewRequired
from .environment import STATE_FEATURES
from control_zoom_v11.detector import create_detector
from control_zoom_v11.run import readiness as preceding_readiness
from control_value_v10.run import check_exclusive
from control_coverage_v8.run import write_json,sha256,stop_check,network_hashes
from control_learning.environment import AtomicTransport
from integration.runtime import REPRO,ROOT,StandaloneArena
from integration.experiment import make_extractor
from integration.focused_training import GPUMonitor

HERE=Path(__file__).parent
READY=REPRO/'verification/center-repair-v12a-20261031'


def readiness():
    preceding_readiness()
    checks=json.loads((READY/'contract-checks.json').read_text(encoding='utf-8'))
    assert checks['passed'] and checks['historical_training_replay_exact']
    for n,d in checks['source_sha256'].items():assert sha256(HERE/n)==d,n
    assert sha256(SOURCE)==SOURCE_SHA
    review=json.loads((READY/'design-review.json').read_text(encoding='utf-8'))
    assert review['phase_a_live_allowed'] and review['phase_b_c_started'] is False
    assert review['contract_sha256']==sha256(READY/'contract-checks.json')


def require_pass(report,label):
    entries=report['episodes']
    total=sum(e['q_bound_diagnostic_count'] for e in entries)
    excess=sum(e['q_bound_violation_count'] for e in entries)
    if not report['complete'] or report['passed_cases']!=report['planned_total']:
        raise PolicyReviewRequired(f'{label}: {report["passed_cases"]}/{report["planned_total"]}; phase A gate failed')
    if excess/max(total,1)>.01 or max(e['max_q_bound_excess'] for e in entries)>1:
        raise PolicyReviewRequired(f'{label}: Q bound screen failed')
    if report['reversal_mean']>12:
        raise PolicyReviewRequired(f'{label}: excessive reversals')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--headless',action='store_true')
    args=parser.parse_args(argv)
    check_exclusive();readiness();torch.set_num_threads(4)
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=True)
    if any((output/n).exists() for n in ('configuration.json','initial.pt','training')):
        raise FileExistsError('Never resume or overwrite bounded experiment')
    stop_check(output)
    agent=CenterAgent();agent.initialize_from_v10()
    cfg=load_protocol();frontend=cfg['frozen_frontend']
    detector=create_detector(checkpoint=REPRO/frontend['detector'],device='cuda')
    extractor=make_extractor('cuda')
    assert detector.provenance['sha256']==frontend['detector_sha256']
    assert extractor.metadata()['weights_sha256']==frontend['reid_sha256']
    for key in ('threshold','imgsz','half','preprocessing_profile'):assert detector.provenance[key]==frontend[key]
    before={str(p):sha256(p) for p in (detector.checkpoint,extractor.weights,
        REPRO/'weights/yolo26s_ue_gpu30.pt',REPRO/'weights/hascam_focused_gpu.pt')}
    metadata=dict(profile=agent.FORMAT,phase='A_center_repair',stage=STAGE,seed=agent.seed,
        config=asdict(agent.config),initialization=str(SOURCE),initialization_sha256=SOURCE_SHA,
        initialization_scope='v10 online weights+TRAIN history; fresh optimizer/new buffer/counters; target synced',
        initial_network_hashes=network_hashes(agent),historical_training_transitions=30060,
        replay_policy='64new+64historical each update; new replay/checkpoint buffer excludes history',
        actual_learning=dict(q_lr=.0005,parameter_lr=None,batch=128,new_samples=64,historical_samples=64,
            warmup_new_interactions=256,update_every_real_transitions=4,target_sync_every_updates=100),
        algorithm='five-action potential-factored Double DQN, center objective, rehearsal; not P-DQN',
        requested_rounds=20,total_repair_round_budget=60,wall_budget_seconds=7200,repair_started_unix=time.time(),
        fixed_pulse_seconds=.04,train_epsilon=.4,state_features=STATE_FEATURES,
        detector=detector.provenance,reid=extractor.metadata(),active_zoom=False,
        no_GT_policy_or_reward=True,no_expert_controller=True,
        train_seeds=[c['seed'] for c in training_cases()],probe_seeds=[c['seed'] for c in probe_cases()],
        full_probe_seeds=[c['seed'] for c in probe_cases(True)],regression_seeds=[c['seed'] for c in regression_cases()],
        new_development_gate='balanced20/20 then disjoint40/40',legacy_regression_gate='same v10 development60/60',
        independent_validation='not_executed',final_test_partition='sealed',later_stages_started=False,
        source_sha256={},root_source_sha256={},protocol_sha256=sha256(HERE/'protocol.json'),
        design_sha256=sha256(READY/'DESIGN.md'),design_review_sha256=sha256(READY/'design-review.json'),
        contract_sha256=sha256(READY/'contract-checks.json'),preflight_actions_in_this_run=0,
        prior_calibration_actions=210,prior_calibration_scope='v11; not new interactions or replay',
        ue_module_sha256=sha256(ROOT/'PTZSim/Binaries/Win64/UnrealEditor-PTZSim.dll'),
        ue_arena_source_sha256=sha256(ROOT/'PTZSim/Source/PTZSim/PTZArena.cpp'))
    for directory in ('control_center_v12a','control_zoom_v11','control_value_v10','control_value_v9','control_coverage_v8',
        'control_corners','control_v6','control_v5','control_v4','control_v3','control_learning','control_curriculum','hascam','integration'):
        for path in sorted((REPRO/directory).glob('*.py')):
            relative=path.relative_to(REPRO);metadata['source_sha256'][str(relative)]=sha256(path)
            target=output/'source-snapshot'/relative;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(path.read_bytes())
    for relative in ('scripts/train_center_repair_50mamba.py','PTZSim/Source/PTZSim/PTZArena.cpp'):
        path=ROOT/relative;metadata['root_source_sha256'][relative]=sha256(path)
        target=output/'root-source-snapshot'/relative;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(path.read_bytes())
    (output/'protocol.json').write_bytes((HERE/'protocol.json').read_bytes())
    (output/'experiment-design.md').write_bytes((READY/'DESIGN.md').read_bytes())
    write_json(output/'configuration.json',metadata);write_json(output/'frozen-before.json',before)
    write_json(output/'status.json',dict(state='starting',phase='A',runner_pid=psutil.Process().pid))
    agent.save(output/'initial.pt',metadata)
    history=[];passed=False
    try:
        detector.warmup();stop_check(output)
        with GPUMonitor(output,True),StandaloneArena(output,headless=args.headless) as arena:
            transport=AtomicTransport(arena.port,hud=not args.headless)
            try:
                write_json(output/'status.json',dict(state='training',phase='A',runner_pid=psutil.Process().pid,ue_pid=arena.process.pid))
                with (output/'episodes.jsonl').open('w',encoding='utf-8',buffering=1) as ep,(output/'round-metrics.log').open('w',encoding='utf-8',buffering=1) as log:
                    for index,case in enumerate(training_cases(),1):
                        stop_check(output)
                        if time.time()-metadata['repair_started_unix']>7200:raise ValueReviewRequired('Repair two-hour budget exhausted')
                        result=episode(case,output/'training'/f'round-{index:04d}-seed{case["seed"]}',transport,detector,extractor,agent,True,output)
                        agent.completed_episodes=index;result['round']=index;history.append(result)
                        ep.write(json.dumps(result,allow_nan=False)+'\n')
                        line=f'ROUND {index} Q={result["mean_q_loss"]} parameter_max=not_updated reward={result["reward_sum"]:.6f} epsilon=.4 steps={result["steps"]} updates={result["network_updates"]}/{agent.update_count} new/historical_batch=64/64'
                        log.write(line+'\n');print(line,flush=True)
                        agent.save(output/'last.pt',metadata)
                        write_json(output/'progress.json',dict(phase='A',completed_rounds=index,planned_rounds=20,
                            steps=agent.environment_steps,updates=agent.update_count,historical_training_transitions=30060,
                            historical_sample_draws=agent.update_count*64,new_sample_draws=agent.update_count*64,last_round=result,later_stages_started=False))
                agent.save(output/'checkpoint-0020.pt',metadata)
                small=probe_cases();ids={c['seed'] for c in small}
                remaining=[c for c in probe_cases(True) if c['seed'] not in ids]
                reports=[]
                for label,plan in (('probe-0020',small),('probe-extra-0020',remaining),('regression-0020',regression_cases())):
                    stop_check(output)
                    write_json(output/'status.json',dict(state='development_probe',phase='A',probe=label,round=20))
                    report=evaluate(plan,output/label,transport,detector,extractor,agent,output)
                    reports.append(dict(label=label,passed=report['passed_cases'],total=report['total']))
                    write_json(output/'review-0020.json',dict(phase='A',reports=reports,complete=False))
                    require_pass(report,label)
                passed=True;agent.save(output/'center-skill-passed.pt',metadata)
                write_json(output/'review-0020.json',dict(phase='A',reports=reports,complete=True,passed=True,qualified=False))
            finally:transport.close()
        write_json(output/'status.json',dict(state='development_passed',phase='A',completed_rounds=len(history),
            steps=agent.environment_steps,updates=agent.update_count,development_passed=True,qualified=False,
            next_stage='B_zoom_skill_pending',later_stages_started=False))
    except ValueReviewRequired as exc:
        agent.save(output/'review-stopped.pt',metadata)
        write_json(output/'review-stop.json',dict(reason=str(exc),kind='policy' if isinstance(exc,PolicyReviewRequired) else 'value_or_budget'))
        write_json(output/'status.json',dict(state='review_required',phase='A',completed_rounds=len(history),
            steps=agent.environment_steps,updates=agent.update_count,qualified=False,reasons=[str(exc)],later_stages_started=False))
    except BaseException as exc:
        agent.save(output/'interrupted.pt',metadata)
        write_json(output/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc(),completed_rounds=len(history),steps=agent.environment_steps,updates=agent.update_count))
        write_json(output/'status.json',dict(state='stopped_by_user' if isinstance(exc,InterruptedError) else 'anomaly',phase='A',
            completed_rounds=len(history),steps=agent.environment_steps,updates=agent.update_count,error=repr(exc),qualified=False,later_stages_started=False))
        raise
    finally:
        after={p:sha256(p) for p in before};write_json(output/'frozen-after.json',after)
        integrity=dict(frontend_weights_unchanged=before==after,
            source_unchanged=all(sha256(REPRO/n)==h for n,h in metadata['source_sha256'].items()) and all(sha256(ROOT/n)==h for n,h in metadata['root_source_sha256'].items()),
            protocol_unchanged=sha256(HERE/'protocol.json')==metadata['protocol_sha256'],historical_source_unchanged=sha256(SOURCE)==SOURCE_SHA)
        write_json(output/'integrity-after.json',integrity)
        if not all(integrity.values()):raise RuntimeError('Frozen source/weights/history changed')
    return passed


if __name__=='__main__':main()
