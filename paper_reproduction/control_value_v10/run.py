"""One bounded fresh potential-factored Double DQN run; no automatic retry."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time
import traceback
import psutil
import torch
from control_value_v10.agent import FactoredPotentialAgent as FixedPulseAgent
from control_value_v10.rollout import episode,evaluate,ValueReviewRequired
from control_coverage_v8.run import write_json,sha256,preflight
from control_coverage_v8.scenes import cases,layout,load_protocol,stage_spec,materialize
from control_coverage_v8.environment import CoverageEnvironment,STATE_FEATURES
from control_learning.environment import AtomicTransport
from integration.runtime import REPRO,ROOT,StandaloneArena
from integration.pretrained_detector import create_detector
from integration.experiment import make_extractor
from integration.focused_training import GPUMonitor

STAGE='static_center'
READINESS=REPRO/'verification/value-redesign-v10-20261031'


def review(probe,prior,completed):
    reasons=[]
    entries=probe['episodes']
    centered=[r for r in entries if r['coverage']['layer']==0]
    if any(not r['passed'] for r in centered):
        reasons.append('already-centered development case failed; investigate before expansion')
    if probe['collapsed_cases']>=probe['total']/2 or probe['lost_cases']>=probe['total']/2:
        reasons.append('at least half the development cases lost or stalled')
    previous_by_seed={r['seed']:r for r in prior['episodes']} if prior else {}
    matched=[r for r in entries if r['seed'] in previous_by_seed]
    improved=bool(matched) and sum(r['passed'] for r in matched)>sum(previous_by_seed[r['seed']]['passed'] for r in matched)
    if probe['reversal_mean']>12 and not improved:
        reasons.append('frequent reversals without improved strict success')
    count=sum(r['q_bound_diagnostic_count'] for r in entries)
    bad=sum(r['q_bound_violation_count'] for r in entries)
    largest=max(r['max_q_bound_excess'] for r in entries)
    if bad/max(count,1)>.01 or largest>1:
        reasons.append('development greedy Q values violate preregistered calibration screen')
    if completed==60 and probe['passed_cases']!=60:
        reasons.append('fixed60-round budget exhausted without complete development success')
    return reasons


def check_exclusive():
    own=psutil.Process().pid
    for process in psutil.process_iter(['pid','name','cmdline']):
        if process.info['pid']==own:
            continue
        try:
            args=process.info['cmdline'] or []
            names=[a.replace('\\','/').split('/')[-1] for a in args]
            conflict='UnrealEditor' in (process.info['name'] or '') or any(
                n.startswith('train_') and n.endswith('_50mamba.py') for n in names)
            if conflict:
                raise RuntimeError(f'Existing UE/training process {process.info["pid"]}; refusing a second instance')
        except (psutil.AccessDenied,psutil.NoSuchProcess):
            continue


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--headless',action='store_true')
    args=parser.parse_args(argv)
    check_exclusive()
    output=args.output.resolve()
    output.mkdir(parents=True,exist_ok=True)
    if (output/'configuration.json').exists():
        raise FileExistsError('Do not resume or overwrite a bounded experiment')
    assert json.loads((READINESS/'offline-screen/report.json').read_text())['completed']
    assert json.loads((READINESS/'offline-screen/gate-recheck.json').read_text())['live_candidate_eligible']
    assert json.loads((READINESS/'design-review.json').read_text())['bounded_live_trial_allowed']
    contracts=json.loads((READINESS/'contract-checks.json').read_text())
    assert contracts['passed']
    for name,digest in contracts['source_sha256'].items():
        assert sha256(Path(__file__).parent/name)==digest, f'Untested source change: {name}'
    prior_ready=REPRO/'verification/coverage-static-v8-20261031'
    assert json.loads((prior_ready/'initial-rgb-calibration/report.json').read_text())['passed']
    v8_metadata=json.loads((REPRO/'runs/coverage-static-v8-attempt1/configuration.json').read_text())
    assert sha256(ROOT/'PTZSim/Binaries/Win64/UnrealEditor-PTZSim.dll')==v8_metadata['ue_module_sha256']
    assert sha256(ROOT/'PTZSim/Source/PTZSim/PTZArena.cpp')==v8_metadata['ue_arena_source_sha256']
    torch.set_num_threads(4)
    agent=FixedPulseAgent()
    expected=load_protocol()['frozen_frontend']
    detector=create_detector(checkpoint=REPRO/expected['detector'],device='cuda')
    extractor=make_extractor('cuda')
    for key in ('threshold','imgsz','half'):
        assert detector.provenance[key]==expected[key]
    assert detector.provenance['sha256']==expected['detector_sha256']
    assert extractor.metadata()['weights_sha256']==expected['reid_sha256']
    before={str(detector.checkpoint):detector.provenance['sha256'],
        str(REPRO/'weights/yolo26s_ue_gpu30.pt'):sha256(REPRO/'weights/yolo26s_ue_gpu30.pt'),
        str(extractor.weights):extractor.metadata()['weights_sha256'],
        str(REPRO/'weights/hascam_focused_gpu.pt'):sha256(REPRO/'weights/hascam_focused_gpu.pt')}
    metadata=dict(profile=agent.FORMAT,stage=STAGE,from_scratch=True,seed=agent.seed,
        config=asdict(agent.config),algorithm='potential-factored fixed-pulse Double DQN; same original RGB return; no duration learning',
        actual_learning=dict(q_lr=.0005,parameter_lr=None,batch=128,update_every_real_transitions=4,warmup_transitions=256),
        fixed_pulse_seconds=.04,train_epsilon=.4,requested_rounds=60,
        state_features=STATE_FEATURES,detector=detector.provenance,reid=extractor.metadata(),
        no_GT_policy_or_reward=True,no_expert_controller=True,active_zoom=False,video_recording=False,
        time_limit_semantics='truncated=True,terminated=False,replay.done=False; actual final observation bootstrap',
        probe_schedule={20:20,40:20,60:60},independent_validation_count=120,
        validation_trigger='all60 development cases pass and all review checks pass',
        final_test_partition='sealed',train_seeds=[2100000,2100059],probe_seeds=[2200000,2200059],
        validation_seeds=[2300000,2300119],qualified_stages=[],source_sha256={},
        protocol_sha256=sha256(REPRO/'control_coverage_v8/protocol.json'),
        design_sha256=sha256(READINESS/'DESIGN.md'),candidate_readme_sha256=sha256(REPRO/'control_value_v10/README.md'),
        offline_report_sha256=sha256(READINESS/'offline-screen/report.json'),
        exact_count_gate_sha256=sha256(READINESS/'offline-screen/gate-recheck.json'),
        design_review_sha256=sha256(READINESS/'design-review.json'),
        contract_sha256=sha256(READINESS/'contract-checks.json'),
        ue_module_sha256=sha256(ROOT/'PTZSim/Binaries/Win64/UnrealEditor-PTZSim.dll'),
        ue_arena_source_sha256=sha256(ROOT/'PTZSim/Source/PTZSim/PTZArena.cpp'),
        interpretation='one fresh potential-factored candidate; offline paired screen is not independent control validation')
    for directory in ('control_value_v10','control_value_v9','control_coverage_v8','control_corners','control_v6','control_v5','control_v4',
                      'control_v3','control_learning','control_curriculum','hascam','integration'):
        for path in sorted((REPRO/directory).glob('*.py')):
            relative=path.relative_to(REPRO)
            metadata['source_sha256'][str(relative)]=sha256(path)
            destination=output/'source-snapshot'/relative
            destination.parent.mkdir(parents=True,exist_ok=True)
            destination.write_bytes(path.read_bytes())
    (output/'protocol.json').write_bytes((REPRO/'control_coverage_v8/protocol.json').read_bytes())
    (output/'experiment-design.md').write_bytes((READINESS/'DESIGN.md').read_bytes())
    (output/'candidate-design.md').write_bytes((REPRO/'control_value_v10/README.md').read_bytes())
    write_json(output/'configuration.json',metadata)
    write_json(output/'frozen-before.json',before)
    write_json(output/'status.json',dict(state='starting',profile=agent.FORMAT,pid=psutil.Process().pid))
    agent.save(output/'initial.pt',metadata,include_replay=False)
    history=[];prior=None;passed=False;reasons=[]
    started=time.monotonic()
    try:
        detector.warmup()
        with GPUMonitor(output,True),StandaloneArena(output,headless=args.headless) as arena:
            transport=AtomicTransport(arena.port,hud=not args.headless)
            try:
                write_json(output/'status.json',dict(state='preflight'))
                preflight(output/'preflight',transport,detector,extractor,agent,output)
                responses=[]
                # Fixed diagnostic pulses only: no replay, updates or expert labels.
                for action in range(5):
                    scene=materialize(STAGE,'train',2100027,output/'pulse04-preflight'/str(action))
                    env=CoverageEnvironment(transport,detector,extractor,stage_spec(STAGE))
                    env.reset(scene,2100027)
                    _,reward,_,row=env.step(action,.04)
                    responses.append(dict(action=action,before=env.rows[0],after=row,reward=reward))
                assert len(agent.memory)==0 and agent.update_count==0
                write_json(output/'pulse04-preflight/report.json',dict(passed=True,responses=responses,
                    excluded_from_replay=True,updates=0))
                write_json(output/'status.json',dict(state='training',runner_pid=psutil.Process().pid,ue_pid=arena.process.pid))
                with (output/'episodes.jsonl').open('w',encoding='utf-8',buffering=1) as episodes, \
                     (output/'round-metrics.log').open('w',encoding='utf-8',buffering=1) as metrics:
                    for index,case in enumerate(cases(STAGE,'train',60),1):
                        if time.monotonic()-started>7200:
                            raise ValueReviewRequired('Predeclared two-hour wall budget exhausted')
                        result=episode(case,output/'training'/f'round-{index:04d}-seed{case["seed"]}',
                                       transport,detector,extractor,agent,True,output)
                        agent.completed_episodes=index
                        result['round']=index
                        history.append(result)
                        episodes.write(json.dumps(result,allow_nan=False)+'\n')
                        q='not_updated' if result['mean_q_loss'] is None else f'{result["mean_q_loss"]:.8f}'
                        line=(f'ROUND {index} Q={q} parameter_max=not_updated reward={result["reward_sum"]:.6f} '
                              f'epsilon={result["epsilon"]:.2f} steps={result["steps"]} '
                              f'updates={result["network_updates"]}/{agent.update_count} Qlr=0.0005 Paramlr=disabled batch=128')
                        metrics.write(line+'\n');print(line,flush=True)
                        agent.save(output/'last.pt',metadata)
                        write_json(output/'progress.json',dict(completed_rounds=index,planned_rounds=60,
                            steps=agent.environment_steps,updates=agent.update_count,last_round=result,passed=False,later_stages_started=False))
                        if index%20==0:
                            agent.save(output/f'checkpoint-{index:04d}.pt',metadata)
                            write_json(output/'status.json',dict(state='development_probe',round=index))
                            probe=evaluate(cases(STAGE,'probe',60 if index==60 else 20),output/f'probe-{index:04d}',
                                           transport,detector,extractor,agent,output)
                            reasons=review(probe,prior,index)
                            write_json(output/f'review-{index:04d}.json',dict(stop_for_review=bool(reasons),reasons=reasons))
                            prior=probe
                            if reasons:
                                break
                            if index==60 and probe['passed_cases']==60:
                                write_json(output/'status.json',dict(state='independent_validation',round=index))
                                validation=evaluate(cases(STAGE,'validation'),output/f'validation-{index:04d}',
                                                    transport,detector,extractor,agent,output,official=True)
                                passed=validation['gate']['passed']
                                if passed:
                                    metadata['qualified_stages']=[STAGE]
                                    agent.save(output/'qualified.pt',metadata)
                                else:
                                    reasons=['independent120-case grouped validation failed']
                                break
                            write_json(output/'status.json',dict(state='training',round=index))
            finally:
                transport.close()
        agent.save(output/'last.pt',metadata)
        write_json(output/'status.json',dict(state='qualified' if passed else 'review_required',
            completed_rounds=len(history),steps=agent.environment_steps,updates=agent.update_count,passed=passed,
            reasons=reasons or ([] if passed else ['bounded attempt ended']),later_stages_started=False))
    except ValueReviewRequired as exc:
        agent.save(output/'review-stopped.pt',metadata)
        write_json(output/'policy-review.json',dict(reason=str(exc),completed_rounds=len(history),steps=agent.environment_steps,updates=agent.update_count))
        write_json(output/'status.json',dict(state='review_required',completed_rounds=len(history),
            steps=agent.environment_steps,updates=agent.update_count,passed=False,reasons=[str(exc)],later_stages_started=False))
    except BaseException as exc:
        agent.save(output/'interrupted.pt',metadata)
        write_json(output/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc(),
            completed_rounds=len(history),steps=agent.environment_steps,updates=agent.update_count))
        write_json(output/'status.json',dict(state='stopped_by_user' if isinstance(exc,InterruptedError) else 'anomaly',
            error=repr(exc),passed=False,completed_rounds=len(history),later_stages_started=False))
        raise
    finally:
        after={path:sha256(path) for path in before}
        write_json(output/'frozen-after.json',after)
        if after!=before:
            raise RuntimeError('Frozen frontend changed')
    return passed


if __name__=='__main__':
    main()
