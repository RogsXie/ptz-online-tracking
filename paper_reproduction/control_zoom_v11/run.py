"""One bounded active-zoom trial; no automatic retry or independent validation."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time
import traceback
import psutil
import torch
from .agent import ZoomAgent
from .rollout import episode, evaluate, ValueReviewRequired, PolicyReviewRequired
from .scenes import cases, probe_cases, load_protocol, STAGE
from .environment import STATE_FEATURES
from .detector import create_detector
from control_value_v10.run import check_exclusive
from control_coverage_v8.run import write_json, sha256, stop_check, network_hashes
from control_learning.environment import AtomicTransport
from integration.runtime import REPRO, ROOT, StandaloneArena
from integration.experiment import make_extractor
from integration.focused_training import GPUMonitor

READINESS = REPRO/'verification/active-zoom-v11-20261031'
INITIALIZATION = REPRO/'runs/potential-factored-v10-attempt1/checkpoint-0060.pt'


def review(probe, prior, completed):
    entries = probe['episodes']; reasons = []
    if any(r['initial_task_ready'] and not r['passed'] for r in entries):
        reasons.append('Already centered and correctly sized development case failed')
    if max(probe['collapsed_cases'], probe['lost_cases']) >= probe['planned_total']/2:
        reasons.append('At least half the planned cases lost or stalled')
    previous = {r['seed']:r for r in prior['episodes']} if prior else {}
    matched = [r for r in entries if r['seed'] in previous]
    improved = bool(matched) and sum(r['passed'] for r in matched) > sum(previous[r['seed']]['passed'] for r in matched)
    if probe['reversal_mean'] > 12 and not improved:
        reasons.append('Frequent reversals without improved strict success on common seeds')
    count = sum(r['q_bound_diagnostic_count'] for r in entries)
    bad = sum(r['q_bound_violation_count'] for r in entries)
    if bad/max(count,1) > .01 or max(r['max_q_bound_excess'] for r in entries) > 1:
        reasons.append('Development Q upper-bound screen failed')
    if completed == 60 and (not probe['complete'] or probe['passed_cases'] != 60):
        reasons.append('Fixed60-round budget exhausted without60/60 joint development success')
    return reasons


def readiness():
    contracts = json.loads((READINESS/'contract-checks.json').read_text())
    assert contracts['passed']
    for name,digest in contracts['source_sha256'].items():
        assert sha256(Path(__file__).parent/name) == digest, f'Untested source change: {name}'
    calibration = json.loads((READINESS/'calibration-rgb-pyramid-v2/report.json').read_text())
    assert calibration['passed'] and calibration['actual_actions'] == 105
    design = json.loads((READINESS/'design-review.json').read_text())
    assert design['bounded_live_trial_allowed']
    for relative,digest in design['source_sha256'].items():
        assert sha256(ROOT/relative) == digest, f'Reviewed source changed: {relative}'
    old = json.loads((REPRO/'runs/potential-factored-v10-attempt1/configuration.json').read_text())
    assert sha256(ROOT/'PTZSim/Binaries/Win64/UnrealEditor-PTZSim.dll') == old['ue_module_sha256']
    assert sha256(ROOT/'PTZSim/Source/PTZSim/PTZArena.cpp') == old['ue_arena_source_sha256']
    assert sha256(INITIALIZATION) == 'c13f0c7ee6448a960e66d8b3f895fe1f605c5c3a9422f6ca1c7c40f41ded629e'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--headless',action='store_true')
    args = parser.parse_args(argv)
    check_exclusive(); readiness()
    output = args.output.resolve()
    output.mkdir(parents=True,exist_ok=True)
    if any((output/name).exists() for name in ('configuration.json','initial.pt','training')):
        raise FileExistsError('Do not resume or overwrite a bounded experiment')
    stop_check(output)
    torch.set_num_threads(4)
    agent = ZoomAgent(); agent.initialize_from_v10(INITIALIZATION)
    expected = load_protocol()['frozen_frontend']
    detector = create_detector(checkpoint=REPRO/expected['detector'],device='cuda')
    extractor = make_extractor('cuda')
    for key in ('threshold','imgsz','half','preprocessing_profile'):
        assert detector.provenance[key] == expected[key]
    assert detector.provenance['sha256'] == expected['detector_sha256']
    assert extractor.metadata()['weights_sha256'] == expected['reid_sha256']
    before = {str(p):sha256(p) for p in (detector.checkpoint,extractor.weights,
        REPRO/'weights/yolo26s_ue_gpu30.pt',REPRO/'weights/hascam_focused_gpu.pt')}
    metadata = dict(profile=agent.FORMAT,stage=STAGE,seed=agent.seed,from_scratch=False,
        initialization=str(INITIALIZATION),initialization_sha256=sha256(INITIALIZATION),
        initialization_scope='v10 unqualified residual weights only; empty replay and fresh optimizer/counters',
        initial_network_hashes=network_hashes(agent),config=asdict(agent.config),
        algorithm='seven-action potential-factored fixed-pulse Double DQN; not P-DQN',
        actual_learning=dict(q_lr=.0005,parameter_lr=None,batch=128,warmup_transitions=256,
            update_every_real_transitions=4,target_sync_every_updates=100),
        target_area_ratio=.04,fixed_pulse_seconds=.04,train_epsilon=.4,requested_rounds=60,
        wall_budget_seconds=7200,wall_budget_check='training episode entry; includes intermediate probes',
        state_features=STATE_FEATURES,detector=detector.provenance,reid=extractor.metadata(),
        no_GT_policy_or_reward=True,no_expert_controller=True,active_zoom=True,
        video_recording='actual complete frozen development cases; raw and annotated25fps',
        time_limit_semantics='truncated=True,terminated=False,replay.done=False; actual final observation bootstrap',
        probe_schedule={20:20,40:20,60:60},probe_seeds=[c['seed'] for c in probe_cases()],
        full_probe_seeds=[c['seed'] for c in probe_cases(True)],
        train_seeds=[c['seed'] for c in cases(STAGE,'train',60)],
        independent_validation='not automatic; reserved120cases',final_test_partition='sealed',
        qualified_stages=[],source_sha256={},root_source_sha256={},
        protocol_sha256=sha256(Path(__file__).parent/'protocol.json'),
        design_sha256=sha256(READINESS/'DESIGN.md'),design_review_sha256=sha256(READINESS/'design-review.json'),
        contract_sha256=sha256(READINESS/'contract-checks.json'),
        calibration_report=str(READINESS/'calibration-rgb-pyramid-v2/report.json'),
        calibration_sha256=sha256(READINESS/'calibration-rgb-pyramid-v2/report.json'),
        preflight_actions_in_this_run=0,prior_calibration_actions=210,
        ue_module_sha256=sha256(ROOT/'PTZSim/Binaries/Win64/UnrealEditor-PTZSim.dll'),
        ue_arena_source_sha256=sha256(ROOT/'PTZSim/Source/PTZSim/PTZArena.cpp'))
    for directory in ('control_zoom_v11','control_value_v10','control_value_v9','control_coverage_v8','control_corners',
        'control_v6','control_v5','control_v4','control_v3','control_learning','control_curriculum','hascam','integration'):
        for path in sorted((REPRO/directory).glob('*.py')):
            relative = path.relative_to(REPRO)
            metadata['source_sha256'][str(relative)] = sha256(path)
            destination = output/'source-snapshot'/relative
            destination.parent.mkdir(parents=True,exist_ok=True)
            destination.write_bytes(path.read_bytes())
    for relative in ('scripts/train_zoom_control_50mamba.py','PTZSim/Source/PTZSim/PTZArena.cpp'):
        path=ROOT/relative
        metadata['root_source_sha256'][relative]=sha256(path)
        destination=output/'root-source-snapshot'/relative
        destination.parent.mkdir(parents=True,exist_ok=True);destination.write_bytes(path.read_bytes())
    (output/'protocol.json').write_bytes((Path(__file__).parent/'protocol.json').read_bytes())
    (output/'experiment-design.md').write_bytes((READINESS/'DESIGN.md').read_bytes())
    write_json(output/'configuration.json',metadata); write_json(output/'frozen-before.json',before)
    write_json(output/'status.json',dict(state='starting',profile=agent.FORMAT,pid=psutil.Process().pid))
    agent.save(output/'initial.pt',metadata,include_replay=False)
    history=[]; prior=None; passed=False; reasons=[]; started=time.monotonic()
    try:
        detector.warmup(); stop_check(output)
        with GPUMonitor(output,True),StandaloneArena(output,headless=args.headless) as arena:
            transport = AtomicTransport(arena.port,hud=not args.headless)
            try:
                write_json(output/'status.json',dict(state='training',runner_pid=psutil.Process().pid,ue_pid=arena.process.pid))
                with (output/'episodes.jsonl').open('w',encoding='utf-8',buffering=1) as episodes, \
                     (output/'round-metrics.log').open('w',encoding='utf-8',buffering=1) as metrics:
                    for index,case in enumerate(cases(STAGE,'train',60),1):
                        stop_check(output)
                        if time.monotonic()-started > 7200:
                            raise ValueReviewRequired('Predeclared two-hour wall budget exhausted')
                        result = episode(case,output/'training'/f'round-{index:04d}-seed{case["seed"]}',
                            transport,detector,extractor,agent,True,output)
                        agent.completed_episodes=index; result['round']=index; history.append(result)
                        episodes.write(json.dumps(result,allow_nan=False)+'\n')
                        q='not_updated' if result['mean_q_loss'] is None else f'{result["mean_q_loss"]:.8f}'
                        line=(f'ROUND {index} Q={q} parameter_max=not_updated reward={result["reward_sum"]:.6f} '
                            f'epsilon={result["epsilon"]:.2f} steps={result["steps"]} '
                            f'updates={result["network_updates"]}/{agent.update_count} Qlr=0.0005 Paramlr=disabled batch=128 '
                            f'final_area={result["final_area"]} final_FOV={result["final_fov"]}')
                        metrics.write(line+'\n'); print(line,flush=True)
                        agent.save(output/'last.pt',metadata)
                        write_json(output/'progress.json',dict(completed_rounds=index,planned_rounds=60,
                            steps=agent.environment_steps,updates=agent.update_count,last_round=result,
                            elapsed_wall_seconds=time.monotonic()-started,passed=False))
                        if index % 20 == 0:
                            agent.save(output/f'checkpoint-{index:04d}.pt',metadata)
                            write_json(output/'status.json',dict(state='development_probe',round=index))
                            probe=evaluate(probe_cases(index==60),output/f'probe-{index:04d}',
                                transport,detector,extractor,agent,output)
                            reasons=review(probe,prior,index)
                            write_json(output/f'review-{index:04d}.json',dict(stop_for_review=bool(reasons),reasons=reasons))
                            prior=probe
                            if reasons: break
                            if index==60:
                                passed=True
                                agent.save(output/'development-passed.pt',metadata)
                                break
                            write_json(output/'status.json',dict(state='training',round=index))
            finally: transport.close()
        agent.save(output/'last.pt',metadata)
        write_json(output/'status.json',dict(state='development_passed' if passed else 'review_required',
            completed_rounds=len(history),steps=agent.environment_steps,updates=agent.update_count,
            development_passed=passed,qualified=False,independent_validation='not_executed',
            reasons=reasons or ([] if passed else ['bounded attempt ended']),later_stages_started=False))
    except ValueReviewRequired as exc:
        agent.save(output/'review-stopped.pt',metadata)
        kind='policy' if isinstance(exc,PolicyReviewRequired) else 'value_or_budget'
        write_json(output/'review-stop.json',dict(reason=str(exc),kind=kind,completed_rounds=len(history),
            steps=agent.environment_steps,updates=agent.update_count))
        write_json(output/'status.json',dict(state='review_required',completed_rounds=len(history),
            steps=agent.environment_steps,updates=agent.update_count,qualified=False,reasons=[str(exc)]))
    except BaseException as exc:
        agent.save(output/'interrupted.pt',metadata)
        write_json(output/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc(),
            completed_rounds=len(history),steps=agent.environment_steps,updates=agent.update_count))
        write_json(output/'status.json',dict(state='stopped_by_user' if isinstance(exc,InterruptedError) else 'anomaly',
            error=repr(exc),qualified=False,completed_rounds=len(history),steps=agent.environment_steps,updates=agent.update_count))
        raise
    finally:
        after={path:sha256(path) for path in before}
        write_json(output/'frozen-after.json',after)
        source_ok=all(sha256(REPRO/n)==digest for n,digest in metadata['source_sha256'].items())
        source_ok=source_ok and all(sha256(ROOT/n)==digest for n,digest in metadata['root_source_sha256'].items())
        protocol_ok=sha256(Path(__file__).parent/'protocol.json')==metadata['protocol_sha256']
        write_json(output/'integrity-after.json',dict(frontend_weights_unchanged=after==before,
            source_unchanged=source_ok,protocol_unchanged=protocol_ok))
        if after!=before or not source_ok or not protocol_ok:
            write_json(output/'integrity-failure.json',dict(reason='Frozen frontend or runtime source changed'))
            raise RuntimeError('Frozen frontend or runtime source changed')
    return passed


if __name__=='__main__': main()
