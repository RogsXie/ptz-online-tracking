"""One bounded RGB response-learning attempt, then frozen joint development."""
import argparse,json,pickle,time,traceback
from pathlib import Path
import cv2,numpy as np,psutil,torch
from .agent import ResponseAgent,observation,frozen
from .detector import create_detector
from .scenes import STAGE,HERE,load_protocol,stage_spec,layout,materialize,cases,training_cases,probe_cases
from .scenes import initial_zone
from .gates import assess_episode
from .video import CaseVideo
from control_zoom_v11.environment import ZoomEnvironment
from control_learning.environment import AtomicTransport
from control_value_v10.run import check_exclusive
from control_coverage_v8.run import write_json,sha256
from integration.runtime import ROOT,REPRO,StandaloneArena
from integration.experiment import make_extractor

READY=REPRO/'verification/zoom-model-v13-20261031'
class ReviewRequired(RuntimeError):pass

def stop_check(root):
    if (root/'STOP_REQUEST').exists():raise InterruptedError('User or monitor stop request')

def start_case(case,folder,transport,detector,extractor):
    folder.mkdir(parents=True,exist_ok=False)
    scene=materialize(STAGE,case['split'],case['seed'],folder)
    env=ZoomEnvironment(transport,detector,extractor,stage_spec(STAGE));env.reset(scene,case['seed'])
    transport.mode('LEARNED_RGB_RESPONSE_MODEL_V13')
    row=env.rows[0];row['initial_zone_confirmed']=initial_zone(row,case['seed'],case['split'])
    row['model_state']=observation(row).tolist() if row['visible'] else None
    write_json(folder/'resolved-scene.json',transport.diagnostics)
    write_json(folder/'initial-perception.json',dict(selected=row,candidates=[dict(bbox=d.bbox,confidence=d.confidence) for d in env.detections]))
    cv2.imwrite(str(folder/'initial-rgb.png'),cv2.cvtColor(transport.rgb,cv2.COLOR_RGB2BGR))
    if not row['initial_zone_confirmed']:raise RuntimeError('Initial RGB layout/full-box contract failed')
    return env

def episode(case,folder,transport,detector,extractor,agent,training,root):
    env=start_case(case,folder,transport,detector,extractor)
    video=None;fits=[];random_actions=0;reward_sum=0.;start=time.perf_counter();updates_before=agent.update_count
    try:
        if not training:video=CaseVideo(folder,case['seed'])
        with (folder/'steps.jsonl').open('w',encoding='utf-8',buffering=1) as log,(folder/'updates.jsonl').open('w',encoding='utf-8',buffering=1) as updates:
            log.write(json.dumps(env.rows[0]|dict(initial=True))+'\n')
            if video:video.write(transport.rgb,env.rows[0])
            for decision in range(stage_spec(STAGE)['max_decisions']):
                stop_check(root);state=observation(env.rows[-1]);action=agent.select(state,training)
                selection=agent.last_selection.copy();_,reward,done,row=env.step(action,.04)
                next_state=observation(row);row.update(model_state=next_state.tolist() if next_state is not None else None,selection=selection)
                # Save the completed physical action even if a subsequent learner check fails.
                log.write(json.dumps(row,allow_nan=False)+'\n');reward_sum+=reward;random_actions+=int(selection['random'])
                if video:video.write(transport.rgb,row)
                if training:
                    fit=agent.learn(state,action,next_state,reward,row['terminated'],row['truncated'],case['seed'],decision)
                    if fit:updates.write(json.dumps(fit,allow_nan=False)+'\n');fits.append(fit)
                if next_state is None:raise RuntimeError('RGB target unavailable; preserve actual action, stop before guessing recovery')
                if done:break
        result=assess_episode(STAGE,env.rows)
        result.update(seed=case['seed'],split=case['split'],coverage=layout(case['seed'],case['split']),
            reward_sum=reward_sum,steps=len(env.rows)-1,model_updates=agent.update_count-updates_before,
            updates_total=agent.update_count,epsilon=.4 if training else 0.,random_actions=random_actions,
            fit_mse_xyz_mean=np.mean([f['mse_xyz'] for f in fits],axis=0).tolist() if fits else None,
            model_sha256=agent.digest(),wall_seconds=time.perf_counter()-start,
            action_counts=np.bincount([r['action_type'] for r in env.rows[1:]],minlength=7).tolist(),
            initial_area=env.rows[0]['area_ratio'],final_area=env.rows[-1]['area_ratio'],
            final_center_error=env.rows[-1]['center_error'],initial_fov=env.rows[0]['fov'],final_fov=env.rows[-1]['fov'],
            initial_task_ready=bool(env.rows[0]['visible'] and env.rows[0]['center_error']<=.025 and .025<=env.rows[0]['area_ratio']<=.055),
            saturated_nonstay=sum(r.get('saturated_nonstay',False) for r in env.rows[1:]),
            video_frames=video.frames if video else 0)
        write_json(folder/'summary.json',result);return result
    finally:
        if video:video.close()
        write_json(folder/'action-receipt.json',dict(actual_actions=len(env.rows)-1,
            completed_video_frames=video.frames if video else 0,model_updates=agent.update_count-updates_before))

def evaluate(plan,folder,transport,detector,extractor,agent,root):
    folder.mkdir(parents=True,exist_ok=False);receipt={};results=[];error=None
    try:
        with frozen(agent,receipt):
            for i,case in enumerate(plan,1):
                stop_check(root)
                result=episode(case,folder/f'seed{case["seed"]}',transport,detector,extractor,agent,False,root);results.append(result)
                write_json(folder/'progress.json',dict(completed=i,planned=len(plan),passed=sum(r['passed'] for r in results),last_case=result))
                print(f'PROBE {i}/{len(plan)} seed={case["seed"]} passed={result["passed"]} failures={result["failures"]}',flush=True)
                if result['initial_task_ready'] and not result['passed']:raise ReviewRequired('Initially centered and correctly sized case failed')
                stalled=sum(not r['passed'] and r['initial_center_error']-r['final_center_error']<.01 and abs(np.log(r['initial_area']/.04))-abs(np.log(r['final_area']/.04))<.01 for r in results)
                lost=sum(r['selected_availability']<.5 for r in results)
                if max(stalled,lost)>=len(plan)/2:raise ReviewRequired('At least half the planned cases stalled or lost')
            if np.mean([r['direction_reversals'] for r in results])>12:raise ReviewRequired('Mean reversals above12 without prior common-seed improvement')
    except BaseException as exc:error=repr(exc);raise
    finally:
        write_json(folder/'freeze-receipt.json',receipt)
        write_json(folder/'report.json',dict(planned=len(plan),total=len(results),passed_cases=sum(r['passed'] for r in results),
            passed=len(results)==len(plan) and all(r['passed'] for r in results),actual_actions=sum(r['steps'] for r in results),
            freeze_receipt=receipt,episodes=results,error=error,kind='development_not_independent_validation'))
    return results

def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=Path);p.add_argument('--headless',action='store_true');args=p.parse_args(argv)
    check_exclusive();torch.set_num_threads(4);root=args.output.resolve();root.mkdir(parents=True,exist_ok=False)
    contract=json.loads((READY/'contract-checks.json').read_text(encoding='utf-8'));assert contract['passed']
    for n,d in contract['source_sha256'].items():assert sha256(HERE/n)==d,'Untested source change'
    assert sha256(HERE/'protocol.json')==contract['protocol_sha256'],'Untested protocol change'
    cfg=load_protocol();front=cfg['frozen_frontend'];agent=ResponseAgent()
    detector=create_detector(checkpoint=REPRO/front['detector'],device='cuda');extractor=make_extractor('cuda')
    assert detector.provenance['sha256']==front['detector_sha256'];assert extractor.metadata()['weights_sha256']==front['reid_sha256']
    previous=REPRO/'runs/center-repair-v12a-attempt1';prior=json.loads((previous/'configuration.json').read_text(encoding='utf-8'))
    before={str(path):sha256(path) for path in [detector.checkpoint,extractor.weights,REPRO/'weights/yolo26s_ue_gpu30.pt',REPRO/'weights/hascam_focused_gpu.pt']}
    meta=dict(profile=agent.FORMAT,algorithm='learned action-conditioned RGB response + beam planning',
        learning=cfg['learning'],q_networks=False,parameter_networks=False,new_data_only=True,
        detector=detector.provenance,reid=extractor.metadata(),prior_training_rounds=9,requested_training_rounds=20,
        repair_started_unix=prior['repair_started_unix'],total_repair_round_budget=60,wall_budget_seconds=7200,
        state_features=['RGB x','RGB y','log(RGB area/.04)'],no_fov_or_gt_policy_input=True,
        independent_validation='not_executed',final_tests='sealed',source_sha256={},root_source_sha256={},
        protocol_sha256=sha256(HERE/'protocol.json'),design_sha256=sha256(READY/'DESIGN.md'),contract_sha256=sha256(READY/'contract-checks.json'),
        ue_module_sha256=sha256(ROOT/'PTZSim/Binaries/Win64/UnrealEditor-PTZSim.dll'))
    directories=['control_model_v13','control_zoom_v11','control_value_v10','control_value_v9','control_coverage_v8','control_corners','control_v6','control_v5','control_v4','control_v3','control_learning','control_curriculum','hascam','integration']
    for directory in directories:
        for path in (REPRO/directory).glob('*.py'):
            rel=path.relative_to(REPRO);meta['source_sha256'][str(rel)]=sha256(path);target=root/'source-snapshot'/rel;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(path.read_bytes())
    for rel in ['scripts/train_model_zoom_50mamba.py','PTZSim/Source/PTZSim/PTZArena.cpp']:
        path=ROOT/rel;meta['root_source_sha256'][rel]=sha256(path);target=root/'root-source-snapshot'/rel;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(path.read_bytes())
    (root/'protocol.json').write_bytes((HERE/'protocol.json').read_bytes());(root/'experiment-design.md').write_bytes((READY/'DESIGN.md').read_bytes())
    write_json(root/'configuration.json',meta);write_json(root/'frozen-before.json',before);agent.save(root/'initial.pkl',meta)
    history=[];probes=[]
    try:
        stop_check(root);detector.warmup()
        with StandaloneArena(root,headless=args.headless) as arena:
            transport=AtomicTransport(arena.port,hud=not args.headless)
            try:
                write_json(root/'status.json',dict(state='initial_rgb_preflight',runner_pid=psutil.Process().pid,ue_pid=arena.process.pid))
                initial=[]
                for case in cases(STAGE,'calibration'):
                    stop_check(root);env=start_case(case,root/'preflight'/f'seed{case["seed"]}',transport,detector,extractor)
                    initial.append(dict(seed=case['seed'],initial_zone_confirmed=env.rows[0]['initial_zone_confirmed'],area_ratio=env.rows[0]['area_ratio']))
                    write_json(root/'preflight-progress.json',dict(completed=len(initial),planned=60,actual_actions=0))
                write_json(root/'preflight-report.json',dict(passed=len(initial)==60,actual_actions=0,initials=initial,scope='perception only; no policy success'))
                write_json(root/'status.json',dict(state='training',runner_pid=psutil.Process().pid,ue_pid=arena.process.pid))
                with (root/'episodes.jsonl').open('w',encoding='utf-8',buffering=1) as log:
                    for index,case in enumerate(training_cases(),1):
                        stop_check(root)
                        if time.time()-meta['repair_started_unix']>7200:raise ReviewRequired('Inherited two-hour budget exhausted')
                        if meta['prior_training_rounds']+index>60:raise ReviewRequired('Inherited sixty-round budget exhausted')
                        result=episode(case,root/'training'/f'round-{index:04d}-seed{case["seed"]}',transport,detector,extractor,agent,True,root)
                        agent.completed_episodes=index;result['round']=index;history.append(result);log.write(json.dumps(result)+'\n')
                        agent.save(root/'last.pkl',meta)
                        write_json(root/'progress.json',dict(completed_rounds=index,planned_rounds=20,steps=agent.environment_steps,model_updates=agent.update_count,last_round=result))
                        print(f'ROUND {index} steps={result["steps"]} reward={result["reward_sum"]:.5f} model_updates={result["model_updates"]}/{agent.update_count} mse={result["fit_mse_xyz_mean"]}',flush=True)
                agent.save(root/'checkpoint-0020.pkl',meta)
                small=probe_cases();ids={c['seed'] for c in small};rest=[c for c in probe_cases(True) if c['seed'] not in ids]
                for label,plan in [('probe-0020',small),('probe-extra-0020',rest)]:
                    if time.time()-meta['repair_started_unix']>7200:raise ReviewRequired('Inherited two-hour budget exhausted before development group')
                    write_json(root/'status.json',dict(state='development_probe',probe=label))
                    results=evaluate(plan,root/label,transport,detector,extractor,agent,root);probes+=results
                    if not all(r['passed'] for r in results):raise ReviewRequired(f'{label} did not pass every case')
                write_json(root/'development-report.json',dict(passed=len(probes)==60 and all(r['passed'] for r in probes),total=len(probes),passed_cases=sum(r['passed'] for r in probes),episodes=probes,independent_validation='not_executed'))
            finally:transport.close()
        write_json(root/'status.json',dict(state='development_passed',completed_rounds=len(history),steps=agent.environment_steps,model_updates=agent.update_count,qualified=False))
    except BaseException as exc:
        agent.save(root/'interrupted.pkl',meta)
        write_json(root/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc(),completed_rounds=len(history),steps=agent.environment_steps,model_updates=agent.update_count))
        write_json(root/'status.json',dict(state='review_required' if isinstance(exc,ReviewRequired) else 'stopped_by_user' if isinstance(exc,InterruptedError) else 'anomaly',error=repr(exc),completed_rounds=len(history),steps=agent.environment_steps,model_updates=agent.update_count))
        raise
    finally:
        after={n:sha256(n) for n in before};write_json(root/'frozen-after.json',after)
        integrity=dict(frontend_unchanged=before==after,source_unchanged=all(sha256(REPRO/n)==d for n,d in meta['source_sha256'].items()) and all(sha256(ROOT/n)==d for n,d in meta['root_source_sha256'].items()),protocol_unchanged=sha256(HERE/'protocol.json')==meta['protocol_sha256'])
        write_json(root/'integrity-after.json',integrity)
        if not all(integrity.values()):raise RuntimeError('Frozen source/weight/protocol integrity failed')
if __name__=='__main__':main()
