"""Bounded RGB visibility/scale and fixed zoom-pulse preflight; no learner."""
import json
from pathlib import Path
import time
import torch
import cv2
from .scenes import cases,layout,materialize,stage_spec,initial_zone,STAGE,load_protocol
from .environment import ZoomEnvironment
from control_value_v10.run import check_exclusive
from control_coverage_v8.run import write_json,sha256
from control_learning.environment import AtomicTransport
from integration.runtime import ROOT,REPRO,StandaloneArena
from .detector import create_detector
from integration.experiment import make_extractor

OUT=REPRO/'verification/active-zoom-v11-20261031/calibration-rgb-pyramid-v2'


def main():
    check_exclusive();OUT.mkdir(exist_ok=False,parents=True);torch.set_num_threads(4)
    expected=load_protocol()['frozen_frontend']
    detector=create_detector(checkpoint=REPRO/expected['detector'],device='cuda')
    extractor=make_extractor('cuda')
    assert detector.provenance['sha256']==expected['detector_sha256']
    assert extractor.metadata()['weights_sha256']==expected['reid_sha256']
    result=dict(passed=False,purpose='Initial RGB and fixed command calibration; no control success or training',initials=[],pulses=[],actual_actions=0)
    try:
        detector.warmup()
        with StandaloneArena(OUT,headless=True) as arena:
            transport=AtomicTransport(arena.port,hud=False)
            try:
                for case in cases(STAGE,'calibration'):
                    folder=OUT/f'seed{case["seed"]}';scene=materialize(STAGE,'calibration',case['seed'],folder)
                    env=ZoomEnvironment(transport,detector,extractor,stage_spec(STAGE))
                    state=env.reset(scene,case['seed']);row=env.rows[0]
                    item=dict(seed=case['seed'],coverage=layout(case['seed'],'calibration'),observation=row,
                              initial_zone_passed=initial_zone(row,case['seed'],'calibration'),state=state.tolist())
                    result['initials'].append(item);write_json(folder/'initial.json',item)
                    cv2.imwrite(str(folder/'rgb.png'),cv2.cvtColor(transport.rgb,cv2.COLOR_RGB2BGR))
                    write_json(OUT/'progress.json',dict(initials=len(result['initials']),planned=60,failures=sum(not x['initial_zone_passed'] for x in result['initials'])))
                # All seven commands checked in a centered 3x case, reset independently.
                centered=next(x for x in result['initials'] if x['coverage']['layer']==0 and x['coverage']['scale_group']==3)
                seed=centered['seed']
                for action in range(7):
                    folder=OUT/f'pulse-{action}';scene=materialize(STAGE,'calibration',seed,folder)
                    env=ZoomEnvironment(transport,detector,extractor,stage_spec(STAGE));env.reset(scene,seed)
                    initial=env.rows[0]
                    for _ in range(15):
                        _,reward,_,row=env.step(action,.04);result['actual_actions']+=1
                    entry=dict(action=action,steps=15,before=initial,after=row,area_change=row['area_ratio']-initial['area_ratio'] if row['visible'] else None)
                    result['pulses'].append(entry);write_json(folder/'report.json',entry)
                    (folder/'steps.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in env.rows),encoding='utf-8')
                    cv2.imwrite(str(folder/'rgb.png'),cv2.cvtColor(transport.rgb,cv2.COLOR_RGB2BGR))
            finally:transport.close()
        groups={str(g):[x['observation']['area_ratio'] for x in result['initials'] if x['coverage']['scale_group']==g and x['observation']['visible']] for g in range(5)}
        means={g:sum(values)/len(values) if values else None for g,values in groups.items()}
        result['mean_initial_area_by_scale']=means
        result['checks']=dict(all60_visible_full_box_and_layout=all(x['initial_zone_passed'] for x in result['initials']),
            increasing_area=all(means[str(g)] is not None and means[str(g+1)]>means[str(g)] for g in range(4)),
            includes_too_small=any(x['observation']['area_ratio']<.025 for x in result['initials'] if x['observation']['visible']),
            includes_too_large=any(x['observation']['area_ratio']>.055 for x in result['initials'] if x['observation']['visible']),
            zoom_in_area_increases=result['pulses'][5]['area_change'] is not None and result['pulses'][5]['area_change']>0,
            zoom_out_area_decreases=result['pulses'][6]['area_change'] is not None and result['pulses'][6]['area_change']<0)
        result['passed']=all(result['checks'].values())
    except BaseException as exc:
        result['error']=repr(exc);raise
    finally:
        result['frontend_unchanged']=sha256(detector.checkpoint)==expected['detector_sha256'] and sha256(extractor.weights)==expected['reid_sha256']
        write_json(OUT/'report.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('initials','pulses')},ensure_ascii=True))
    if not result['passed']:raise RuntimeError('Calibration requires diagnosis; training remains blocked')


if __name__=='__main__':main()
