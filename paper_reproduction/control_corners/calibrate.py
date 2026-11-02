"""Real-RGB starts and existing frozen v5 policy; no training or GT actions."""
import argparse
import json
from pathlib import Path
import cv2
import torch
from control_corners.scenes import materialize,stage_spec,initial_zone,CORNER_NAMES
from control_corners.environment import CornerEnvironment
from control_corners.gates import assess_episode
from control_v5.agent import ContinuingAgent
from control_v5.run import frozen,write_json
from control_learning.environment import AtomicTransport
from integration.runtime import StandaloneArena,REPRO
from integration.pretrained_detector import create_detector
from integration.experiment import make_extractor


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--initial-only',action='store_true');args=parser.parse_args()
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4)
    detector,extractor=create_detector(device='cuda'),make_extractor('cuda');detector.warmup()
    agent=ContinuingAgent.load(REPRO/'runs/continuing-static-v5-attempt1/last.pt',device='cuda')
    cases=[];receipt={}
    with StandaloneArena(out,headless=True) as arena:
        transport=AtomicTransport(arena.port,hud=False)
        try:
            with frozen(agent,receipt):
                for index,name in enumerate(CORNER_NAMES):
                    folder=out/name;seed=718950+index
                    scene=materialize('static_center','train',seed,folder)
                    env=CornerEnvironment(transport,detector,extractor,stage_spec('static_center'),legacy_observation=True)
                    state=env.reset(scene,seed);initial=env.rows[0]
                    initial['initial_zone_confirmed']=initial_zone(initial,seed,'train')
                    cv2.imwrite(str(folder/'initial-rgb.jpg'),cv2.cvtColor(transport.rgb,cv2.COLOR_RGB2BGR))
                    result={'corner':name,'seed':seed,'initial':initial,'zone_passed':initial['initial_zone_confirmed']}
                    print(json.dumps({'corner':name,'zone_passed':result['zone_passed'],'center':[(initial['x']+1)/2,(initial['y']+1)/2] if initial['visible'] else None,'bbox':initial['selected_bbox']},ensure_ascii=False),flush=True)
                    if not args.initial_only and result['zone_passed']:
                        midway=False
                        with (folder/'steps.jsonl').open('w',encoding='utf-8',buffering=1) as log:
                            log.write(json.dumps(initial|{'initial':True,'state':state.tolist()})+'\n')
                            for decision in range(512):
                                action=agent.select_action(state,False)
                                nxt,reward,done,row=env.step(*action)
                                log.write(json.dumps(row|{'input_state':state.tolist(),'next_state':nxt.tolist(),'policy_selection':agent.last_selection.copy(),'decision':decision})+'\n')
                                state=nxt
                                if not midway and row['sim_time_s']>=4.:
                                    cv2.imwrite(str(folder/'mid-rgb.jpg'),cv2.cvtColor(transport.rgb,cv2.COLOR_RGB2BGR));midway=True
                                if done:break
                        cv2.imwrite(str(folder/'final-rgb.jpg'),cv2.cvtColor(transport.rgb,cv2.COLOR_RGB2BGR))
                        result['quality']=assess_episode('static_center',env.rows)
                        print('BASELINE',name,result['quality']['passed'],result['quality']['terminal_center_mean'],flush=True)
                    write_json(folder/'report.json',result);cases.append(result)
        finally:transport.close()
    write_json(out/'report.json',{'purpose':'RGB initial coverage calibration' if args.initial_only else 'frozen v5 out-of-distribution four-corner baseline; original clipped observations; not trained corner controller',
        'initial_only':args.initial_only,'cases':cases,'all_start_zones_confirmed':all(c['zone_passed'] for c in cases),
        'four_network_freeze':receipt,'training_updates_performed':0,'final_frames_are_stable_only_if_quality_passes':True})


if __name__=='__main__':main()
