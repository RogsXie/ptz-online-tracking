"""Read-only development-scene coverage audit; never accesses validation RGB."""
import json
import cv2
import torch
from integration.runtime import StandaloneArena,REPRO
from integration.pretrained_detector import create_detector
from integration.experiment import make_extractor
from control_learning.environment import AtomicTransport
from control_corners.environment import CornerEnvironment
from control_corners.scenes import materialize,stage_spec,initial_zone,category
from control_corners.run import write_json


def main():
    torch.set_num_threads(4)
    out=REPRO/'verification/four-corners-v7-20261031/initial-perception-audit'
    out.mkdir(exist_ok=False);detector,extractor=create_detector(device='cuda'),make_extractor('cuda');detector.warmup()
    results=[]
    with StandaloneArena(out,headless=True) as arena:
        transport=AtomicTransport(arena.port,hud=False)
        try:
            for seed in [711003,718953]+list(range(711000,711032))+list(range(719100,719112)):
                folder=out/f'case{len(results):03d}-seed{seed}';scene=materialize('static_center','train',seed,folder)
                env=CornerEnvironment(transport,detector,extractor,stage_spec('static_center'))
                env.reset(scene,seed);row=env.rows[0]
                cv2.imwrite(str(folder/'initial-rgb.jpg'),cv2.cvtColor(transport.rgb,cv2.COLOR_RGB2BGR))
                record={'seed':seed,'kind':category(seed,'train')[0],'zone_passed':initial_zone(row,seed,'train'),'observation':row,
                    'detector_candidates':[{'bbox':list(d.bbox),'confidence':float(d.confidence)} for d in env.detections],
                    'diagnostics':transport.diagnostics,'image':str(folder/'initial-rgb.jpg')}
                results.append(record);write_json(folder/'report.json',record)
                print(json.dumps({'seed':seed,'passed':record['zone_passed'],'candidates':record['detector_candidates']}),flush=True)
        finally:transport.close()
    write_json(out/'report.json',{'scope':'Development scene initial perception; no gradient updates, no heldout validation rendered','cases':results,
        'passed':sum(x['zone_passed'] for x in results),'total':len(results),'frontend':detector.provenance})


if __name__=='__main__':main()
