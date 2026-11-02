"""Two-class RGB detector adaptation with whole truck supervision only in training."""
import speed_bootstrap
from speed_bootstrap import REV, REPRO
import os, json, shutil, hashlib, time
from pathlib import Path
os.environ['YOLO_AUTOINSTALL'] = 'false'
os.environ.setdefault('YOLO_CONFIG_DIR', str(REPRO/'verification/pretrained-detectors-gpu/ultralytics-settings'))
import torch
from ultralytics import YOLO

ROOT = REV/'unified_motion_20261101/whole_vehicle_detector_v1'

def main():
    from train import exclusive
    exclusive()
    torch.set_num_threads(4)
    ROOT.mkdir(exist_ok=False)
    data = ROOT/'data'
    for split in ['train','val']:
        (data/'images'/split).mkdir(parents=True)
        (data/'labels'/split).mkdir(parents=True)
    records = []
    old = REV/'unified_motion_20261101/detector_data_v1'
    for split in ['train','val']:
        for image in sorted((old/'images'/split).glob('*.jpg')):
            stem = 'old_'+image.stem
            shutil.copy2(image, data/'images'/split/(stem+'.jpg'))
            shutil.copy2(old/'labels'/split/(image.stem+'.txt'), data/'labels'/split/(stem+'.txt'))
            records.append(dict(split=split,source=str(image),type='prior_car_data'))
    sources = [('train','user-crossroad-truck-occlusion-video',3,9.),
               ('train','user-crossroad-purple-distance-video',12,18.),
               ('val','user-crossroad-reacquire-area2-video',8,12.)]
    for index,(split,name,stride,end) in enumerate(sources):
        folder = REV/'unified_motion_20261101'/name
        diagnostics = [json.loads(x) for x in (folder/'offline-diagnostics.jsonl').read_text().splitlines()]
        for i,d in enumerate(diagnostics):
            if i%stride or d['sim_time']>end:
                continue
            lines = []
            instances = []
            for target in d['targets']:
                if not target.get('gt_in_frame') or target.get('gt_center_occluded'):
                    continue
                b = target.get('gt_bbox_clipped_normalized')
                if not b:
                    continue
                x1,y1,x2,y2 = b
                if (x2-x1)*960<5 or (y2-y1)*540<5:
                    continue
                cls = 1 if target['id'].startswith('truck_') else 0
                lines.append(f'{cls} {(x1+x2)/2:.9f} {(y1+y2)/2:.9f} {x2-x1:.9f} {y2-y1:.9f}')
                instances.append(dict(class_id=cls,box=b))
            stem = f'new{index}_{i:05d}'
            image = folder/f'frames/{i:05d}.jpg'
            shutil.copy2(image,data/'images'/split/(stem+'.jpg'))
            (data/'labels'/split/(stem+'.txt')).write_text('\n'.join(lines)+'\n' if lines else '',encoding='utf-8')
            records.append(dict(split=split,source=str(image),type='whole_vehicle',instances=instances))
    yaml = data/'data.yaml'
    yaml.write_text(f'path: {data.as_posix()}\ntrain: images/train\nval: images/val\nnames:\n  0: car\n  1: truck\n',encoding='utf-8')
    counts = {s:sum(r['split']==s for r in records) for s in ['train','val']}
    instances = {s:sum(sum(t['class_id']==1 for t in r.get('instances',[])) for r in records if r['split']==s) for s in counts}
    provenance = dict(counts=counts,truck_instances=instances,names={0:'car',1:'truck'},
        box_definition='Native actor whole silhouette projection bounding box including chassis,cab,cargo. Hidden-centre actors excluded; partial boxes clipped to RGB image.',
        scope='Simulator labels only for separate detector supervised training/heldout diagnostics. Control/ReID use RGB predictions only. No GT box enlargement or oracle inference.',
        split='Truck train=old failed recording. Heldout=separate area2 recording; no frame shared across splits.',sources=records)
    (data/'provenance.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')
    parent = REV/'unified_motion_20261101/detector_longroad_v2.pt'
    model = YOLO(str(parent),task='detect')
    model.train(data=str(yaml),epochs=5,imgsz=960,batch=4,device=0,workers=0,amp=False,plots=False,cache=False,
        project=str(ROOT/'training'),name='car-truck',exist_ok=False,seed=20261102,deterministic=True,
        freeze=10,lr0=.0005,lrf=.1,warmup_epochs=.5,mosaic=.1,close_mosaic=0,translate=.1,scale=.3,fliplr=.5)
    best = ROOT/'training/car-truck/weights/best.pt'
    output = ROOT/'whole_vehicle.pt'
    shutil.copy2(best,output)
    output.with_suffix('.source.json').write_text(json.dumps(dict(adapted_to_UE=True,parent=str(parent),
        sha256=hashlib.sha256(output.read_bytes()).hexdigest(),supervision='Whole car/truck boxes in separate supervised RGB detector training only; no control GT.',
        dataset=str(data/'provenance.json'),counts=counts,truck_instances=instances,epochs=5,utc=time.time()),indent=2),encoding='utf-8')
    print('WHOLE VEHICLE DETECTOR COMPLETE', output, counts, instances, flush=True)

if __name__=='__main__':main()
