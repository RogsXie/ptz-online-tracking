"""Separate supervised detector adaptation, never a GT-driven controller."""
import speed_bootstrap
from speed_bootstrap import REV,REPRO
import os,json,shutil,hashlib,time
from pathlib import Path
os.environ['YOLO_AUTOINSTALL']='false'
os.environ.setdefault('YOLO_CONFIG_DIR',str(REPRO/'verification/pretrained-detectors-gpu/ultralytics-settings'))
import torch
from ultralytics import YOLO

def main():
    from train import exclusive
    exclusive();torch.set_num_threads(4);root=REV/'unified_motion_20261101';root.mkdir(exist_ok=True);data=root/'detector_data_v1';data.mkdir(exist_ok=False)
    sources=[('train',REV/'user-requested-9ms-long-road-mask-video',5),('train',REV/'user-requested-9ms-association-video',5),
        ('train',REV/'user-requested-uniform-motion-stop-video',8),('train',REV/'motion-tracking-v5-latency-video',8),
        ('train',REV/'user-requested-tiny-optics-color-video',8),('train',REV/'user-requested-zoom15-video',8),
        ('val',REV/'user-requested-9ms-long-road-video',8)]
    for folder in sorted((REV.parent/'release-live-evidence').glob('*')):
        if (folder/'offline-diagnostics.jsonl').exists():sources.append(('train',folder,40))
    counts={'train':0,'val':0};negatives={'train':0,'val':0};provenance=[]
    for source_index,(split,folder,stride) in enumerate(sources):
        diagnostics=[json.loads(s) for s in (folder/'offline-diagnostics.jsonl').read_text(encoding='utf-8').splitlines()]
        for i,d in enumerate(diagnostics):
            if i%stride:continue
            stem=f's{source_index:02d}_f{i:05d}';images=data/'images'/split;labels=data/'labels'/split;images.mkdir(parents=True,exist_ok=True);labels.mkdir(parents=True,exist_ok=True)
            source=folder/f'frames/{i:05d}.jpg';shutil.copy2(source,images/f'{stem}.jpg');box=d.get('gt_bbox_normalized');text=''
            if d.get('gt_in_frame') and box:
                x1,y1,x2,y2=[min(1.,max(0.,float(v))) for v in box]
                if x2>x1 and y2>y1:text=f'0 {(x1+x2)/2:.9f} {(y1+y2)/2:.9f} {x2-x1:.9f} {y2-y1:.9f}\n'
            if not text:negatives[split]+=1
            (labels/f'{stem}.txt').write_text(text,encoding='utf-8');counts[split]+=1
            provenance.append(dict(source=str(source),split=split,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),label_kind='UEdetectortrainingonly;notpolicyinput'))
    yaml=data/'data.yaml';yaml.write_text(f'path: {data.as_posix()}\ntrain: images/train\nval: images/val\nnames:\n  0: car\n',encoding='utf-8')
    (data/'provenance.json').write_text(json.dumps(dict(counts=counts,negative_frames=negatives,scope='Separate detector supervised adaptation; allGT use confined to detectorlabels, never PNcontrolor RLreward',sources=provenance),indent=2))
    print('DETECTOR DATA',counts,negatives,flush=True)
    parent=REPRO/'weights/yolo26s_ue_overhead_v1.pt';model=YOLO(str(parent),task='detect')
    model.train(data=str(yaml),epochs=5,imgsz=960,batch=4,device=0,workers=0,amp=False,plots=False,cache=False,
        project=str(root/'detector_training'),name='longroad-v2',exist_ok=False,seed=20261101,deterministic=True,
        freeze=10,lr0=.0003,lrf=.1,warmup_epochs=.5,mosaic=.1,close_mosaic=0,translate=.1,scale=.3,fliplr=.5)
    best=root/'detector_training/longroad-v2/weights/best.pt';assert best.is_file();out=root/'detector_longroad_v2.pt';shutil.copy2(best,out)
    out.with_suffix('.source.json').write_text(json.dumps(dict(adapted_to_UE=True,parent=str(parent),sha256=hashlib.sha256(out.read_bytes()).hexdigest(),supervision='OnlyseparateYOLOdetector usesUEbboxlabels; frozenbefore P-DQNtraining',dataset=str(data/'provenance.json'),counts=counts,negative_frames=negatives,epochs=5,utc=time.time()),indent=2))
    print('NEW FROZEN DETECTOR',out,flush=True)
if __name__=='__main__':main()
