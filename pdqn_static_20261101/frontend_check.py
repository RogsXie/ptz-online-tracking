"""Compare existing frozen frontend preprocessing on the identical real RGBs."""
import bootstrap
from bootstrap import HERE,REPRO
from calibrate import write
from control_model_v13.detector import create_detector
import cv2,torch,numpy as np

def main():
    torch.set_num_threads(4)
    detector=create_detector(checkpoint=REPRO/'weights/yolo26s_ue_overhead_v1.pt',device='cuda',threshold=.15,imgsz=960)
    detector.warmup();rows=[]
    for index in range(80):
        rgb=cv2.cvtColor(cv2.imread(str(HERE/f'calibration-v1/initial-{index:03d}.jpg')),cv2.COLOR_BGR2RGB)
        boxes=detector(rgb)
        rows.append(dict(index=index,detections=[dict(bbox=d.bbox,confidence=d.confidence) for d in boxes],seconds=detector.last_seconds))
    write(HERE/'frontend-comparison.json',dict(profile=detector.provenance,rows=rows,visible=sum(bool(r['detections']) for r in rows)))
    print('FULL+HALF visible',sum(bool(r['detections']) for r in rows),'of',len(rows))

if __name__=='__main__':main()
