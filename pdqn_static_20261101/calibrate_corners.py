"""32 fresh true-corner starts, complete boxes at four scales; no control policy."""
import bootstrap
from bootstrap import HERE,REPRO
from pathlib import Path
import argparse,json
import cv2,torch,numpy as np
from evaluation_scenes import calibration_corner
from scenes import materialize
from train import exclusive
from author_aligned.environment import AtomicTransport
from integration.runtime import StandaloneArena
from control_model_v13.detector import create_detector
from integration.experiment import make_extractor
from calibrate import measure,write

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=HERE/'corner-layout-calibration');args=p.parse_args()
    exclusive();torch.set_num_threads(4);out=args.output;out.mkdir(exist_ok=False)
    detector=create_detector(checkpoint=REPRO/'weights/yolo26s_ue_overhead_v1.pt',device='cuda',threshold=.15,imgsz=960)
    detector.warmup();extractor=make_extractor('cuda');results=[]
    with StandaloneArena(out,headless=True) as arena:
        t=AtomicTransport(arena.port,hud=False)
        try:
            for index in range(32):
                item=calibration_corner(index);scene=materialize(out/'scenes',item)
                t.scenario(scene);t.reset(item['seed']);row=measure(t,detector,extractor)|item
                if row['bbox']:
                    x1,y1,x2,y2=row['bbox'];row['border_gaps_pixels']=[x1*960,y1*540,(1-x2)*960,(1-y2)*540]
                    row['full_box']=all(.003<v<.997 for v in row['bbox'])
                else:row['full_box']=False;row['border_gaps_pixels']=None
                gt=t.diagnostics.get('gt_bbox_normalized')
                row['offline_gt_bbox']=gt
                row['geometry_full_box']=bool(gt and all(0.<=v<=1. for v in gt))
                cv2.imwrite(str(out/f'corner-{index:02d}.jpg'),cv2.cvtColor(t.rgb,cv2.COLOR_RGB2BGR))
                results.append(row);write(out/'results.json',results)
                print('CORNER',index,row['visible'],row['full_box'],row['center'],row['area'],row['border_gaps_pixels'],flush=True)
        finally:t.close()
    canvas=np.full((4*198,4*320,3),245,np.uint8)
    for index in range(16):
        im=cv2.imread(str(out/f'corner-{index:02d}.jpg'));im=cv2.resize(im,(320,180))
        y,x=index//4*198,index%4*320;canvas[y:y+180,x:x+320]=im
        cv2.putText(canvas,f'q{index%4} scale{index//4}',(x+7,y+193),cv2.FONT_HERSHEY_SIMPLEX,.4,(20,20,20),1)
    cv2.imwrite(str(out/'four-corners-four-scales.jpg'),canvas)
    write(out/'summary.json',dict(count=len(results),visible=sum(r['visible'] for r in results),
        full_box=sum(r['full_box'] for r in results),geometry_full_box=sum(r['geometry_full_box'] for r in results),
        passed=all(r['full_box'] and r['geometry_full_box'] for r in results),
        scope='Real layout calibration, no policy/training/validation claim'))

if __name__=='__main__':main()
