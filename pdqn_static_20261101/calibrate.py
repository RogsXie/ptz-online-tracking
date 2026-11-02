"""Actual UE frames, independent action-sign/duration checks, and RGB noise."""
import bootstrap
from bootstrap import HERE,REPRO
import json,time
from pathlib import Path
import cv2,numpy as np,torch
from scenes import layout,materialize
from author_aligned.environment import AtomicTransport
from integration.runtime import StandaloneArena
from integration.pretrained_detector import create_detector
from integration.experiment import make_extractor
from hascam.cfff import CFFF
from integration.transport import ACTION_RATES

def write(path,obj):
    path.write_text(json.dumps(obj,indent=2,allow_nan=False),encoding='utf-8')

def frontend():
    torch.set_num_threads(4)
    detector=create_detector(checkpoint=REPRO/'weights/yolo26s_ue_overhead_v1.pt',device='cuda',threshold=.15,imgsz=960)
    assert detector.provenance['sha256']=='1dd0de55c1791dce289cc7dc1d8eb6c50e488b26d38941f4c20845a6cadb5ae2'
    detector.warmup()
    extractor=make_extractor('cuda')
    assert extractor.sha256=='57fb9c17d88911ea64390bf5427f43511435e7f88f6eed9dbc969d4b611e53cd'
    return detector,extractor

def measure(t,detector,extractor):
    start=time.perf_counter();boxes=detector(t.rgb);detect_s=time.perf_counter()-start
    start=time.perf_counter();selected=CFFF(extractor).choose(t.rgb,boxes,count_step=False)
    reid_s=time.perf_counter()-start
    return dict(visible=selected is not None,bbox=list(selected.bbox) if selected else None,
        center=list(selected.center) if selected else None,area=selected.area if selected else None,
        confidence=selected.confidence if selected else 0.,detector_s=detect_s,reid_s=reid_s)

def main():
    out=HERE/'calibration-v1';out.mkdir(exist_ok=False)
    detector,extractor=frontend()
    write(out/'frontend.json',dict(detector=detector.provenance,reid=extractor.metadata(),torch=torch.__version__,gpu=torch.cuda.get_device_name()))
    starts=[];responses=[];noise=[]
    with StandaloneArena(out,headless=True) as arena:
        t=AtomicTransport(arena.port,hud=False)
        try:
            for index in range(80):
                item=layout('calibration',index);scene=materialize(out/'scenes',item)
                t.scenario(scene);t.reset(item['seed'])
                row=measure(t,detector,extractor)|item
                row['full_box']=bool(row['bbox'] and all(.003<v<.997 for v in row['bbox']))
                starts.append(row)
                cv2.imwrite(str(out/f'initial-{index:03d}.jpg'),cv2.cvtColor(t.rgb,cv2.COLOR_RGB2BGR))
                write(out/'starts.json',starts)
                print('START',index,row['visible'],row['full_box'],row['center'],row['area'],flush=True)
            # Same fixed target and camera for all seven independent action probes.
            for fov in (14.,20.,30.,60.):
                item=layout('calibration',0)|dict(planned_center=[.5,.5],fov_deg=fov)
                scene=materialize(out/f'physics-fov{fov:g}',item)
                for action in range(7):
                    for duration in (.06,1.):
                        t.scenario(scene);t.reset(item['seed']);before=measure(t,detector,extractor)
                        diag0=t.diagnostics.copy();p,y,z=ACTION_RATES[action]
                        wall=time.perf_counter();t.rpc(f'vset /ptz/step {p} {y} {z} {duration}')
                        communication_s=time.perf_counter()-wall
                        wall=time.perf_counter();t.capture();capture_s=time.perf_counter()-wall
                        after=measure(t,detector,extractor)
                        row=dict(fov=fov,action=action,duration=duration,before=before,after=after,
                            rotation_before=diag0['rotation'],rotation_after=t.diagnostics['rotation'],
                            fov_after=t.diagnostics['fov'],measured_s=t.diagnostics['sim_time']-diag0['sim_time'],
                            communication_s=communication_s,capture_s=capture_s)
                        assert abs(row['measured_s']-duration)<1e-5
                        responses.append(row);write(out/'responses.json',responses)
                t.scenario(scene);t.reset(item['seed'])
                for _ in range(15):
                    t.capture();noise.append(dict(fov=fov,**measure(t,detector,extractor)))
            write(out/'noise.json',noise)
        finally:t.close()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(9.6,5.4))
    for scale in range(4):
        rows=[r for r in starts if r['scale_group']==scale and r['visible']]
        ax.scatter([r['center'][0]*960 for r in rows],[r['center'][1]*540 for r in rows],s=45,label=f'initial FOV group {scale}')
    ax.set(xlim=(0,960),ylim=(540,0),xlabel='Detected center x (pixels)',ylabel='Detected center y (pixels)',title='80 actual UE starting frames: 4 quadrants x 5 offsets x 4 FOV groups')
    ax.axvline(480,color='gray',alpha=.3);ax.axhline(270,color='gray',alpha=.3);ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(out/'coverage.png',dpi=160);plt.close(fig)
    # Contact sheet uses original sensor frames without reconstructed target boxes.
    canvas=np.full((4*192,5*320,3),245,np.uint8)
    for k,index in enumerate(range(0,80,4)):
        im=cv2.imread(str(out/f'initial-{index:03d}.jpg'));im=cv2.resize(im,(320,180))
        y,x=k//5*192,k%5*320;canvas[y:y+180,x:x+320]=im
        cv2.putText(canvas,f'case {index}',(x+5,y+190),cv2.FONT_HERSHEY_SIMPLEX,.35,(20,20,20),1)
    cv2.imwrite(str(out/'raw-contact-sheet.jpg'),canvas)
    write(out/'summary.json',dict(visible=sum(r['visible'] for r in starts),full_box=sum(r['full_box'] for r in starts),count=len(starts),responses=len(responses)))

if __name__=='__main__':main()
