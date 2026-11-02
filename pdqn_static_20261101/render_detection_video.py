"""Add frozen RGB detector boxes to saved realtime frames, after control ends.

This process sends no UE commands and does not load or modify the P-DQN agent.
Every displayed box is inferred from that same saved RGB frame, never GT.
"""
import bootstrap
from bootstrap import HERE,REPRO
import argparse,hashlib,json,subprocess,sys,time,warnings
from pathlib import Path
import cv2,numpy as np,torch
from control_model_v13.detector import create_detector
from integration.experiment import make_extractor
from hascam.cfff import CFFF

def lines(p):return [json.loads(s) for s in p.read_text(encoding='utf-8').splitlines()]
def write(p,v):p.write_text(json.dumps(v,indent=2),encoding='utf-8')

def encode(output,frames):
    sys.path.insert(0,str(REPRO/'verification/value-redesign-v10-20261031/video-tools'))
    from imageio_ffmpeg import get_ffmpeg_exe
    parts=['ffconcat version 1.0']
    for index,row in enumerate(frames):
        parts.append(f"file 'frames/{index:05d}.jpg'")
        if index+1<len(frames):parts.append(f"duration {frames[index+1]['wall_s']-row['wall_s']:.9f}")
    manifest=output/'frames.ffconcat';manifest.write_text('\n'.join(parts)+'\n',encoding='utf-8')
    startup=subprocess.STARTUPINFO();startup.dwFlags|=subprocess.STARTF_USESHOWWINDOW;startup.wShowWindow=0
    result=subprocess.run([get_ffmpeg_exe(),'-y','-loglevel','error','-safe','1','-f','concat','-i',str(manifest),
        '-vsync','vfr','-c:v','libx264','-crf','18','-preset','fast','-threads','2','-pix_fmt','yuv420p',
        '-movflags','+faststart',str(output/'live-detection-vfr.mp4')],startupinfo=startup,capture_output=True)
    if result.returncode:raise RuntimeError(result.stderr.decode(errors='replace'))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('folder',type=Path);args=p.parse_args()
    warnings.filterwarnings('ignore');torch.set_num_threads(4)
    root=args.folder.resolve();frames=lines(root/'capture-timestamps.jsonl');cfg=json.loads((root/'configuration.json').read_text(encoding='utf-8'))
    output=root/'detection-overlay';output.mkdir(exist_ok=False);(output/'frames').mkdir()
    detector=create_detector(checkpoint=REPRO/'weights/yolo26s_ue_overhead_v1.pt',device='cuda',threshold=.15,imgsz=960)
    assert detector.provenance['sha256']==cfg['detector']['sha256']=='1dd0de55c1791dce289cc7dc1d8eb6c50e488b26d38941f4c20845a6cadb5ae2'
    detector.warmup();extractor=make_extractor('cuda')
    assert extractor.sha256==cfg['reid']['weights_sha256']=='57fb9c17d88911ea64390bf5427f43511435e7f88f6eed9dbc969d4b611e53cd'
    selector=CFFF(extractor);selected_count=0;start=time.perf_counter()
    with (output/'detections.jsonl').open('w',encoding='utf-8',buffering=1) as log:
        for index,frame in enumerate(frames):
            raw=(root/f"frames/{frame['frame']:05d}.jpg").read_bytes()
            bgr=cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_COLOR)
            if bgr is None:raise RuntimeError('Saved RGB frame could not be decoded')
            rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB);selected=selector.choose(rgb,detector(rgb),count_step=False)
            row=dict(frame=frame['frame'],wall_s=frame['wall_s'],sim_time=frame['sim_time'],
                source_jpeg_sha256=hashlib.sha256(raw).hexdigest(),visible=selected is not None,
                bbox=list(selected.bbox) if selected else None,confidence=selected.confidence if selected else None)
            log.write(json.dumps(row)+'\n');h,w=bgr.shape[:2];cx,cy=w//2,h//2
            cv2.line(bgr,(cx-8,cy),(cx+8,cy),(255,180,40),1,cv2.LINE_AA)
            cv2.line(bgr,(cx,cy-8),(cx,cy+8),(255,180,40),1,cv2.LINE_AA)
            if selected:
                selected_count+=1;x1,y1,x2,y2=[int(round(v*s)) for v,s in zip(selected.bbox,(w,h,w,h))]
                cv2.rectangle(bgr,(x1,y1),(x2,y2),(60,235,60),2,cv2.LINE_AA)
                center=((x1+x2)//2,(y1+y2)//2)
                cv2.line(bgr,(cx,cy),center,(0,210,255),1,cv2.LINE_AA)
                label=f'car {selected.confidence:.2f}';ty=max(42,y1-8)
                cv2.putText(bgr,label,(x1,ty),cv2.FONT_HERSHEY_SIMPLEX,.48,(0,0,0),3,cv2.LINE_AA)
                cv2.putText(bgr,label,(x1,ty),cv2.FONT_HERSHEY_SIMPLEX,.48,(60,235,60),1,cv2.LINE_AA)
            cv2.rectangle(bgr,(0,0),(w,27),(24,24,24),-1)
            caption=f"RGB detector overlay | original realtime capture | {frame['wall_s']:.2f}s"
            cv2.putText(bgr,caption,(10,19),cv2.FONT_HERSHEY_SIMPLEX,.48,(235,235,235),1,cv2.LINE_AA)
            cv2.imwrite(str(output/f'frames/{index:05d}.jpg'),bgr,[cv2.IMWRITE_JPEG_QUALITY,95])
            if (index+1)%200==0 or index+1==len(frames):
                write(output/'progress.json',dict(completed=index+1,total=len(frames),visible=selected_count,wall_s=time.perf_counter()-start))
                print('DETECTION OVERLAY',index+1,'/',len(frames),flush=True)
    encode(output,frames)
    # GT is read only after rendering, for a separate same-frame audit.
    from independent_validation import iou
    detections=lines(output/'detections.jsonl');diagnostics=lines(root/'offline-diagnostics.jsonl')
    assert len(diagnostics)==len(detections)
    matches=[]
    for d,g in zip(detections,diagnostics):
        assert abs(d['sim_time']-g['sim_time'])<1e-8
        matches.append(iou(d['bbox'],g['gt_bbox_normalized']) if d['bbox'] and g.get('gt_bbox_normalized') else 0.)
    video=cv2.VideoCapture(str(output/'live-detection-vfr.mp4'));encoded=int(video.get(cv2.CAP_PROP_FRAME_COUNT));fps=video.get(cv2.CAP_PROP_FPS);video.release()
    wall=frames[-1]['wall_s']-frames[0]['wall_s'];duration=encoded/fps
    summary=dict(completed=True,source_capture=str(root),checkpoint_sha256=cfg['checkpoint_sha256'],
        source_configuration_sha256=hashlib.sha256((root/'configuration.json').read_bytes()).hexdigest(),
        detector_sha256=detector.provenance['sha256'],reid_sha256=extractor.sha256,
        scope='Offline same-frame RGB detector/ReID overlay on unchanged realtime recording; no GT boxes drawn and no control changes',
        raw_frames=len(frames),visible_frames=selected_count,encoded_frames=encoded,wall_duration_s=wall,encoded_duration_s=duration,
        same_frame_match_fraction=sum(v>=.3 for v in matches)/len(matches),mean_gt_iou=float(np.mean(matches)),minimum_gt_iou=min(matches),
        timestamps_unchanged=True,encoded_duration_matches=abs(duration-wall)<.2,video=str(output/'live-detection-vfr.mp4'))
    write(output/'provenance-and-audit.json',summary)
    assert summary['same_frame_match_fraction']>=.98 and summary['encoded_duration_matches']
    print('BOXED VIDEO',output/'live-detection-vfr.mp4',flush=True)

if __name__=='__main__':main()
