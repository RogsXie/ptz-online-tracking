"""Same-frame RGB boxes and unchanged wall timing; GT only after rendering."""
import speed_bootstrap
import argparse,json,math,hashlib,time,warnings
from types import SimpleNamespace
from pathlib import Path
import cv2,numpy as np,torch
from unified_core import frontend
from unified_perception import Selector
from hascam.cfff import CFFF
from render_centered_detection_video import encode,lines,write

def main():
    p=argparse.ArgumentParser();p.add_argument('folder',type=Path);args=p.parse_args();root=args.folder.resolve()
    warnings.filterwarnings('ignore');torch.set_num_threads(4);cfg=json.loads((root/'configuration.json').read_text());frames=lines(root/'capture-timestamps.jsonl')
    out=root/'detection-overlay';out.mkdir(exist_ok=False);(out/'frames').mkdir()
    if cfg.get('whole_vehicle_detector',False):
        from whole_vehicle_frontend import frontend as whole_frontend
        detector,extractor=whole_frontend(cfg['detector']['checkpoint'])
    else:detector,extractor=frontend()
    assert detector.provenance['sha256']==cfg['detector']['sha256']
    assert extractor.sha256==cfg['reid']['weights_sha256']
    # The initial controller observation is an RGB detection, never a GT box.
    initial=json.loads((root/'initial-observation.json').read_text());scene=json.loads((root/'scene.json').read_text())
    t=SimpleNamespace(diagnostics=dict(sim_time=0.,fov=scene['camera']['fov_deg'],rotation=scene['camera']['rotation_deg']))
    selector=Selector(CFFF(extractor),t,detector)
    if cfg.get('control_reference_overlay',False):
        from control_reference_overlay import Selector as ControlReferenceSelector
        selector=ControlReferenceSelector(CFFF(extractor),t,detector,root)
    elif cfg.get('target_selection')=='highest_confidence_on_loss_v1':
        from confidence_reacquire import Selector as ReacquireSelector
        selector=ReacquireSelector(CFFF(extractor),t,detector)
    from hascam.detection import Detection
    if initial['visible']:
        # Re-read its crop from the initial saved image is impossible; use only
        # its recorded RGB-derived bearing/area to initialize association history.
        from unified_perception import bearing
        center=((initial['x']+1)/2,(initial['y']+1)/2);a=bearing(center,t.diagnostics)
        selector.history.append(dict(time=0.,bearing=a,angular_area=initial['area']*math.tan(math.radians(t.diagnostics['fov']/2))**2))
        detector.track=dict(time=0.,bearing=a,velocity=np.zeros(2),area=initial['area'])
    with (out/'detections.jsonl').open('w',encoding='utf-8') as log:
        for index,frame in enumerate(frames):
            t.diagnostics=frame;raw=(root/f"frames/{frame['frame']:05d}.jpg").read_bytes();bgr=cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_COLOR)
            rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB);candidates=detector(rgb);s=selector.choose(rgb,candidates,count_step=False)
            e=math.hypot(*(2*v-1 for v in s.center)) if s else None;centered=bool(s and e<=.025);color=(0,235,255) if centered else (60,235,60)
            distance_px=math.hypot((s.center[0]-.5)*960,(s.center[1]-.5)*540) if s else None
            row=dict(frame=frame['frame'],sim_time=frame['sim_time'],wall_s=frame['wall_s'],visible=s is not None,bbox=list(s.bbox) if s else None,
                center_error=e,centered=centered,distance_to_image_center_px=distance_px,confidence=s.confidence if s else None,
                label=s.label if s else None,class_id=s.class_id if s else None,source_jpeg_sha256=hashlib.sha256(raw).hexdigest())
            if cfg.get('target_selection') in ('highest_confidence_on_loss_v1','locked_identity_timeout_v1'):row['association']=selector.info.copy()
            if cfg.get('explicit_image_area_constraint',False):
                reliable=[d for d in candidates if d.confidence>=.5 and d.label in ('car','truck') and d.bbox[0]>.01 and d.bbox[1]>.01 and d.bbox[2]<.99 and d.bbox[3]<.99]
                row.update(max_reliable_vehicle_area=max([d.area for d in reliable]+([s.area] if s else []),default=0.),reliable_vehicle_count=len(reliable))
            log.write(json.dumps(row)+'\n');h,w=bgr.shape[:2];cx,cy=w//2,h//2
            cv2.drawMarker(bgr,(cx,cy),(255,180,40),cv2.MARKER_CROSS,16,1)
            if s:
                x1,y1,x2,y2=[int(round(v*z)) for v,z in zip(s.bbox,(w,h,w,h))];cv2.rectangle(bgr,(x1,y1),(x2,y2),color,2,cv2.LINE_AA)
                if cfg.get('overlay_target_distance',False):
                    target=(int(round(s.center[0]*w)),int(round(s.center[1]*h)));yellow=(0,235,255)
                    cv2.line(bgr,(cx,cy),target,yellow,2,cv2.LINE_AA);cv2.circle(bgr,target,3,yellow,-1,cv2.LINE_AA)
                    label=f'{distance_px:.1f} px';size=cv2.getTextSize(label,cv2.FONT_HERSHEY_SIMPLEX,.52,1)[0]
                    location=(min(w-size[0]-6,max(8,(cx+target[0])//2+10)),min(h-8,max(45,(cy+target[1])//2-10)))
                    cv2.putText(bgr,label,location,cv2.FONT_HERSHEY_SIMPLEX,.52,(0,0,0),3,cv2.LINE_AA)
                    cv2.putText(bgr,label,location,cv2.FONT_HERSHEY_SIMPLEX,.52,yellow,1,cv2.LINE_AA)
                text=f'{s.label} {s.confidence:.2f}'+(' CENTERED' if centered else '')
                if selector.info.get('overlay_observation_source')=='verified_optical_flow':text=f'{s.label} TRACKED'+(' CENTERED' if centered else '')
                cv2.putText(bgr,text,(x1,max(45,y1-8)),cv2.FONT_HERSHEY_SIMPLEX,.48,(0,0,0),3,cv2.LINE_AA)
                cv2.putText(bgr,text,(x1,max(45,y1-8)),cv2.FONT_HERSHEY_SIMPLEX,.48,color,1,cv2.LINE_AA)
            mult=1.
            for boundary,value in cfg['motion']['speed_profile']:
                if frame['sim_time']>=boundary:mult=value
            speed=cfg['motion']['speed_m_s']*mult
            caption=f"{speed:.2f} m/s | Yellow=centered | Original realtime | {frame['wall_s']:.2f}s"
            if cfg.get('target_selection')=='highest_confidence_on_loss_v1':
                area_label=f'{100*s.area:.2f}%' if s else 'lost'
                caption=f"Box {area_label} / target 2.0% | T{selector.generation} | Reacquire on loss | {frame['wall_s']:.2f}s"
                if cfg.get('explicit_image_area_constraint',False):
                    caption=f"Max box {100*row['max_reliable_vehicle_area']:.2f}% | Target {area_label} | Cap 2.0% | {frame['wall_s']:.2f}s"
            if cfg.get('target_selection')=='locked_identity_timeout_v1':
                info=selector.info;generation=info.get('target_generation',0)
                if not s:
                    missing=info.get('missing_for_s',0.)
                    if info.get('memory_state')=='occluded_search':missing+=max(0.,frame['sim_time']-info.get('reference_sim_time',frame['sim_time']))
                    caption=f"Searching T{generation} | {min(3.,missing):.1f} / 3.0s | No other target box | {frame['wall_s']:.2f}s"
                else:
                    caption=f"Locked T{generation} | Box {100*s.area:.2f}% | 2.0% cap | Forget after 3.0s missing | {frame['wall_s']:.2f}s"
            cv2.rectangle(bgr,(0,0),(w,27),(24,24,24),-1);cv2.putText(bgr,caption,(10,19),cv2.FONT_HERSHEY_SIMPLEX,.48,(235,235,235),1,cv2.LINE_AA)
            cv2.imwrite(str(out/f'frames/{index:05d}.jpg'),bgr,[cv2.IMWRITE_JPEG_QUALITY,95])
            if (index+1)%200==0:print('RGB BOXES',index+1,len(frames),flush=True)
    encode(out,frames)
    # Post-render diagnostics only. These values never alter boxes or video.
    from integration.metrics import iou
    ds=lines(out/'detections.jsonl');gs=lines(root/'offline-diagnostics.jsonl');matches=[];pixels=[]
    for d,g in zip(ds,gs):
        match=iou(d['bbox'],g['gt_bbox_normalized']) if d['bbox'] and g.get('gt_bbox_normalized') else 0.;matches.append(match)
        if match>=.3:
            b=d['bbox'];pixels.append(math.hypot(((b[0]+b[2])/2-.5)*960,((b[1]+b[3])/2-.5)*540))
    v=cv2.VideoCapture(str(out/'live-detection-vfr.mp4'));n=int(v.get(cv2.CAP_PROP_FRAME_COUNT));fps=v.get(cv2.CAP_PROP_FPS);v.release()
    summary=dict(completed=True,raw_frames=len(frames),encoded_frames=n,wall_s=frames[-1]['wall_s']-frames[0]['wall_s'],encoded_s=n/fps,
        same_frame_match_fraction=sum(m>=.3 for m in matches)/len(matches),yellow_fraction=sum(d['centered'] and m>=.3 for d,m in zip(ds,matches))/len(matches),
        matched_mean_center_px=float(np.mean(pixels)) if pixels else None,matched_p95_center_px=float(np.percentile(pixels,95)) if pixels else None,
        detector_sha256=detector.provenance['sha256'],reid_sha256=extractor.sha256,checkpoint_sha256=cfg['checkpoint_sha256'],
        scope='Same-frame RGB boxes, own camera history association, original wall timestamps. GT read only after all frames/encoding for independent match diagnostics.',video=str(out/'live-detection-vfr.mp4'))
    write(out/'provenance-and-audit.json',summary);assert abs(summary['encoded_s']-summary['wall_s'])<.2
    print(json.dumps(summary),flush=True)
if __name__=='__main__':main()
