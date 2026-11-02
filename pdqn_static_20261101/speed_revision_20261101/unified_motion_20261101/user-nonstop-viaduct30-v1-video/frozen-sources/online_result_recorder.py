"""Paint actual completed online outputs on their exact input RGB frames.

No detector, association, optical-flow reconstruction, camera projection or GT
access is performed here. After recording, export only encodes saved pixels at
their actual output-availability timestamps.
"""
import hashlib,json,math,queue,threading,time
from pathlib import Path
import cv2,numpy as np

def export_saved_pixels(root,frames,clock,live_stopped_perf=None):
    """Compression/mux only. Accept variable last-frame display duration, and
    validate each decoded presentation timestamp instead of assuming60Hz."""
    root=Path(root)
    from encode_60fps_vfr import encode as encode_pixels
    intervals=np.diff([x['wall_s'] for x in frames])
    tolerance=max(.2,float(np.max(intervals))*1.1) if len(intervals) else .2
    audit=encode_pixels(root,frames,name='true-online-output.mp4',duration_tolerance_s=tolerance,bframes=0)
    capture=cv2.VideoCapture(str(root/'true-online-output.mp4'));pts=[]
    while True:
        ok,_=capture.read()
        if not ok:break
        pts.append(capture.get(cv2.CAP_PROP_POS_MSEC)/1000.)
    capture.release()
    assert len(pts)==len(frames)
    timestamp_error=max(abs(t-r['wall_s']) for t,r in zip(pts,frames))
    assert timestamp_error<.003,timestamp_error
    observations=frames[1:]
    summary=dict(completed=True,video=str((root/'true-online-output.mp4').resolve()),
        painted_online=True,offline_detection_or_reassociation=False,
        all_frames_are_exact_processed_rgb_inputs=True,one_frame_per_actual_output=True,
        image_frames=len(frames),tracking_updates=len(observations),
        tracking_output_hz=len(observations)/max(1e-9,observations[-1]['wall_s']) if observations else 0.,
        tracking_output_hz_after_first_result=(len(observations)-1)/(observations[-1]['wall_s']-observations[0]['wall_s']) if len(observations)>1 else 0.,
        median_source_to_draw_latency_ms=float(np.median([x['latency_at_draw_ms'] for x in observations])) if observations else None,
        median_draw_and_write_ms=float(np.median([x['draw_and_write_ms'] for x in frames])),
        last_pixels_written_wall_s=frames[-1]['wall_s'],all_noninitial_pixels_written_before_30s=all(x['wall_s']<30. for x in observations),
        initial_pixels_ready_before_runtime_start=frames[0]['pixels_written_perf']<clock,
        all_pixels_written_before_live_stop=all(x['pixels_written_perf']<=live_stopped_perf for x in frames) if live_stopped_perf is not None else None,
        encoding_only_after_run=True,no_synthetic_interpolated_or_repeated_result_frames=True,
        maximum_presentation_timestamp_error_s=timestamp_error,
        timing='Actual online output publication intervals. Encoder onlycompresses alreadyannotated images; no inference, boxcorrection, GT or trajectory replay.',**audit)
    (root/'online-recording-audit.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print('TRUE ONLINE RECORDING',json.dumps(summary),flush=True)
    return summary

class Recorder:
    def __init__(self,root):
        self.root=Path(root)/'online-output';self.root.mkdir(exist_ok=False)
        (self.root/'frames').mkdir()
        self.queue=queue.Queue(maxsize=8);self.frames=[];self.errors=[];self.clock=None
        self.thread=threading.Thread(target=self._write,name='actual-online-output-drawing',daemon=True)
        self.thread.start();self.closed=False;self.published_count=0;self.live_stopped_perf=None

    def publish(self,rgb,row,frame=None,runtime=None):
        if self.errors:raise RuntimeError('Online output recording failed') from self.errors[0]
        completed=time.perf_counter()
        if runtime is not None:self.clock=runtime.started
        association=row.get('rgb_association',{})
        result=dict(visible=bool(row['visible']),bbox=list(row['bbox']) if row['bbox'] else None,
            confidence=float(row['confidence']),center_error=row.get('center_error'),
            sim_time=float(row['sim_time']),target_label=row.get('target_label'),
            target_class_id=row.get('target_class_id'),target_generation=association.get('target_generation',0),
            memory_state=association.get('memory_state','tracking'),missing_for_s=association.get('missing_for_s',0.),
            observation_source=association.get('observation_source','detector'),
            locked_track_id=association.get('locked_track_id'))
        item=dict(rgb=rgb.copy(),result=result,completed_perf=completed,
            source_frame=frame.sequence if frame is not None else None,
            source_wall_s=frame.wall_s if frame is not None else None,
            runtime_start=runtime.started if runtime is not None else None)
        self.queue.put_nowait(item)
        self.published_count+=1

    def flush(self,timeout=5.):
        deadline=time.perf_counter()+timeout
        while len(self.frames)<self.published_count:
            if self.errors:raise RuntimeError('Online painting failed') from self.errors[0]
            if time.perf_counter()>deadline:raise TimeoutError('Online painter did not drain')
            time.sleep(.001)

    def mark_live_stopped(self):self.live_stopped_perf=time.perf_counter()

    def _write(self):
        recent=[]
        try:
            with (self.root/'online-results.jsonl').open('w',encoding='utf-8',buffering=1) as log:
                while True:
                    item=self.queue.get()
                    if item is None:break
                    index=len(self.frames);r=item['result'];rgb=item['rgb']
                    draw_start=time.perf_counter();clock=item['runtime_start']
                    wall=max(0.,draw_start-clock) if clock is not None else 0.
                    recent=[t for t in recent if draw_start-t<1.5]
                    recent.append(draw_start)
                    hz=(len(recent)-1)/(recent[-1]-recent[0]) if len(recent)>1 else 0.
                    age_ms=max(0.,1000*(draw_start-clock-item['source_wall_s'])) if clock is not None else None
                    source_hash=hashlib.sha256(rgb.tobytes()).hexdigest()
                    bgr=cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR);h,w=bgr.shape[:2];cx,cy=w//2,h//2
                    cv2.drawMarker(bgr,(cx,cy),(255,180,40),cv2.MARKER_CROSS,16,1)
                    distance=None
                    if r['visible'] and r['bbox']:
                        b=r['bbox'];x1,y1,x2,y2=[int(round(v*z)) for v,z in zip(b,(w,h,w,h))]
                        normalized_error=math.hypot(b[0]+b[2]-1.,b[1]+b[3]-1.)
                        centered=normalized_error<=.025;colour=(0,235,255) if centered else (60,235,60)
                        cv2.rectangle(bgr,(x1,y1),(x2,y2),colour,2,cv2.LINE_AA)
                        tx,ty=(b[0]+b[2])*w/2,(b[1]+b[3])*h/2
                        distance=math.hypot(tx-cx,ty-cy)
                        target=(int(round(tx)),int(round(ty)));yellow=(0,235,255)
                        cv2.line(bgr,(cx,cy),target,yellow,2,cv2.LINE_AA);cv2.circle(bgr,target,3,yellow,-1,cv2.LINE_AA)
                        label=f'{distance:.1f} px'
                        location=(min(w-100,max(8,(cx+target[0])//2+10)),min(h-8,max(72,(cy+target[1])//2-10)))
                        cv2.putText(bgr,label,location,cv2.FONT_HERSHEY_SIMPLEX,.52,(0,0,0),3,cv2.LINE_AA)
                        cv2.putText(bgr,label,location,cv2.FONT_HERSHEY_SIMPLEX,.52,yellow,1,cv2.LINE_AA)
                        text=f'{r["target_label"] or "target"} {r["confidence"]:.2f}'
                        if r['observation_source']=='verified_optical_flow':text=f'{r["target_label"] or "target"} TRACKED'
                        if centered:text+=' CENTERED'
                        cv2.putText(bgr,text,(x1,max(72,y1-8)),cv2.FONT_HERSHEY_SIMPLEX,.48,(0,0,0),3,cv2.LINE_AA)
                        cv2.putText(bgr,text,(x1,max(72,y1-8)),cv2.FONT_HERSHEY_SIMPLEX,.48,colour,1,cv2.LINE_AA)
                    state=f'LOCKED T{r["target_generation"]}' if r['visible'] else f'SEARCH T{r["target_generation"]} {r["missing_for_s"]:.1f}/3.0s - NO BOX'
                    age=f'{age_ms:.0f}ms' if age_ms is not None else 'INIT'
                    frame_label=str(item['source_frame']) if item['source_frame'] is not None else 'INIT'
                    top=f'ONLINE OUTPUT | {state} | output {hz:.1f} Hz | latency {age}'
                    bottom=f'Exact processed RGB frame {frame_label} | direct tracker output | {wall:.2f}s'
                    cv2.rectangle(bgr,(0,0),(w,53),(24,24,24),-1)
                    cv2.putText(bgr,top,(10,20),cv2.FONT_HERSHEY_SIMPLEX,.47,(235,235,235),1,cv2.LINE_AA)
                    cv2.putText(bgr,bottom,(10,43),cv2.FONT_HERSHEY_SIMPLEX,.45,(210,210,210),1,cv2.LINE_AA)
                    path=self.root/f'frames/{index:05d}.jpg'
                    if not cv2.imwrite(str(path),bgr,[cv2.IMWRITE_JPEG_QUALITY,95]):raise RuntimeError('Online frame write failed')
                    published=time.perf_counter();output_wall=max(0.,published-clock) if clock is not None else 0.
                    result_hash=hashlib.sha256(json.dumps(r,sort_keys=True).encode()).hexdigest()
                    entry=dict(frame=index,wall_s=output_wall,sim_time=r['sim_time'],source_frame=item['source_frame'],
                        source_capture_wall_s=item['source_wall_s'],result_completed_perf=item['completed_perf'],
                        drawn_started_perf=draw_start,pixels_written_perf=published,
                        raw_rgb_sha256=source_hash,rendered_jpeg_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                        result_sha256=result_hash,latency_at_draw_ms=age_ms,draw_and_write_ms=1000*(published-draw_start),
                        distance_px=distance,result=r)
                    self.frames.append(entry);log.write(json.dumps(entry)+'\n')
        except BaseException as error:self.errors.append(error)

    def close(self,encode=True):
        if self.closed:return
        self.closed=True;self.queue.put(None,timeout=5);self.thread.join(15)
        if self.thread.is_alive():raise TimeoutError('Online painter failed to stop')
        if self.errors:raise RuntimeError('Online painting failed') from self.errors[0]
        if not encode or not self.frames:return
        export_saved_pixels(self.root,self.frames,self.clock,self.live_stopped_perf)
