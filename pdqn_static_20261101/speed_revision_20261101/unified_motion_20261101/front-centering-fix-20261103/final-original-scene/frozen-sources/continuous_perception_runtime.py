"""Single-owner online perception, independent of motor-completion waiting."""
import copy,json,math,threading,time
from dataclasses import dataclass
from types import SimpleNamespace
import numpy as np

@dataclass(frozen=True)
class Result:
    index:int
    frame:object
    selected:object
    detections:tuple
    row:dict
    completed:float

class PerceptionView:
    def __init__(self,transport):
        self.rgb=transport.rgb;self.diagnostics=transport.diagnostics;self.frame=transport.frame
    def pin(self,frame):
        self.rgb=frame.rgb;self.diagnostics=frame.diagnostics;self.frame=frame

class CompletedCache:
    """Only a handoff of a real result; never performs or invents inference."""
    def __init__(self,generation):
        self.result=None;self.info={};self.last_detections=[];self.generation=generation
        self.on_target_change=None
    def install(self,result):
        self.result=result;self.last_detections=list(result.detections)
        self.info=copy.deepcopy(result.row['rgb_association'])
    def __call__(self,rgb):return self.last_detections
    def choose(self,rgb,detections,count_step=False):
        generation=self.info.get('target_generation',0)
        if generation!=self.generation:
            self.generation=generation
            if self.on_target_change:self.on_target_change()
        return self.result.selected

def policy_view(env,transport):
    """Keep the trained per-command state history separate from frame-rate tracking."""
    policy=copy.copy(env)
    policy.t=transport
    policy.motion_features=copy.deepcopy(env.motion_features)
    cache=CompletedCache(env.cfff.info.get('target_generation',0))
    policy.detector=cache;policy.cfff=cache
    return policy,cache

class Worker:
    def __init__(self,env,transport,runtime,recorder,root,max_fps=30.):
        self.env,self.runtime,self.recorder,self.root=env,runtime,recorder,root
        self.view=PerceptionView(transport);env.t=self.view
        env.cfff.t=self.view;env.detector.transport=self.view;env.extractor.transport=self.view
        self.period=1./max_fps;self.condition=threading.Condition();self.stopping=threading.Event()
        self.latest=None;self.errors=[];self.results=[]
        self.thread=threading.Thread(target=self._run,name='continuous-real-perception',daemon=True)
    def start(self):self.thread.start()
    def check(self):
        self.runtime.check()
        if self.errors:raise RuntimeError('Continuous perception failed') from self.errors[0]
    def _next_frame(self,after):
        with self.runtime.condition:
            while not self.stopping.is_set() and time.perf_counter()<self.runtime.started+self.runtime.seconds:
                self.runtime.check();frame=self.runtime.latest
                if frame is not None and frame.sequence>after:return frame
                self.runtime.condition.wait(.02)
        return None
    def _run(self):
        after=-1;due=time.perf_counter()
        try:
            with (self.root/'perception-online.jsonl').open('w',encoding='utf-8',buffering=1) as log:
                while not self.stopping.is_set():
                    if due>time.perf_counter():self.stopping.wait(min(.05,due-time.perf_counter()))
                    frame=self._next_frame(after)
                    if frame is None:break
                    after=frame.sequence;self.view.pin(frame);start=time.perf_counter()
                    detections=self.env.detector(frame.rgb);detector_done=time.perf_counter()
                    selected=self.env.cfff.choose(frame.rgb,detections,count_step=False);done=time.perf_counter()
                    association=copy.deepcopy(self.env.cfff.info)
                    if selected:
                        x,y=[2*v-1 for v in selected.center]
                        area=selected.area;confidence=selected.confidence;box=list(selected.bbox)
                    else:x=y=area=None;confidence=0.;box=None
                    row=dict(visible=selected is not None,x=x,y=y,area=area,confidence=confidence,bbox=box,
                        center_error=math.hypot(x,y) if selected else None,sim_time=float(frame.diagnostics['sim_time']),
                        detector_s=detector_done-start,reid_s=done-detector_done,rgb_association=association,
                        target_label=selected.label if selected else None,target_class_id=selected.class_id if selected else None,
                        whole_vehicle_box=True,source_frame=frame.sequence,perception_s=done-start)
                    result=Result(len(self.results),frame,selected,tuple(detections),row,done)
                    # Draw this exact newly processed RGB frame online, including
                    # mid-action frames; never reuse a previous result as a new one.
                    self.recorder.publish(frame.rgb,row,frame,self.runtime)
                    record=dict(result_index=result.index,source_wall_s=frame.wall_s,
                        result_completed_wall_s=done-self.runtime.started,source_action_remaining_s=frame.diagnostics['action_remaining_seconds'],**row)
                    self.results.append(record);log.write(json.dumps(record)+'\n')
                    with self.condition:self.latest=result;self.condition.notify_all()
                    due=max(due+self.period,start+self.period)
        except BaseException as error:
            self.errors.append(error)
            with self.condition:self.condition.notify_all()
    def wait_completed(self,event,after=-1):
        deadline=time.perf_counter()+5.
        with self.condition:
            while True:
                self.check();r=self.latest
                if r is not None and r.index>after and r.frame.diagnostics['sim_time']>=event['native_sim_s']+event['duration_s']-1e-5 and r.frame.diagnostics['action_remaining_seconds']<=1e-6:return r
                remaining=deadline-time.perf_counter()
                if remaining<=0:raise TimeoutError('No completed-action perception result')
                self.condition.wait(min(.05,remaining))
    def stop(self):
        self.stopping.set()
        with self.runtime.condition:self.runtime.condition.notify_all()
        self.thread.join(10)
        if self.thread.is_alive():raise TimeoutError('Perception thread did not stop')
        self.check()
