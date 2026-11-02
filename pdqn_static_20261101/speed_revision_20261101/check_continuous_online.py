"""Check real mid-action perception and post-action-only control handoff."""
import speed_bootstrap
from speed_bootstrap import REV
import tempfile,time,threading,copy
from types import SimpleNamespace
from pathlib import Path
import numpy as np
from async_frame_runtime import Frame
from continuous_perception_runtime import Worker,Result,policy_view
from hascam.detection import Detection
from motion_learning15 import ObservationEnvironment
from short_motion_features import ShortFeatures
from online_result_recorder import Recorder

def main():
    d=Detection((.4,.4,.6,.6),.9,0,'car');rgb=np.zeros((540,960,3),np.uint8)
    diag=dict(sim_time=.01,fov=40.,rotation=[-42.,0.,0.],action_remaining_seconds=.04)
    # The explicit motion governor must reach the actual inherited transport,
    # without changing the command duration or exclusive action components.
    from async_frame_runtime import SnapshotTransport
    raw=SimpleNamespace(rpc=lambda command:command)
    actual=SnapshotTransport.__new__(SnapshotTransport)
    actual.raw=raw;actual.runtime=None;actual.diagnostics={'fov':8.}
    actual.pan_tilt_gain_override=.18;actual.zoom_gain_override=None
    wire=actual.rpc('vset /ptz/play 1 0 0 .01').split()
    assert abs(float(wire[2])-.18)<1e-12 and list(map(float,wire[3:]))==[0.,0.,.01]
    frame=Frame(0,.01,time.perf_counter(),rgb,diag)
    class Detector:
        def __call__(self,rgb):return [d]
    class Selector:
        info=dict(target_generation=1,memory_state='tracking',selected_label='car',selected_class_id=0)
        def choose(self,*args,**kwargs):return d
    class Sink:
        def __init__(self):self.calls=[]
        def publish(self,*args):self.calls.append(args)
    with tempfile.TemporaryDirectory(prefix='continuous-unit-',dir=REV) as directory:
        root=Path(directory).resolve();assert root.is_relative_to(REV.resolve())
        runtime=SimpleNamespace(started=time.perf_counter(),seconds=1.,condition=threading.Condition(),latest=frame,check=lambda:None)
        transport=SimpleNamespace(rgb=rgb,diagnostics=diag,frame=None)
        env=SimpleNamespace(cfff=Selector(),detector=Detector(),extractor=SimpleNamespace())
        sink=Sink();worker=Worker(env,transport,runtime,sink,root,max_fps=60);worker.start()
        deadline=time.perf_counter()+1
        while worker.latest is None and time.perf_counter()<deadline:time.sleep(.002)
        assert worker.latest is not None and worker.results[0]['source_action_remaining_s']>0 and len(sink.calls)==1
        # Inference can run midpulse, but that result cannot release control.
        outcome=[];event=dict(native_sim_s=.02,duration_s=.03)
        waiter=threading.Thread(target=lambda:outcome.append(worker.wait_completed(event)));waiter.start();time.sleep(.03);assert not outcome
        f2=Frame(1,.06,time.perf_counter(),rgb,dict(diag,sim_time=.06,action_remaining_seconds=0.))
        with runtime.condition:runtime.latest=f2;runtime.condition.notify_all()
        waiter.join(2);assert outcome and outcome[0].frame.sequence==1
        worker.stop()
        # Per-command state keeps the previous command fields even when the
        # perceptual worker has processed more intervening source frames.
        t=SimpleNamespace(rgb=rgb,diagnostics=diag)
        e=ObservationEnvironment(t,lambda x:[],lambda *args:np.array([1.,0.]));e.motion_features=ShortFeatures(.25)
        e.forward_x=e.forward_y=e.forward_action=e.forward_time=0.;e.cfff=SimpleNamespace(info={'target_generation':1})
        policy,cache=policy_view(e,t)
        result=Result(0,frame,d,(d,),dict(rgb_association={'target_generation':1}),time.perf_counter())
        cache.install(result);s,_=policy.observe(3,.01);assert s[4]==0
        for index in [1,2,3]:cache.install(Result(index,frame,d,(d,),dict(rgb_association={'target_generation':1}),time.perf_counter()))
        t.diagnostics=dict(diag,sim_time=.11);s,_=policy.observe(4,.01)
        assert abs(s[4]-.5)<1e-6 and abs(s[5]-.6)<1e-6 and e.forward_action==0 and policy.motion_features is not e.motion_features
        recorder=Recorder(root);row=dict(visible=False,bbox=None,confidence=0.,center_error=None,sim_time=.1,rgb_association={'target_generation':1,'memory_state':'occluded_search'})
        recorder.publish(rgb,row,frame,runtime);recorder.flush();recorder.mark_live_stopped();recorder.close(encode=False)
        assert len(recorder.frames)==1 and recorder.frames[0]['pixels_written_perf']<=recorder.live_stopped_perf
    print('PASS: fresh midpulse inference; post-completion-onlycontrol; separate per-commandstate; onlinepaint flushbefore livestop')

if __name__=='__main__':main()
