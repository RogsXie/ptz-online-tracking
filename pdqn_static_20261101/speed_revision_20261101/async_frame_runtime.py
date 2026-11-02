"""One UnrealCV socket owner, independently paced capture and queued commands.

Images are immutable after publication. The policy receives a pinned snapshot,
never the recorder's subsequently changing camera state. No extra UE client,
frame interpolation, duplicated capture or target-pose control is involved.
"""
import speed_bootstrap
import json,time,threading,queue
from dataclasses import dataclass
import cv2
from author_aligned.environment import AtomicTransport
from unified_core import gain

@dataclass(frozen=True)
class Frame:
    sequence:int
    wall_s:float
    received:float
    rgb:object
    diagnostics:dict

class FrameRuntime:
    def __init__(self,raw,root,fps=30.):
        self.raw,self.root,self.period=raw,root,1./fps
        self.condition=threading.Condition();self.commands=queue.Queue();self.images=queue.Queue(maxsize=64)
        self.latest=None;self.frames=[];self.errors=[];self.stopping=False;self.missed_deadlines=0
        self.io_thread=threading.Thread(target=self._io,name='single-socket-capture',daemon=True)
        self.writer_thread=threading.Thread(target=self._writer,name='jpeg-and-metadata-writer',daemon=True)
    def start(self,seconds):
        self.seconds=seconds;self.started=time.perf_counter();self.writer_thread.start();self.io_thread.start()
    def check(self):
        if self.errors:raise RuntimeError('Asynchronous capture failed') from self.errors[0]
    def request(self,command):
        self.check();task=dict(command=command,event=threading.Event());self.commands.put(task)
        with self.condition:self.condition.notify_all()
        if not task['event'].wait(35):raise TimeoutError('UE command queue did not complete')
        self.check()
        if 'error' in task:raise task['error']
        return task['result']
    def wait_frame(self,after=-1,command_done=False):
        deadline=time.perf_counter()+5.
        with self.condition:
            while True:
                self.check();f=self.latest
                if f and f.sequence>after and (not command_done or f.diagnostics['action_remaining_seconds']<=1e-6):return f
                remaining=deadline-time.perf_counter()
                if remaining<=0:raise TimeoutError('No fresh sensor frame')
                self.condition.wait(min(.1,remaining))
    def _io(self):
        due=self.started;sequence=0
        try:
            while not self.stopping:
                while True:
                    try:task=self.commands.get_nowait()
                    except queue.Empty:break
                    try:task['result']=self.raw.rpc(task['command'])
                    except BaseException as e:task['error']=e
                    finally:task['event'].set()
                now=time.perf_counter()
                if now-self.started<self.seconds and now>=due:
                    self.raw.capture();received=time.perf_counter();diag=self.raw.diagnostics
                    assert diag['realtime_control'] and not diag['demo_geometry_follow'] and not diag['free_preview']
                    f=Frame(sequence,received-self.started,received,self.raw.rgb,diag)
                    self.images.put_nowait(f)
                    with self.condition:self.latest=f;self.condition.notify_all()
                    sequence+=1;due+=self.period
                    # Never burst-capture to fabricate missed deadlines.
                    if received-due>self.period:
                        skipped=int((received-due)//self.period);due+=skipped*self.period;self.missed_deadlines+=skipped
                    continue
                wait=min(.02,max(0.,due-now)) if now-self.started<self.seconds else .02
                with self.condition:
                    if self.commands.empty() and not self.stopping:self.condition.wait(wait)
        except BaseException as e:
            self.errors.append(e)
            with self.condition:self.condition.notify_all()
    def _writer(self):
        try:
            with (self.root/'capture-timestamps.jsonl').open('w',encoding='utf-8',buffering=1) as log,(self.root/'offline-diagnostics.jsonl').open('w',encoding='utf-8',buffering=1) as gt:
                while True:
                    f=self.images.get()
                    if f is None:break
                    d=f.diagnostics
                    if not cv2.imwrite(str(self.root/f'frames/{f.sequence:05d}.jpg'),cv2.cvtColor(f.rgb,cv2.COLOR_RGB2BGR),[cv2.IMWRITE_JPEG_QUALITY,95]):raise RuntimeError('Could not save captured frame')
                    row=dict(frame=f.sequence,wall_s=f.wall_s,sim_time=d['sim_time'],rotation=d['rotation'],fov=d['fov'],remaining=d['action_remaining_seconds'])
                    self.frames.append(row);log.write(json.dumps(row)+'\n');gt.write(json.dumps(d)+'\n')
        except BaseException as e:self.errors.append(e)
    def stop(self):
        self.stopping=True
        with self.condition:self.condition.notify_all()
        self.io_thread.join(35)
        if self.io_thread.is_alive():raise TimeoutError('Sensor thread failed to stop')
        self.images.put(None,timeout=5);self.writer_thread.join(10)
        if self.writer_thread.is_alive():raise TimeoutError('Writer failed to drain')
        self.check()

class SnapshotTransport:
    """Policy's stable image + own-camera snapshot; IO worker owns real socket."""
    def __init__(self,port):
        self.raw=AtomicTransport(port,hud=False);self.runtime=None
        self.diagnostics=self.raw.diagnostics;self.rgb=None;self.frame=None
    def rpc(self,command):
        if command.startswith('vset /ptz/play '):
            p,t,z,d=map(float,command.split()[2:]);g,zg=gain(float(self.diagnostics['fov']))
            if getattr(self,'pan_tilt_gain_override',None) is not None:g=float(self.pan_tilt_gain_override)
            if getattr(self,'zoom_gain_override',None) is not None:zg=float(self.zoom_gain_override)
            command=f'vset /ptz/play {p*g:.12f} {t*g:.12f} {z*zg:.12f} {d:.12f}'
        return self.runtime.request(command) if self.runtime else self.raw.rpc(command)
    def scenario(self,path):self.diagnostics=self.rpc(f'vset /ptz/scenario {path.as_posix()}')
    def reset(self,seed):
        self.diagnostics=self.rpc(f'vset /ptz/reset {seed}')
        for _ in range(4):self.capture()
        return self.rgb
    def capture(self):
        if self.runtime:self.pin(self.runtime.wait_frame(self.frame.sequence if self.frame else -1))
        else:self.raw.capture();self.diagnostics=self.raw.diagnostics;self.rgb=self.raw.rgb
        return self.rgb
    def pin(self,frame):
        self.frame=frame;self.rgb=frame.rgb;self.diagnostics=frame.diagnostics
    def close(self):self.raw.close()
