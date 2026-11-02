"""Instrument the real socket dispatch and source-frame acquisition times."""
import speed_bootstrap
import time,json,struct,mmap
import cv2,numpy as np
from author_aligned.environment import AtomicTransport
from async_frame_runtime import FrameRuntime,SnapshotTransport

class TimedRaw(AtomicTransport):
    def __init__(self,*args,**kwargs):
        self.clock=None;self.play_events=[]
        super().__init__(*args,**kwargs)
    def rpc(self,command):
        wire=command.replace('vset /ptz/play ','vset /ptz/play_fast ',1) if getattr(self,'compact_commands',False) and command.startswith('vset /ptz/play ') else command
        start=time.perf_counter();result=super().rpc(wire);end=time.perf_counter()
        if self.clock is not None and command.startswith('vset /ptz/play '):
            p,t,z,d=map(float,command.split()[2:])
            self.play_events.append(dict(io_start_wall_s=start-self.clock,io_ack_wall_s=end-self.clock,
                native_sim_s=result['sim_time'],native_remaining_s=result['action_remaining_seconds'],
                executed_rates=[p,t,z],duration_s=d,own_rotation=result['rotation'],own_fov=result['fov']))
        return result
    def capture(self):
        start=time.perf_counter();rgb=super().capture();end=time.perf_counter()
        if self.clock is not None:
            self.diagnostics=dict(self.diagnostics,capture_request_wall_s=start-self.clock,capture_received_wall_s=end-self.clock,capture_total_wall_s=end-start)
        return rgb

class Runtime(FrameRuntime):
    def start(self,seconds):
        self.seconds=seconds;self.started=time.perf_counter();self.raw.clock=self.started
        self.writer_thread.start();self.io_thread.start()
    def wait_completed(self,event):
        # A valid completed image may already be published when the command
        # acknowledgement wakes the controller. Never discard it needlessly.
        deadline=time.perf_counter()+5
        with self.condition:
            while True:
                self.check();f=self.latest
                if f and f.diagnostics['sim_time']>=event['native_sim_s']+event['duration_s']-1e-5 and f.diagnostics['action_remaining_seconds']<=1e-6:return f
                remaining=deadline-time.perf_counter()
                if remaining<=0:raise TimeoutError('No completed-action image')
                self.condition.wait(min(.1,remaining))

class Transport(SnapshotTransport):
    def __init__(self,port):
        self.raw=TimedRaw(port,hud=False);self.runtime=None;self.diagnostics=self.raw.diagnostics;self.rgb=None;self.frame=None

class SharedTimedRaw(TimedRaw):
    def __init__(self,*args,**kwargs):
        self.mapping=None;self.mapping_name=None;self.last_sequence=0
        super().__init__(*args,**kwargs)
    def capture(self):
        start=time.perf_counter();reply=self.rpc('vget /ptz/frame_shm')
        if 'error' in reply:raise RuntimeError(reply['error'])
        if self.mapping is None:
            self.mapping_name=reply['shared_name'];self.mapping=mmap.mmap(-1,int(reply['capacity_bytes']),tagname=self.mapping_name,access=mmap.ACCESS_READ)
        assert reply['shared_name']==self.mapping_name and reply['sequence']>self.last_sequence
        self.last_sequence=reply['sequence'];self.mapping.seek(0);packet=self.mapping.read(int(reply['packet_bytes']))
        magic,length,width,height=struct.unpack_from('<4I',packet);offset=16+length
        assert magic==0x315A5450 and (width,height)==(960,540) and len(packet)==offset+width*height*4
        self.diagnostics=json.loads(packet[16:offset]);bgra=np.frombuffer(packet,np.uint8,offset=offset).reshape(height,width,4)
        self.rgb=cv2.cvtColor(bgra,cv2.COLOR_BGRA2RGB);end=time.perf_counter()
        self.diagnostics.update(shared_packet_sequence=self.last_sequence,image_transport='same-statePTZ1viaLocalSharedMemory')
        if self.clock is not None:self.diagnostics.update(capture_request_wall_s=start-self.clock,capture_received_wall_s=end-self.clock,capture_total_wall_s=end-start)
        return self.rgb
    def close(self):
        if self.mapping is not None:self.mapping.close();self.mapping=None
        super().close()

class SharedTransport(SnapshotTransport):
    def __init__(self,port):
        self.raw=SharedTimedRaw(port,hud=False);self.runtime=None;self.diagnostics=self.raw.diagnostics;self.rgb=None;self.frame=None
