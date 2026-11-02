"""Native60Hz GPU readback producer; unique real source frames only."""
import speed_bootstrap
import time,json,struct,mmap,queue,threading
import cv2,numpy as np
from async_frame_runtime import Frame
from control_latency_runtime import SharedTimedRaw,Runtime,Transport

class StreamRaw(SharedTimedRaw):
    def __init__(self,*args,**kwargs):
        self.streaming=False;self.last_capture_fresh=True;self.duplicate_polls=0;self.sequence_gaps=0;self.stream_map=None;self.torn_retries=0;self.compact_commands=True
        super().__init__(*args,**kwargs)
    def start_stream(self,fps):
        start=time.perf_counter();reply=self.rpc(f'vset /ptz/async_stream {fps}');end=time.perf_counter()
        if 'error' in reply:raise RuntimeError(reply['error'])
        self.native_anchor_wall=(start+end)/2;self.native_anchor_sim=reply['sim_time'];self.anchor_uncertainty=(end-start)/2
        self.native_to_python_offset=(start+end)/2-reply['native_platform_seconds']
        self.stream_map=mmap.mmap(-1,int(reply['stream_capacity_bytes']),tagname=reply['stream_shared_name'],access=mmap.ACCESS_READ)
        self.streaming=True;self.last_sequence=-1
    def stop_stream(self):
        if self.streaming:self.rpc('vset /ptz/async_stream 0');self.streaming=False
    def capture(self):
        if not self.streaming:return super().capture()
        start=time.perf_counter()
        self.last_capture_fresh=False
        version,sequence,size,slot,capacity,tag=struct.unpack('<QQIIII',self.stream_map[:32])
        if version==0 or version%2:return None
        if sequence==self.last_sequence:self.duplicate_polls+=1;return None
        assert version==sequence*2 and sequence>self.last_sequence and tag==0x325A5450
        slots=struct.unpack_from('<I',self.stream_map,32)[0]
        assert slots==4 and slot<slots and 16<size<=capacity==4*1024*1024
        self.native_latest_sequence=sequence
        wanted=max(1,sequence-slots+1,self.last_sequence+1) if self.last_sequence>=0 else max(1,sequence-slots+1)
        location=64+(wanted%slots)*capacity
        magic,length,width,height=struct.unpack('<4I',self.stream_map[location:location+16]);size=16+length+width*height*4
        assert magic==0x315A5450 and (width,height)==(960,540) and 16<size<=capacity
        packet=self.stream_map[location:location+size]
        if struct.unpack_from('<Q',self.stream_map,0)[0]!=version:self.torn_retries+=1;return None
        if self.last_sequence>=0:self.sequence_gaps+=wanted-self.last_sequence-1
        self.last_sequence=wanted
        magic,length,width,height=struct.unpack_from('<4I',packet);offset=16+length
        assert magic==0x315A5450 and (width,height)==(960,540) and len(packet)==offset+width*height*4
        diag=json.loads(packet[16:offset]);assert diag['async_source_sequence']==self.last_sequence
        self.rgb=cv2.cvtColor(np.frombuffer(packet,np.uint8,offset=offset).reshape(height,width,4),cv2.COLOR_BGRA2RGB);end=time.perf_counter()
        native_gpu_age=diag['capture_readback_seconds']
        # Preserve original GPU latency separately; legacy observation predicts
        # over full image age, including waiting for the consumer's poll.
        source_perf=diag['async_capture_platform_seconds']+self.native_to_python_offset
        source_age=max(0.,end-source_perf)
        diag.update(async_gpu_readback_latency_s=native_gpu_age,capture_readback_seconds=source_age,
            async_delivery_age_at_reply_s=source_age,capture_poll_request_wall_s=start-self.clock,
            capture_request_wall_s=start-self.clock,capture_received_wall_s=end-self.clock,capture_total_wall_s=end-start,
            capture_clock_anchor_uncertainty_s=self.anchor_uncertainty,image_transport='nativeasyncGPU_originalformat_localsharedmemory')
        self.source_wall_s=source_perf-self.clock
        diag['source_capture_wall_s']=self.source_wall_s;self.diagnostics=diag;self.last_capture_fresh=True
        return self.rgb
    def close(self):
        self.stop_stream()
        if self.stream_map is not None:self.stream_map.close();self.stream_map=None
        super().close()

class StreamTransport(Transport):
    def __init__(self,port):
        self.raw=StreamRaw(port,hud=False);self.runtime=None;self.diagnostics=self.raw.diagnostics;self.rgb=None;self.frame=None

class StreamRuntime(Runtime):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.command_thread=threading.Thread(target=self._commands,name='exclusive-command-socket',daemon=True)
    def start(self,seconds):
        super().start(seconds);self.command_thread.start()
    def _commands(self):
        while not self.stopping:
            try:task=self.commands.get(timeout=.02)
            except queue.Empty:continue
            try:task['result']=self.raw.rpc(task['command'])
            except BaseException as e:task['error']=e
            finally:task['event'].set()
    def _io(self):
        due=self.started;sequence=0
        try:
            while not self.stopping:
                now=time.perf_counter()
                if now-self.started<self.seconds and now>=due:
                    batch=[]
                    for _ in range(4):
                        self.raw.capture();received=time.perf_counter()
                        if not self.raw.last_capture_fresh:break
                        diag=self.raw.diagnostics
                        assert diag['realtime_control'] and not diag['demo_geometry_follow'] and not diag['free_preview']
                        f=Frame(sequence,self.raw.source_wall_s,received,self.raw.rgb,diag);self.images.put_nowait(f)
                        batch.append(f);sequence+=1
                        if self.raw.last_sequence>=self.raw.native_latest_sequence:break
                    due+=self.period
                    if received-due>self.period:due=received+self.period
                    if batch:
                        # Archive every buffered source frame, but publish only
                        # the newest to control. Never replay queued old images.
                        with self.condition:self.latest=batch[-1];self.condition.notify_all()
                    continue
                wait=min(.01,max(0.,due-now)) if now-self.started<self.seconds else .01
                with self.condition:
                    if not self.stopping:self.condition.wait(wait)
        except BaseException as e:
            self.errors.append(e)
            with self.condition:self.condition.notify_all()
    def stop(self):
        self.stopping=True
        with self.condition:self.condition.notify_all()
        self.io_thread.join(35);self.command_thread.join(35)
        if self.io_thread.is_alive() or self.command_thread.is_alive():raise TimeoutError('Capture/command thread did not stop')
        self.images.put(None,timeout=5);self.writer_thread.join(10)
        if self.writer_thread.is_alive():raise TimeoutError('Writer did not drain')
        self.check()
