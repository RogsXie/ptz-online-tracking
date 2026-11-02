"""Disclosed acquisition/tracking P-DQN modes on unchanged moving scene.

Both modes select directions/durations through frozen neural policies. The
stage switch selects a policy, never a geometric action. New tracking model
has been trained on actual fine dynamic RGB interactions.
"""
import speed_bootstrap
from speed_bootstrap import REV
import json,math,time,numpy as np
from pathlib import Path
import live_speed_v5 as runner
import live_uniform_motion_stop as scene
import live_fast_tiny_corner as fast
import motion_tracking15 as tracking
from motion_learning15 import LOW,HIGH,SECONDS,ObservationEnvironment as BaseObservation
from speed_agent_v5 import create_agent as coarse_agent
from speed_runner_v5 import resume as coarse_resume
import train as legacy

STAGE='acquisition'
COARSE=REV/'training-v5/checkpoint-0020.pt'

class Policy:
    def __init__(self):
        object.__setattr__(self,'tracker',tracking.create_agent());object.__setattr__(self,'coarse',coarse_agent())
        object.__setattr__(self,'acquired',False);object.__setattr__(self,'still_count',0)
    def __getattr__(self,key):return getattr(self.tracker,key)
    def __setattr__(self,key,value):
        if key in ('tracker','coarse','acquired','still_count','coarse_snapshot'):object.__setattr__(self,key,value)
        else:setattr(self.tracker,key,value)
    def choose_action(self,state,train=True):
        global STAGE
        visible=state[3]>0;error=math.hypot(float(state[0]),float(state[1]));area=float(state[2])/100.
        if not visible:self.acquired=False
        if visible and error<=.12 and .025<=area<=.055:self.acquired=True
        moving=math.hypot(float(state[8]),float(state[9]))*10>=.05
        self.still_count=0 if moving else self.still_count+1
        use_tracking=self.acquired and self.still_count<4
        STAGE='tracking' if use_tracking else 'settling' if self.acquired else 'acquisition'
        if use_tracking:return self.tracker.choose_action(state,train=True)
        act,param,params=self.coarse.choose_action(np.asarray(state[:8]),train=True)
        return act,np.clip(param,.3,5.),params

def resume(a,path):
    s=tracking.resume(a.tracker,path);coarse_resume(a.coarse,COARSE);a.coarse.epsilon=0.;a.coarse_snapshot=legacy.snapshot(a.coarse)
    return s

class Observation(BaseObservation):
    def observe(self,*args,**kwargs):
        state,row=super().observe(*args,**kwargs);row['controller_stage']=STAGE;return state,row

class TransportMixin:
    def rpc(self,command):
        if command.startswith('vset /ptz/play '):
            p,t,z,d=map(float,command.split()[2:]);fov=float(self.diagnostics['fov']);factor=4. if STAGE=='tracking' else 1.
            g=min(1.,factor*(math.tan(math.radians(fov/2))/math.tan(math.radians(33)))**2);zg=min(1.,fov/66.)
            executed=f'vset /ptz/play {p*g:.12f} {t*g:.12f} {z*zg:.12f} {d:.12f}'
            if not hasattr(self,'stage_log'):
                import sys
                folder=Path(sys.argv[sys.argv.index('--output')+1]);self.stage_log=(folder/'actuator-stages.jsonl').open('w',encoding='utf-8',buffering=1)
            self.stage_log.write(json.dumps(dict(stage=STAGE,sim_time=self.diagnostics['sim_time'],fov=fov,gain=g,requested=command,executed=executed))+'\n');command=executed
        return super().rpc(command)
    def close(self):
        try:return super().close()
        finally:
            if hasattr(self,'stage_log'):self.stage_log.close()
class Transport(TransportMixin,runner.AtomicTransport):pass
class DisplayTransport(TransportMixin,runner.DisplayTransport):pass

policy=None
def create_agent():
    global policy
    policy=Policy();return policy

def write(path,value):
    if path.name=='configuration.json':
        value.update(execution_profile=tracking.PROFILE,rates=list(fast.RATES),motion=scene.MOTION,coarse_checkpoint=str(COARSE),
            architecture='Explicittwo-stage engineeringextension: original8input staticP-DQN foracquisition/settling;15input dynamicP-DQN fortracking. Eachhasoriginal4networks. Notexactoriginalpaperprotocol.',
            stage_switch='RGBcenter<=.12 andarea.025..055 latchesacquired;RGB/camera-estimated motion<.05deg/s for4observationsselectssettling. No stage choosesmovementdirections.',
            controller='Everydirection andrawduration fromoneoftwofrozenP-DQN policies;exclusive7actions, nomanualerror-to-actionrule,GTtrajectoryorknownspeed/stop inputs.',
            qualification='Same-origin dynamicdevelopment comparison; no fullstatisticalqualification')
        if value.get('all_learning_state_frozen'):legacy.finish_evaluation(policy.coarse,policy.coarse_snapshot)
    return fast.original_write(path,value)

if __name__=='__main__':
    runner.create_agent=create_agent;runner.resume=resume;runner.SpeedEnvironment=Observation
    runner.LOW=LOW;runner.HIGH=HIGH;runner.SECONDS_PER_RAW=SECONDS
    runner.layout=scene.layout;runner.materialize=scene.materialize
    runner.AtomicTransport=Transport;runner.DisplayTransport=DisplayTransport;runner.write=write
    runner.main()
