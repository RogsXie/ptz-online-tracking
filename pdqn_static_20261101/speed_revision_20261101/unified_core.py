"""One 15-input P-DQN for acquisition, motion and stopped targets.

The pinned author's update remains unmodified. This module declares the
observation, physical action constraint and actuator extensions separately.
"""
import speed_bootstrap
from speed_bootstrap import REV
import math, types
from pathlib import Path
import numpy as np, torch
import motion_learning15 as base
from author_aligned.run import NETWORKS, OPTIMIZERS, restore_rng
from integration.transport import ACTION_RATES
from speed_environment_v5 import image_cost
from unified_perception import FoveatedDetector, Selector

PROFILE='unified15_mixed_motion_v1'
ROOT=REV/'unified_motion_20261101'
LOW,HIGH,SECONDS=base.LOW,base.HIGH,base.SECONDS

def frontend():
    from control_model_v13.detector import create_detector
    from integration.experiment import make_extractor
    d=create_detector(checkpoint=ROOT/'detector_longroad_v2.pt',device='cuda',threshold=.15,imgsz=960)
    return FoveatedDetector(d),make_extractor('cuda')

def create_agent():
    a=base.create_agent()
    for g in a.actor_optimiser.param_groups:g['lr']=.0003
    for g in a.param_net_optimiser.param_groups:g['lr']=.00003
    original=a.choose_action
    def choose(self,state,train=True):
        act,param,params=original(state,train=train)
        if not hasattr(self,'camera_diagnostics'):return act,param,params
        valid=valid_actions(self.camera_diagnostics,self.camera_limits)
        if valid[int(act)]:return act,param,params
        with torch.inference_mode():
            q=self.actor_net(torch.as_tensor(state,dtype=torch.float32,device=self.device).unsqueeze(0),
                torch.as_tensor(params,dtype=torch.float32,device=self.device).unsqueeze(0))[0].cpu().numpy()
        q[~valid]=-np.inf;act=int(q.argmax())
        return act,np.array([np.clip(params[act],LOW[act],HIGH[act])],np.float32),params
    a.choose_action=types.MethodType(choose,a)
    return a

def valid_actions(d,limits):
    valid=np.ones(7,bool);p,y,_=d['rotation'];f=d['fov'];tol=1e-4
    if f>=limits['fov_max_deg']-tol:valid[6]=False
    if f<=limits['fov_min_deg']+tol:valid[5]=False
    if p>=limits['pitch_limits_deg'][1]-tol:valid[2]=False
    if p<=limits['pitch_limits_deg'][0]+tol:valid[1]=False
    if y>=limits['yaw_limits_deg'][1]-tol:valid[3]=False
    if y<=limits['yaw_limits_deg'][0]+tol:valid[4]=False
    return valid

def bind_camera(a,env):
    a.camera_diagnostics=env.t.diagnostics
    a.camera_limits=env.camera_limits

def gain(fov):return min(1.,4*(math.tan(math.radians(fov/2))/math.tan(math.radians(33)))**2),min(1.,fov/66.)

class Observation(base.ObservationEnvironment):
    def __init__(self,t,detector,extractor):
        super().__init__(t,detector,extractor)
        self.cfff=Selector(self.cfff,t,detector)
    def reset(self,scene,*args,**kwargs):
        import json
        self.camera_limits=json.loads(Path(scene).read_text(encoding='utf-8'))['camera']
        return super().reset(scene,*args,**kwargs)
    def observe(self,*args,**kwargs):
        state,row=super().observe(*args,**kwargs)
        row['rgb_association']=self.cfff.info.copy()
        return state,row

class Environment(Observation):
    def step(self,action,raw):
        duration=float(np.clip(raw,LOW[action],HIGH[action]))*SECONDS[action]
        g,zg=gain(float(self.t.diagnostics['fov']));p,t,z=ACTION_RATES[action]
        before=float(self.t.diagnostics['sim_time']);cycle=self.cycle
        self.t.rpc(f'vset /ptz/step {p*g:.12f} {t*g:.12f} {z*zg:.12f} {duration:.12f}')
        idle=max(0.,cycle-duration)
        if idle>=.001:self.t.rpc(f'vset /ptz/step 0 0 0 {idle:.12f}')
        self.t.capture();state,row=self.observe(action,duration);self.steps+=1
        assert abs(row['sim_time']-before-cycle)<1e-5
        if row['visible']:
            scale=abs(math.log(row['area']/.04));inner=row['center_error']<=.015 and scale<=.25
            reward=20*(image_cost(self.previous)-image_cost(row))-30*row['center_error']-2*scale+1.5*inner-.005*(action!=0)
        else:reward=-50.
        done=self.steps>=self.length
        row.update(reward=float(reward),done=done,truncated=done,raw_parameter=float(raw),duration=duration,cycle_sim_s=cycle)
        self.rows.append(row);self.previous=row
        return state,float(reward),done,row

def resume(a,path):
    s=torch.load(path,map_location=a.device,weights_only=False)
    assert s['metadata']['motion_profile']==PROFILE
    for n in (*NETWORKS,*OPTIMIZERS):getattr(a,n).load_state_dict(s[n])
    for k,v in s['replay'].items():setattr(a.memory,k,v)
    for k in ('update_count','actions_count','epsilon'):setattr(a,k,s[k])
    restore_rng(a,s['rng']);return s
