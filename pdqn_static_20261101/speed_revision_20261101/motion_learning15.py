"""Source P-DQN with declared image-derived dynamic observation extension."""
import speed_bootstrap
from speed_bootstrap import REV
import json,math,random
from collections import deque
import numpy as np,torch,gymnasium as gym
from author_aligned.source import load_core
from author_aligned.run import NETWORKS,OPTIMIZERS,restore_rng
from speed_environment_v5 import Environment as BaseEnvironment,image_cost
from integration.transport import ACTION_RATES
from scenes import configuration,INIT,ground_position

PROFILE='dynamic15_rgb_motion_original_pdqn_v1'
LOW=(.3,.25,.25,.25,.25,.3,.3)
HIGH=(5.,10.,10.,10.,10.,5.,5.)
SECONDS=(.01,.004,.004,.8/180.,.8/180.,.01,.01)

def create_agent():
    seed=20261101;random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);cls,_,_=load_core()
    space=gym.spaces.Tuple((gym.spaces.Discrete(7),*[gym.spaces.Box(np.array([lo],np.float32),np.array([hi],np.float32)) for lo,hi in zip(LOW,HIGH)]))
    return cls(gym.spaces.Discrete(15),space,batch_size=512,gamma=.9,seed=seed,actor_lr=.0001,param_net_lr=.00001,
        actor_kwargs={'hidden_layers':[128,256,128,64]},param_net_kwargs={'hidden_layers':[128,256,128,64]},device='cuda')

def transfer(agent):
    parent=REV/'training-v5/checkpoint-0020.pt';saved=torch.load(parent,map_location=agent.device,weights_only=False)
    for name in NETWORKS:
        old=saved[name];new={k:v.clone() for k,v in old.items()};w=old['layers.0.weight'];out=w.new_zeros((w.shape[0],22 if name.startswith('actor') else 15))
        out[:,:8]=w[:,:8]
        if name.startswith('actor'):out[:,15:]=w[:,8:]
        new['layers.0.weight']=out;getattr(agent,name).load_state_dict(new)
    assert len(agent.memory)==0 and agent.update_count==0
    return dict(parent=str(parent),initialization='Old8state columns and7parametercolumns preserved;7new observation columns initialized zero; fresh optimizers and empty replay',profile=PROFILE)

def resume(agent,path):
    s=torch.load(path,map_location=agent.device,weights_only=False);assert s['metadata']['motion_profile']==PROFILE
    for n in (*NETWORKS,*OPTIMIZERS):getattr(agent,n).load_state_dict(s[n])
    for k,v in s['replay'].items():setattr(agent.memory,k,v)
    for k in ('update_count','actions_count','epsilon'):setattr(agent,k,s[k])
    restore_rng(agent,s['rng']);return s

class Features:
    def __init__(self):self.samples=deque();self.previous=None
    def extend(self,state,row,diag):
        now=float(diag['sim_time']);pitch,yaw,roll=map(math.radians,diag['rotation']);fov=float(diag['fov'])
        if row['visible']:
            tangent=math.tan(math.radians(fov/2));u=row['x']*tangent;v=-row['y']*tangent*540/960
            forward=np.array([math.cos(pitch)*math.cos(yaw),math.cos(pitch)*math.sin(yaw),math.sin(pitch)])
            right=np.array([-math.sin(yaw),math.cos(yaw),0.]);up=np.array([-math.sin(pitch)*math.cos(yaw),-math.sin(pitch)*math.sin(yaw),math.cos(pitch)])
            ray=forward+u*right+v*up
            bearing=np.array([math.degrees(math.atan2(ray[1],ray[0])),math.degrees(math.atan2(ray[2],math.hypot(ray[0],ray[1])))])
            self.samples.append((now,bearing))
        while self.samples and now-self.samples[0][0]>.6:self.samples.popleft()
        velocity=np.zeros(2)
        if len(self.samples)>=3 and self.samples[-1][0]-self.samples[0][0]>=.12:
            times=np.array([s[0] for s in self.samples]);angles=np.stack([s[1] for s in self.samples]);dt=times-times.mean()
            velocity=(dt[:,None]*(angles-angles.mean(0))).sum(0)/max(1e-12,(dt*dt).sum())
        dt=0.;camera_delta=np.zeros(3)
        if self.previous:
            dt=now-self.previous[0];camera_delta=np.array([math.degrees(yaw)-self.previous[1],math.degrees(pitch)-self.previous[2],fov-self.previous[3]])
        self.previous=(now,math.degrees(yaw),math.degrees(pitch),fov)
        extras=np.r_[np.clip(velocity/10.,-1.,1.),fov/66.,np.clip(dt/.1,0.,5.),np.clip(camera_delta/10.,-1.,1.)].astype(np.float32)
        row.update(image_motion_bearing_rate_deg_s=velocity.tolist(),dynamic_extras=extras.tolist(),observation_extension='RGBbbox ray plus own cameraorientation/FOV/time; never targetGT or scenevelocity')
        return np.r_[state,extras].astype(np.float32),row

class ObservationEnvironment(BaseEnvironment):
    def reset(self,*args,**kwargs):self.motion_features=Features();return super().reset(*args,**kwargs)
    def observe(self,*args,**kwargs):
        state,row=super().observe(*args,**kwargs)
        return self.motion_features.extend(state,row,self.t.diagnostics)

class TrainingEnvironment(ObservationEnvironment):
    def step(self,action,raw):
        duration=float(np.clip(raw,LOW[action],HIGH[action]))*SECONDS[action];fov=float(self.t.diagnostics['fov'])
        g=min(1.,2*(math.tan(math.radians(fov/2))/math.tan(math.radians(33)))**2);zg=min(1.,fov/66.)
        p,t,z=ACTION_RATES[action];before=float(self.t.diagnostics['sim_time'])
        self.t.rpc(f'vset /ptz/step {p*g:.12f} {t*g:.12f} {z*zg:.12f} {duration:.12f}')
        idle=max(0.,.1-duration)
        if idle>=.001:self.t.rpc(f'vset /ptz/step 0 0 0 {idle:.12f}')
        self.t.capture();state,row=self.observe(action,duration);self.steps+=1
        assert abs(row['sim_time']-before-.1)<1e-5
        cost=image_cost(row);inner=bool(row['visible'] and row['center_error']<=.02 and abs(math.log(row['area']/.04))<=.25)
        reward=20*(image_cost(self.previous)-cost)-2*cost+1.5*inner-.02*(action!=0)
        done=self.steps>=80
        row.update(action=int(action),raw_parameter=float(raw),duration=duration,cycle_sim_s=.1,idle_sim_s=idle,reward=float(reward),done=done,truncated=done)
        self.rows.append(row);self.previous=row
        return state,float(reward),done,row

def training_scene(folder,index):
    rng=random.Random(39600000+index);pitch=rng.uniform(-53.,-40.);fov=24.*math.sin(math.radians(abs(pitch)))/math.sin(math.radians(60))*rng.uniform(.92,1.08)
    item=dict(split='train',index=index,seed=39600000+index,quadrant=index%4,layer=0,scale_group=2,
        planned_center=[.5+rng.uniform(-.045,.045),.5+rng.uniform(-.045,.045)],fov_deg=fov,camera_height_cm=6500.)
    cfg=configuration(item,(180.,100.,100.));init=dict(INIT,camera_location_cm=[-2200.,0.,6500.],nominal_camera_pitch_deg=pitch)
    cfg['camera'].update(location_cm=init['camera_location_cm'],rotation_deg=[pitch,0.,0.],pitch_limits_deg=[-85.,-30.])
    speed=(1 if index%2==0 else -1)*rng.uniform(120.,180.);profile=[[0.,1.],[5.,0.]] if index%3 else [[0.,1.]]
    cfg['targets'][0]['path']=dict(type='line',origin_cm=ground_position(*item['planned_center'],fov,init),velocity_cm_s=[0.,speed,0.],speed_profile=profile)
    cfg.update(name='dynamic15_training',task='image_only_motion_maintenance',features=['single_vehicle','line_motion','optional_stop']);cfg['scope_note']='Training scene layout only; schedule/speed/coordinates never supplied to observations or reward.'
    folder.mkdir(parents=True,exist_ok=True);path=folder/'scene.json';path.write_text(json.dumps(cfg,indent=2),encoding='utf-8')
    return path,item
