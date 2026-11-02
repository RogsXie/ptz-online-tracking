"""Fine dynamic tracking curriculum; pinned original P-DQN update math."""
import speed_bootstrap
import math,json,random,numpy as np,torch
import motion_learning15 as base
from integration.transport import ACTION_RATES
from speed_environment_v5 import image_cost

PROFILE='dynamic15_fine_tracking_gain4_v3'

def create_agent():
    a=base.create_agent()
    for group in a.actor_optimiser.param_groups:group['lr']=.001
    for group in a.param_net_optimiser.param_groups:group['lr']=.0001
    return a

class Environment(base.ObservationEnvironment):
    def step(self,action,raw):
        duration=float(np.clip(raw,base.LOW[action],base.HIGH[action]))*base.SECONDS[action];fov=float(self.t.diagnostics['fov'])
        g=min(1.,4*(math.tan(math.radians(fov/2))/math.tan(math.radians(33)))**2);zg=min(1.,fov/66.)
        p,t,z=ACTION_RATES[action];before=float(self.t.diagnostics['sim_time'])
        self.t.rpc(f'vset /ptz/step {p*g:.12f} {t*g:.12f} {z*zg:.12f} {duration:.12f}')
        idle=max(0.,.1-duration)
        if idle>=.001:self.t.rpc(f'vset /ptz/step 0 0 0 {idle:.12f}')
        self.t.capture();state,row=self.observe(action,duration);self.steps+=1;assert abs(row['sim_time']-before-.1)<1e-5
        if row['visible']:
            scale=abs(math.log(row['area']/.04));inner=row['center_error']<=.025 and scale<=.25
            reward=5*(image_cost(self.previous)-image_cost(row))-30*row['center_error']-2*scale+float(inner)
        else:reward=-50.
        done=self.steps>=80;row.update(reward=float(reward),done=done,truncated=done,raw_parameter=float(raw),duration=duration,cycle_sim_s=.1)
        self.rows.append(row);self.previous=row;return state,float(reward),done,row

def scene(folder,index):
    path,item=base.training_scene(folder,index);cfg=json.loads(path.read_text());rng=random.Random(39800000+index)
    item['seed']=39800000+index;item['planned_center']=[.5+rng.uniform(-.009,.009),.5+rng.uniform(-.009,.009)]
    from scenes import INIT,ground_position
    pitch=cfg['camera']['rotation_deg'][0];init=dict(INIT,camera_location_cm=cfg['camera']['location_cm'],nominal_camera_pitch_deg=pitch)
    cfg['targets'][0]['path']['origin_cm']=ground_position(*item['planned_center'],item['fov_deg'],init)
    cfg['experiment_coverage']=item
    path.write_text(json.dumps(cfg,indent=2));return path,item

def resume(agent,path):
    from author_aligned.run import NETWORKS,OPTIMIZERS,restore_rng
    s=torch.load(path,map_location=agent.device,weights_only=False);assert s['metadata']['motion_profile']==PROFILE
    for n in (*NETWORKS,*OPTIMIZERS):getattr(agent,n).load_state_dict(s[n])
    for k,v in s['replay'].items():setattr(agent.memory,k,v)
    for k in ('update_count','actions_count','epsilon'):setattr(agent,k,s[k])
    restore_rng(agent,s['rng']);return s
