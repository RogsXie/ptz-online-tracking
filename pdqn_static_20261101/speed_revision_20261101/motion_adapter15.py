"""RL-trained motion observation adapter; old static network tensors frozen."""
import speed_bootstrap
import math,numpy as np,torch
import motion_learning15 as base
from author_aligned.run import NETWORKS,OPTIMIZERS,restore_rng

PROFILE='dynamic15_frozen_static_motion_adapter_v2'

def transform(state):
    out=np.array(state,np.float32,copy=True);out[8:]=0.
    if out[3]>0:
        blend=math.exp(-(float(out[0])**2+float(out[1])**2)/.04)
        velocity=np.array(state[8:10])*10.
        velocity[np.abs(velocity)<.05]=0.
        out[8:10]=velocity*blend
    return out

class ObservationMixin:
    def observe(self,*args,**kwargs):
        state,row=super().observe(*args,**kwargs);extended=transform(state)
        row.update(adapter_motion_inputs=extended[8:10].tolist(),adapter_scope='Only RGB/camera-derived velocityfeatures blendedneartrackingregion; no manualaction or rawduration rule')
        return extended,row

class ObservationEnvironment(ObservationMixin,base.ObservationEnvironment):pass
class TrainingEnvironment(ObservationMixin,base.TrainingEnvironment):pass

def create_agent():
    a=base.create_agent()
    for name in NETWORKS:
        net=getattr(a,name)
        if name in ('actor_net','param_net'):
            # Original update accesses every online .grad unconditionally.
            # Keep gradient tensors allocated, masking frozen entries to zero.
            for p in net.parameters():
                p.requires_grad_(True);mask=torch.zeros_like(p)
                if p is net.layers[0].weight:mask[:,8:15]=1.
                p.register_hook(lambda grad,m=mask:grad*m)
        else:
            for p in net.parameters():p.requires_grad_(False)
    for group in a.actor_optimiser.param_groups:group['lr']=.001
    for group in a.param_net_optimiser.param_groups:group['lr']=.0001
    return a

def resume(agent,path):
    s=torch.load(path,map_location=agent.device,weights_only=False);assert s['metadata']['motion_profile']==PROFILE
    for n in (*NETWORKS,*OPTIMIZERS):getattr(agent,n).load_state_dict(s[n])
    for k,v in s['replay'].items():setattr(agent.memory,k,v)
    for k in ('update_count','actions_count','epsilon'):setattr(agent,k,s[k])
    restore_rng(agent,s['rng']);return s
