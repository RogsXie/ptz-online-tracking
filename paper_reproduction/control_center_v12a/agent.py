"""Phase A: v10 center objective with explicitly separated 64/64 rehearsal."""
from pathlib import Path
from dataclasses import replace
import hashlib
import numpy as np
import torch
from hascam.agent import ReplayBuffer
from control_value_v10.agent import FactoredPotentialAgent
from control_value_v9.agent import control_config

SOURCE=Path(__file__).resolve().parents[1]/'runs/potential-factored-v10-attempt1/checkpoint-0060.pt'
SOURCE_SHA='c13f0c7ee6448a960e66d8b3f895fe1f605c5c3a9422f6ca1c7c40f41ded629e'


class MixedReplay(ReplayBuffer):
    def __init__(self,capacity,rng,history):
        super().__init__(capacity,rng)
        self.history=tuple(history)
        self.last_sample=None

    def sample(self,batch_size):
        if batch_size!=128:raise ValueError('Declared batch must be64new+64historical')
        ni=self.rng.choice(len(self.buffer),64,replace=False)
        hi=self.rng.choice(len(self.history),64,replace=False)
        selected=[self.buffer[i] for i in ni]+[self.history[i] for i in hi]
        self.last_sample=dict(new_indices=ni.tolist(),historical_indices=hi.tolist(),
            new_count=64,historical_count=64,new_pool_size=len(self.buffer),historical_pool_size=len(self.history))
        s,a,p,r,ns,d=zip(*selected)
        return (np.stack(s).astype(np.float32),np.array(a,np.int64),np.stack(p).astype(np.float32),
                np.array(r,np.float32),np.stack(ns).astype(np.float32),np.array(d,np.bool_))

    def state_dict(self):
        return super().state_dict()|dict(historical_source=str(SOURCE),historical_sha256=SOURCE_SHA,
            historical_count=len(self.history),scope='buffer contains only NEW real training transitions')

    def load_state_dict(self,state):
        assert state['historical_sha256']==SOURCE_SHA and state['historical_count']==len(self.history)
        super().load_state_dict(state)


class CenterAgent(FactoredPotentialAgent):
    FORMAT='center_repair_rehearsal_v12a'

    def __init__(self,config=None,seed=20261124,device='cuda',allowed_actions=(0,1,2,3,4)):
        super().__init__(config or replace(control_config(),reward_mode=self.FORMAT),seed,device,allowed_actions)
        assert tuple(allowed_actions)==(0,1,2,3,4)
        if hashlib.sha256(SOURCE.read_bytes()).hexdigest()!=SOURCE_SHA:raise ValueError('Historical checkpoint changed')
        cp=torch.load(SOURCE,map_location='cpu',weights_only=False)
        history=cp['replay']['buffer']
        assert len(history)==30060
        assert all(a in allowed_actions and not d and abs(float(p[a])-.04)<1e-7 for s,a,p,r,n,d in history)
        self.memory=MixedReplay(self.memory.capacity,self.replay_rng,history)

    def initialize_from_v10(self):
        assert not len(self.memory) and not self.actor_optimiser.state and self.update_count==0
        cp=torch.load(SOURCE,map_location=self.device,weights_only=False)
        for name in ('actor_net','param_net','param_net_target'):
            getattr(self,name).load_state_dict(cp[name])
        self.actor_target.load_state_dict(cp['actor_net'])
        self.param_net.requires_grad_(False);self.param_net_target.requires_grad_(False)

    def update(self):
        result=super().update()
        if result is not None:
            result.update(rehearsal=self.memory.last_sample,new_environment_steps=self.environment_steps)
        return result
