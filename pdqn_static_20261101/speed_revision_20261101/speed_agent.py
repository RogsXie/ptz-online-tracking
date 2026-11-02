"""Use the original author's class and update with declared action bounds."""
import speed_bootstrap
from speed_bootstrap import REV,LEGACY
import json,random
import numpy as np,torch,gymnasium as gym
from author_aligned.source import load_core,verify_source
from author_aligned.run import NETWORKS

PROFILE=json.loads((REV/'training_profile.json').read_text(encoding='utf-8'))

def create_agent(seed=20261101,device='cuda'):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    klass,_,_=load_core()
    space=gym.spaces.Tuple((gym.spaces.Discrete(7),*[
        gym.spaces.Box(np.array([lo],np.float32),np.array([hi],np.float32))
        for lo,hi in zip(PROFILE['raw_parameter_lows'],PROFILE['raw_parameter_highs'])]))
    return klass(gym.spaces.Discrete(8),space,batch_size=512,gamma=.9,seed=seed,
        actor_kwargs={'hidden_layers':[128,256,128,64]},param_net_kwargs={'hidden_layers':[128,256,128,64]},device=device)

def transfer(agent,parent):
    saved=torch.load(parent,map_location=agent.device,weights_only=False)
    for name in NETWORKS:getattr(agent,name).load_state_dict(saved[name])
    scales=torch.tensor(PROFILE['transfer_scales'],device=agent.device,dtype=torch.float32)
    with torch.no_grad():
        for name in ('param_net','param_net_target'):
            layer=getattr(agent,name).layers[-1];layer.weight.mul_(scales[:,None]);layer.bias.mul_(scales)
        for name in ('actor_net','actor_target'):getattr(agent,name).layers[0].weight[:,8:].div_(scales[None,:])
    agent.epsilon=.1
    assert len(agent.memory)==0 and agent.update_count==0
    return dict(parent=str(parent),parent_completed_episodes=saved['completed_episodes'],parent_updates=saved['update_count'],
        new_replay_size=0,new_updates=0,initialization=PROFILE['initialization'],source_sha256=verify_source())
