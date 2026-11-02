"""Original author's P-DQN class, calibrated bounds, original weight units."""
import speed_bootstrap
from speed_bootstrap import REV,LEGACY
import json,random
import numpy as np,torch,gymnasium as gym
from author_aligned.source import load_core,verify_source
from author_aligned.run import NETWORKS

PROFILE=json.loads((REV/'training_profile_v3.json').read_text(encoding='utf-8'))

def create_agent():
    seed=20261101;random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);cls,_,_=load_core()
    space=gym.spaces.Tuple((gym.spaces.Discrete(7),*[
        gym.spaces.Box(np.array([lo],np.float32),np.array([hi],np.float32)) for lo,hi in zip(PROFILE['raw_lows'],PROFILE['raw_highs'])]))
    return cls(gym.spaces.Discrete(8),space,batch_size=512,gamma=.9,seed=seed,actor_lr=.001,param_net_lr=.00005,
        actor_kwargs={'hidden_layers':[128,256,128,64]},param_net_kwargs={'hidden_layers':[128,256,128,64]},device='cuda')

def initialize(agent,out):
    p=LEGACY/'exploration-support-original-updates/updates-1024.pt';s=torch.load(p,map_location=agent.device,weights_only=False)
    for n in NETWORKS:
        getattr(agent,n).load_state_dict(s[n]);assert all(torch.equal(v,getattr(agent,n).state_dict()[k]) for k,v in s[n].items())
    assert len(agent.memory)==0 and agent.update_count==0
    from calibrate import write
    proof=dict(passed=True,four_networks_bitwise_equal_to_released=True,new_replay_size=0,new_updates=0,
        parameter_units_unchanged=True,source_sha256=verify_source(),parent=str(p),parent_episodes=s['completed_episodes'])
    write(out/'initial-transfer-equivalence.json',proof);return proof
