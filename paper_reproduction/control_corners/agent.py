"""New observation coverage means a new checkpoint format and fresh replay."""
from dataclasses import replace
from control_v6.agent import TemporalAgent, control_config as temporal_config


def control_config():
    return replace(temporal_config(),max_steps=512,reward_mode='physical_time_four_corners_v7')


class CornerAgent(TemporalAgent):
    FORMAT='corner_static_pdqn_v7'

    def __init__(self,config=None,seed=20261102,device='cuda',allowed_actions=(0,1,2,3,4)):
        super().__init__(config or control_config(),seed,device,allowed_actions)
