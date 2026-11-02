"""Same isolated Q learner; incompatible observation/checkpoint contract."""
from dataclasses import replace
from control_v3.agent import ImprovedAgent
from control_learning.agent import control_config as previous_config


def control_config():
    return replace(previous_config(), max_steps=256, reward_mode="rgb_center_dense_v3")


class StaticAgent(ImprovedAgent):
    FORMAT = "static_observation_pdqn_v4"

    def __init__(self, config=None, seed=20261102, device="cuda", allowed_actions=(0,1,2,3,4)):
        super().__init__(config or control_config(), seed, device, allowed_actions)
