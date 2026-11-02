"""Preserve the isolated-Q learner while declaring a new task contract."""
from control_v4.agent import StaticAgent, control_config


class ContinuingAgent(StaticAgent):
    FORMAT = "continuing_static_pdqn_v5"
