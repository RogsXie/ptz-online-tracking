"""A ten-second collection segment does not terminate continuous holding."""
from control_v4.environment import StaticEnvironment, StaticState

STATE_FEATURES=["4*normalized_x","4*normalized_y","log(area/.04)/4","confidence_or0",
    "reserved0","reserved0","reserved0","reserved0"]


class ContinuingState(StaticState):
    def build(self, detection, action=(0,0.)):
        state=super().build(detection,action)
        state[4:]=0.
        return state


class ContinuingEnvironment(StaticEnvironment):
    def reset(self, scenario, seed):
        super().reset(scenario,seed)
        self.transport.mode("CONTINUING_STATIC_PDQN_V5")
        self.builder=ContinuingState(self.spec["horizon_seconds"])
        return self.builder.build(self.selected)

    def step(self, action, duration):
        state,reward,collector_done,row=super().step(action,duration)
        # The stationary holding task has no absorbing success/loss event.
        # The reset only diversifies the next collection segment. Bootstrap
        # from this exact final RGB observation, never the next reset image.
        row["terminated"]=False
        row["truncated"]=bool(collector_done)
        row["bootstrap_from_actual_next_observation"]=True
        return state,reward,collector_done,row


def transition_terminal(row):
    """Only task termination, not a collector time limit, masks target Q."""
    if not isinstance(row.get("terminated"),bool) or not isinstance(row.get("truncated"),bool):
        raise ValueError("Explicit terminated/truncated flags are required")
    return row["terminated"]
