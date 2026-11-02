"""v5 observation/truncation and mechanics, with physical-time reward."""
from control_v5.environment import ContinuingEnvironment, STATE_FEATURES, transition_terminal
from control_v6.temporal import temporal_reward


class TemporalEnvironment(ContinuingEnvironment):
    def reset(self, scenario, seed):
        state = super().reset(scenario, seed)
        self.transport.mode("TEMPORAL_STATIC_PDQN_V6")
        return state

    def step(self, action, duration):
        before = self.previous.copy()
        state, _, collector_done, row = super().step(action, duration)
        reward, parts = temporal_reward(before, row, action, duration)
        # The inherited audit list contains this same row object.
        row.update(reward=float(reward), reward_parts=parts)
        return state, float(reward), collector_done, row
