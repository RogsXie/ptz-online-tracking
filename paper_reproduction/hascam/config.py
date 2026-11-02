"""Paper settings, with implementation assumptions recorded beside each option."""
from dataclasses import dataclass, asdict


@dataclass
class PDQNConfig:
    state_dim: int = 8
    num_actions: int = 7
    hidden_layers: tuple[int, ...] = (128, 256, 128, 64)
    min_duration: float = .04
    max_duration: float = 1.0
    gamma: float = .90
    batch_size: int = 512
    replay_capacity: int = 20_000  # PDF, not agent.py's overridden 100k default.
    actor_lr: float = .001  # Q_Actor in author code.
    param_lr: float = .0001
    update_interval: int = 32
    target_update_interval: int = 100  # Author agent.py.
    param_grad_clip: float = 10.0
    # Linear outputs + inverting gradients remain the original algorithm.
    # Execution/Q/replay candidates are clamped to the physical paper bounds.
    project_durations: bool = True
    # Output init is unspecified by the paper. Feasible midpoint initialization
    # avoids an all-minimum-duration initial policy after physical clipping.
    feasible_param_init: bool = True
    max_steps: int = 128
    confidence_threshold: float = .9
    consecutive_steps: int = 10
    reward_mode: str = "author_code"

    def __post_init__(self):
        if self.state_dim != 8 or self.num_actions != 7:
            raise ValueError("Paper HAS-CAM requires 8 states and 7 action types")
        if not 0 < self.min_duration < self.max_duration:
            raise ValueError("Invalid duration bounds")
        if not 0 < self.batch_size <= self.replay_capacity:
            raise ValueError("Replay capacity must fit a batch")
        if self.target_update_interval < 1 or self.update_interval < 1:
            raise ValueError("Update intervals must be positive")

    def as_dict(self):
        return asdict(self)


ACTION_NAMES = ("Stay", "Down", "Up", "Right", "Left", "ZoomIn", "ZoomOut")
# UE pan is yaw+, tilt is pitch+, zoom+ narrows FOV.
ACTION_RATES = ((0., 0., 0.), (0., -1., 0.), (0., 1., 0.),
                (1., 0., 0.), (-1., 0., 0.), (0., 0., 1.), (0., 0., -1.))


def epsilon(completed_episodes: int) -> float:
    """PDF Eq13. e means completed episodes, hence epsilon(0)==0.9."""
    e = max(0, int(completed_episodes))
    return .9 - (.8 / 300) * e if e < 300 else .1
