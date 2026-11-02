"""One temporal objective for replay targets, rates and shaping."""
import math

# Preserve v5's discount at its original midpoint action duration.
REFERENCE_SECONDS = .145
REFERENCE_GAMMA = .98
DECAY_PER_SECOND = -math.log(REFERENCE_GAMMA) / REFERENCE_SECONDS


def discount(duration):
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("Finite positive physical duration required")
    return math.exp(-DECAY_PER_SECOND * duration)


def discounted_seconds(duration):
    discount(duration)
    return -math.expm1(-DECAY_PER_SECOND * duration) / DECAY_PER_SECOND


def image_cost(observation):
    # Static fixed-FOV stage only. No GT or simulator target geometry.
    return float(observation["center_error"]) if observation["visible"] else 2.


def temporal_reward(before, after, action, duration):
    old_cost, new_cost = image_cost(before), image_cost(after)
    rate = (1. if new_cost <= .025 else -.2) - 6.*new_cost if after["visible"] else -8.
    factor = discount(duration)
    # Phi(s)=-20*cost(s); gamma(dt)*Phi(s')-Phi(s).
    shaping = 20.*(old_cost-factor*new_cost)
    rate_reward = rate * discounted_seconds(duration)
    effort = -.02 if action != 0 else 0.
    return shaping+rate_reward+effort, {
        "image_progress": shaping, "image_rate": rate, "duration": duration,
        "discounted_duration_s": discounted_seconds(duration), "discount_factor": factor,
        "rate_reward": rate_reward, "effort": effort,
        "rate_quadrature": "post-action observed RGB rate held over interval; no unobserved trajectory claim"}
