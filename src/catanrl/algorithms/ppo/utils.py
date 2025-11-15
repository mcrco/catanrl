from __future__ import annotations

from typing import Tuple

import numpy as np


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    next_value: float,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generalized Advantage Estimation shared between SARL and MARL trainers.
    """
    advantages = np.zeros_like(rewards)
    last_gae = 0.0

    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_value_t = 0.0 if dones[t] else next_value
        else:
            next_value_t = values[t + 1]

        delta = rewards[t] + gamma * next_value_t - values[t]
        last_gae = delta + gamma * gae_lambda * (1 - dones[t]) * last_gae
        advantages[t] = last_gae

    returns = advantages + values
    return advantages, returns

