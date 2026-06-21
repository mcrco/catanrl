from __future__ import annotations

from typing import Tuple

import numpy as np


def compute_gae_batched(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    next_values: np.ndarray,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Batched Generalized Advantage Estimation for vectorized envs.

    ``dones[t]`` marks that the episode terminated (or was truncated) *at* step
    ``t``; because the vectorized envs auto-reset, ``values[t + 1]`` is the value
    of the *next* episode's first state. The next-step bootstrap must therefore be
    zeroed at episode boundaries, otherwise a terminal step's advantage leaks the
    value of an unrelated freshly-reset state.

    Args:
        rewards: [T, E] rewards
        values: [T, E] value estimates
        dones: [T, E] episode done flags (terminated or truncated at step t)
        next_values: [E] bootstrap values for the state following the last step
    Returns:
        advantages: [T, E]
        returns: [T, E]
    """
    rewards = np.asarray(rewards, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32)
    dones = np.asarray(dones, dtype=np.bool_)
    next_values = np.asarray(next_values, dtype=np.float32)

    time_steps, num_envs = rewards.shape
    advantages = np.zeros_like(rewards, dtype=np.float32)
    last_gae = np.zeros(num_envs, dtype=np.float32)

    for t in reversed(range(time_steps)):
        next_nonterminal = (~dones[t]).astype(np.float32)
        next_value_t = next_values if t == time_steps - 1 else values[t + 1]
        delta = rewards[t] + gamma * next_nonterminal * next_value_t - values[t]
        last_gae = delta + gamma * gae_lambda * next_nonterminal * last_gae
        advantages[t] = last_gae

    returns = advantages + values
    return advantages, returns
