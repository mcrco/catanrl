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

    Args:
        rewards: [T, E] rewards
        values: [T, E] value estimates
        dones: [T, E] episode done flags
        next_values: [E] bootstrap values for final timestep
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
        if t == time_steps - 1:
            next_value_t = np.where(dones[t], 0.0, next_values)
        else:
            next_value_t = values[t + 1]
        delta = rewards[t] + gamma * next_value_t - values[t]
        last_gae = delta + gamma * gae_lambda * (~dones[t]) * last_gae
        advantages[t] = last_gae

    returns = advantages + values
    return advantages, returns
