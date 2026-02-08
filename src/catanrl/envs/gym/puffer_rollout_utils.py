"""
Utility helpers for PufferLib vectorized rollouts.

These helpers are intentionally model-agnostic (no torch imports) so they can
be reused by both training and evaluation loops.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np


def flatten_puffer_observation(obs: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """Convert a Puffer observation dict to flattened actor/critic vectors."""
    actor_obs = obs["observation"]
    numeric = np.asarray(actor_obs["numeric"], dtype=np.float32).reshape(-1)
    board = np.asarray(actor_obs["board"], dtype=np.float32)
    board_wh_last = np.transpose(board, (1, 2, 0))
    board_flat = board_wh_last.reshape(-1)
    actor_vec = np.concatenate([numeric, board_flat], axis=0)
    critic_vec = np.asarray(obs["critic"], dtype=np.float32).reshape(-1)
    return actor_vec, critic_vec


def get_action_mask_from_obs(obs: Dict[str, Any]) -> np.ndarray:
    """Extract action mask from a Puffer observation."""
    return np.asarray(obs["action_mask"], dtype=np.int8) > 0


@dataclass
class EpisodeBuffer:
    critic_states: List[np.ndarray]
    rewards: List[float]
    steps: int = 0


def init_episode_buffers(num_envs: int) -> List[EpisodeBuffer]:
    """Initialize per-env episode buffers for vectorized rollouts."""
    return [EpisodeBuffer(critic_states=[], rewards=[], steps=0) for _ in range(num_envs)]
