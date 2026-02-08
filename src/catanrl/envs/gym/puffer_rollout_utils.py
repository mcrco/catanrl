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


def extract_expert_actions_from_infos(infos: Any, batch_size: int) -> np.ndarray:
    """Extract expert actions from PufferLib vectorized env infos."""
    expert_actions = np.zeros(batch_size, dtype=np.int64)

    if isinstance(infos, dict) and "expert_action" in infos:
        expert_arr = infos["expert_action"]
        if hasattr(expert_arr, "__len__") and len(expert_arr) == batch_size:
            expert_actions[:] = expert_arr
        else:
            expert_actions[:] = int(expert_arr)
    elif isinstance(infos, (list, tuple)):
        for idx, info in enumerate(infos):
            if isinstance(info, dict) and "expert_action" in info:
                expert_actions[idx] = int(info["expert_action"])
    elif hasattr(infos, "__iter__"):
        for idx, info in enumerate(infos):
            if idx >= batch_size:
                break
            if isinstance(info, dict) and "expert_action" in info:
                expert_actions[idx] = int(info["expert_action"])

    return expert_actions


@dataclass
class EpisodeBuffer:
    critic_states: List[np.ndarray]
    rewards: List[float]
    steps: int = 0


def init_episode_buffers(num_envs: int) -> List[EpisodeBuffer]:
    """Initialize per-env episode buffers for vectorized rollouts."""
    return [EpisodeBuffer(critic_states=[], rewards=[], steps=0) for _ in range(num_envs)]
