"""
Utility helpers for PufferLib vectorized rollouts.

These helpers are intentionally model-agnostic (no torch imports) so they can
be reused by both training and evaluation loops.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np


BOARD_WIDTH = 21
BOARD_HEIGHT = 11


def _flatten_actor_components(numeric: np.ndarray, board: np.ndarray) -> np.ndarray:
    """Flatten actor features to match game_to_features ordering.

    Canonical ordering is [numeric, board_flat] where board_flat is flattened from
    board shape (W, H, C). If board arrives as channels-first (C, W, H), convert it.
    """
    numeric_flat = np.asarray(numeric, dtype=np.float32).reshape(-1)
    board_arr = np.asarray(board, dtype=np.float32)
    if board_arr.ndim != 3:
        raise ValueError(f"Expected 3D board tensor, got shape {board_arr.shape}")

    if board_arr.shape[0] == BOARD_WIDTH and board_arr.shape[1] == BOARD_HEIGHT:
        board_wh_last = board_arr
    elif board_arr.shape[1] == BOARD_WIDTH and board_arr.shape[2] == BOARD_HEIGHT:
        board_wh_last = np.transpose(board_arr, (1, 2, 0))
    else:
        raise ValueError(
            "Unrecognized board tensor layout; expected (W,H,C) or (C,W,H), "
            f"got {board_arr.shape}"
        )

    board_flat = board_wh_last.reshape(-1)
    return np.concatenate([numeric_flat, board_flat], axis=0)


def flatten_puffer_observation(obs: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """Convert a Puffer observation dict to flattened actor/critic vectors."""
    actor_obs = obs["observation"]
    actor_vec = _flatten_actor_components(actor_obs["numeric"], actor_obs["board"])
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
