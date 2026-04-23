from __future__ import annotations

from typing import Dict, Sequence, Tuple

import numpy as np
from catanatron.gym.envs.action_space import ACTION_TYPES


def compute_action_distributions(
    actions: np.ndarray,
    action_to_type_idx: np.ndarray,
    action_type_names: Sequence[str],
    action_space_size: int,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Compute normalized action-type and raw-action frequencies."""
    n_actions = len(actions)
    if n_actions == 0:
        return {}, {}

    action_types_taken = action_to_type_idx[actions]
    type_counts = np.bincount(action_types_taken, minlength=len(ACTION_TYPES))
    action_type_dist = {
        name: float(count / n_actions) for name, count in zip(action_type_names, type_counts)
    }

    action_counts = np.bincount(actions, minlength=action_space_size)
    action_dist = {
        f"action_{idx}": float(count / n_actions)
        for idx, count in enumerate(action_counts)
        if count > 0
    }
    return action_type_dist, action_dist


def build_raw_policy_log_dict(
    raw_policy_argmax_buffer: list[np.ndarray],
    action_to_type_idx: np.ndarray,
    action_type_names: Sequence[str],
    action_space_size: int,
    rollout_steps_collected: int,
    rollout_samples_collected: int,
) -> dict[str, float]:
    """Convert raw argmax traces into the common WandB logging payload."""
    if not raw_policy_argmax_buffer:
        return {}

    all_raw_argmax = np.concatenate(raw_policy_argmax_buffer)
    raw_action_type_dist, _ = compute_action_distributions(
        all_raw_argmax,
        action_to_type_idx,
        list(action_type_names),
        action_space_size,
    )
    raw_policy_log = {
        f"raw_policy/type_{name}": freq for name, freq in raw_action_type_dist.items()
    }
    raw_policy_log.update(
        {
            "train/rollout_steps_collected": float(rollout_steps_collected),
            "train/rollout_samples_collected": float(rollout_samples_collected),
        }
    )
    return raw_policy_log


__all__ = ["build_raw_policy_log_dict", "compute_action_distributions"]
