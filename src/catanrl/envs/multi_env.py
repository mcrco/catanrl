from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from catanatron.features import get_feature_ordering
from catanatron.gym.board_tensor_features import get_channels, is_graph_feature
from catanatron.zoo.env.catanatron_aec_env import CatanatronAECEnv, aec_env


def build_marl_env(config: Dict) -> CatanatronAECEnv:
    """Create a PettingZoo AEC environment with the given configuration."""
    return aec_env(config)


def compute_multiagent_input_dim(
    num_players: int, map_type: str, representation: str
) -> Tuple[int, Optional[Tuple[int, int, int]], int]:
    """
    Compute flattened observation dimensions for multi-agent training.
    Returns (input_dim, board_shape, numeric_dim).
    """
    features = get_feature_ordering(num_players, map_type)
    if representation == "mixed":
        numeric_features = [f for f in features if not is_graph_feature(f)]
        channels = get_channels(num_players)
        board_shape = (channels, 21, 11)
        board_dim = int(np.prod(board_shape))
        numeric_dim = len(numeric_features)
        return numeric_dim + board_dim, board_shape, numeric_dim
    return len(features), None, len(features)


def flatten_marl_observation(observation: np.ndarray, representation: str) -> np.ndarray:
    """Flatten observation dicts into 1D feature vectors."""
    if representation == "mixed":
        numeric = observation["numeric"].astype(np.float32, copy=False)
        board = observation["board"].astype(np.float32, copy=False)
        board_ch_last = np.transpose(board, (1, 2, 0))
        board_flat = board_ch_last.reshape(-1).astype(np.float32, copy=False)
        return np.concatenate([numeric, board_flat], axis=0)
    return observation.astype(np.float32, copy=False)


def get_valid_actions(info: Dict) -> np.ndarray:
    """Extract valid action indices from PettingZoo info dict."""
    actions = info.get("valid_actions")
    if not actions:
        from catanatron.gym.envs.catanatron_env import ACTION_SPACE_SIZE

        return np.arange(ACTION_SPACE_SIZE, dtype=np.int64)
    return np.array(actions, dtype=np.int64)

