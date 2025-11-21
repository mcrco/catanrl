from __future__ import annotations

from functools import lru_cache
from typing import Sequence, Tuple

import numpy as np

from catanatron.features import create_sample, get_feature_ordering
from catanatron.game import Game
from catanatron.gym.board_tensor_features import create_board_tensor, is_graph_feature
from catanatron.models.map import build_map
from catanatron.models.player import Color, RandomPlayer

COLOR_ORDER: Tuple[Color, ...] = (
    Color.RED,
    Color.BLUE,
    Color.ORANGE,
    Color.WHITE,
)


def is_non_graph_feature_parquet(col: str) -> bool:
    if not col.startswith("F_"):
        return False
    return not (
        col.startswith("F_NODE")
        or col.startswith("F_EDGE")
        or col.startswith("F_TILE")
        or col.startswith("F_PORT")
    )


@lru_cache(maxsize=None)
def get_numeric_feature_names(num_players: int, map_type: str) -> Tuple[str, ...]:
    """Return the non-graph supervised feature names for the given setup."""
    ordering = get_feature_ordering(num_players, map_type)
    return tuple(name for name in ordering if not is_graph_feature(name))


@lru_cache(maxsize=None)
def _board_tensor_size(num_players: int, map_type: str) -> int:
    """Compute board tensor length for the given setup (cached)."""
    colors = COLOR_ORDER[:num_players]
    dummy_players = [RandomPlayer(color) for color in colors]
    dummy_game = Game(dummy_players, catan_map=build_map(map_type))
    board_tensor = create_board_tensor(dummy_game, colors[0])
    return int(board_tensor.size)


def compute_feature_vector_dim(num_players: int, map_type: str) -> int:
    """Return total flattened feature vector length (numeric + board tensor)."""
    numeric_len = len(get_numeric_feature_names(num_players, map_type))
    return numeric_len + _board_tensor_size(num_players, map_type)


def game_to_features(
    game: Game,
    color: Color,
    numeric_feature_names: Sequence[str],
) -> np.ndarray:
    """Convert a game state into the flattened feature vector used across players."""
    sample = create_sample(game, color)
    numeric_vector = np.array([float(sample.get(name, 0.0)) for name in numeric_feature_names], dtype=np.float32)
    board_tensor = create_board_tensor(game, color).astype(np.float32, copy=False).reshape(-1)
    return np.concatenate([numeric_vector, board_tensor], axis=0)