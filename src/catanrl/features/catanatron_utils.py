from __future__ import annotations

from functools import lru_cache
from typing import Dict, Literal, Tuple

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


def _build_dummy_game(
    num_players: int, map_type: Literal["BASE", "MINI", "TOURNAMENT"]
) -> Tuple[Tuple[Color, ...], Game]:
    """Construct a deterministic dummy game for feature metadata helpers."""
    if num_players <= 0:
        raise ValueError("num_players must be positive")
    colors = COLOR_ORDER[:num_players]
    dummy_players = [RandomPlayer(color) for color in colors]
    dummy_game = Game(dummy_players, catan_map=build_map(map_type))
    return colors, dummy_game


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
def get_numeric_feature_names(
    num_players: int, map_type: Literal["BASE", "MINI", "TOURNAMENT"]
) -> Tuple[str, ...]:
    """Return the non-graph supervised feature names for the given setup."""
    ordering = get_feature_ordering(num_players, map_type)
    return tuple(name for name in ordering if not is_graph_feature(name))


@lru_cache(maxsize=None)
def _board_tensor_size(num_players: int, map_type: Literal["BASE", "MINI", "TOURNAMENT"]) -> int:
    """Compute board tensor length for the given setup (cached)."""
    colors, dummy_game = _build_dummy_game(num_players, map_type)
    board_tensor = create_board_tensor(dummy_game, colors[0])
    return int(board_tensor.size)


@lru_cache(maxsize=None)
def _private_feature_suffixes(
    num_players: int, map_type: Literal["BASE", "MINI", "TOURNAMENT"]
) -> Tuple[str, ...]:
    """Feature name suffixes that only exist for the perspective player."""
    colors, dummy_game = _build_dummy_game(num_players, map_type)
    sample = create_sample(dummy_game, colors[0])
    suffixes = sorted(name[3:] for name in sample if name.startswith("P0_"))
    return tuple(suffixes)


@lru_cache(maxsize=None)
def get_full_numeric_feature_names(
    num_players: int, map_type: Literal["BASE", "MINI", "TOURNAMENT"]
) -> Tuple[str, ...]:
    """
    Return numeric feature ordering that includes private info for every player.
    """
    base_names = list(get_numeric_feature_names(num_players, map_type))
    suffixes = _private_feature_suffixes(num_players, map_type)
    for player_idx in range(1, num_players):
        for suffix in suffixes:
            feature_name = f"P{player_idx}_{suffix}"
            if feature_name not in base_names:
                base_names.append(feature_name)
    return tuple(base_names)


def compute_feature_vector_dim(
    num_players: int, map_type: Literal["BASE", "MINI", "TOURNAMENT"]
) -> int:
    """Return total flattened feature vector length (numeric + board tensor)."""
    numeric_len = len(get_numeric_feature_names(num_players, map_type))
    return numeric_len + _board_tensor_size(num_players, map_type)


def get_actor_indices_from_critic(
    num_players: int, map_type: Literal["BASE", "MINI", "TOURNAMENT"]
) -> np.ndarray:
    """Return indices into critic feature vector that yield actor feature vector.
    
    The actor observation is [numeric, board] and the critic observation is
    [full_numeric, board]. Since numeric is a prefix of full_numeric and the
    board is identical, actor can be extracted from critic as:
        actor = critic[indices]
    where indices = [0..numeric_dim-1, full_numeric_dim..end]
    
    This allows storing only critic observations and deriving actor on demand.
    """
    numeric_dim = len(get_numeric_feature_names(num_players, map_type))
    full_numeric_dim = len(get_full_numeric_feature_names(num_players, map_type))
    board_dim = _board_tensor_size(num_players, map_type)
    
    numeric_indices = np.arange(numeric_dim)
    board_indices = np.arange(full_numeric_dim, full_numeric_dim + board_dim)
    return np.concatenate([numeric_indices, board_indices])


def _augment_with_private_features(
    game: Game, base_sample: Dict[str, float], num_players: int, base_color: Color
) -> Dict[str, float]:
    """
    Add every player's private info (resources, hidden VPs) into the sample.

    Players are ordered relative to the acting player (P0), ensuring the mapping
    is consistent with how `create_sample` rotates perspective-specific fields.
    """
    colors = tuple(game.state.colors)
    if len(colors) != num_players:
        raise ValueError(
            f"Expected game with {num_players} players, found {len(colors)} players instead."
        )
    try:
        base_index = colors.index(base_color)
    except ValueError as exc:  # pragma: no cover - defensive
        raise ValueError(f"Base color {base_color} not found in game colors {colors}") from exc

    augmented = dict(base_sample)
    for player_offset in range(num_players):
        color = colors[(base_index + player_offset) % num_players]
        perspective_sample = create_sample(game, color)
        for key, value in perspective_sample.items():
            if key.startswith("P0_"):
                suffix = key[3:]
                augmented[f"P{player_offset}_{suffix}"] = value
    return augmented


def game_to_features(
    game: Game,
    color: Color,
    num_players: int,
    map_type: Literal["BASE", "MINI", "TOURNAMENT"],
) -> np.ndarray:
    """Convert a game state into the flattened feature vector used across players."""
    numeric_feature_names = get_numeric_feature_names(num_players, map_type)
    sample = create_sample(game, color)
    numeric_vector = np.array(
        [float(sample.get(name, 0.0)) for name in numeric_feature_names], dtype=np.float32
    )
    board_tensor = create_board_tensor(game, color).astype(np.float32, copy=False).reshape(-1)
    return np.concatenate([numeric_vector, board_tensor], axis=0)


def full_game_to_features(
    game: Game,
    num_players: int,
    map_type: Literal["BASE", "MINI", "TOURNAMENT"],
    base_color: Color | None = None,
) -> np.ndarray:
    """
    Convert the full game state into a perfect-information flattened feature vector.

    This mirrors `game_to_features` but injects every player's private information,
    making it suitable for centralized critics that observe the joint state.
    """
    colors = tuple(game.state.colors)
    if len(colors) != num_players:
        raise ValueError(
            f"Expected game with {num_players} players, found {len(colors)} players instead."
        )

    base_color = base_color or game.state.current_color()
    if base_color not in colors:  # pragma: no cover - defensive
        raise ValueError(f"Current color {base_color} not present in game state {colors}.")

    numeric_feature_names = get_full_numeric_feature_names(num_players, map_type)
    base_sample = create_sample(game, base_color)
    numeric_sample = _augment_with_private_features(game, base_sample, num_players, base_color)
    numeric_vector = np.array(
        [float(numeric_sample.get(name, 0.0)) for name in numeric_feature_names],
        dtype=np.float32,
    )
    board_tensor = create_board_tensor(game, base_color).astype(np.float32, copy=False).reshape(-1)
    return np.concatenate([numeric_vector, board_tensor], axis=0)
