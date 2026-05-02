from __future__ import annotations

from functools import lru_cache
from typing import Dict, Literal, Tuple

import numpy as np

from catanatron.features import create_sample, get_feature_ordering
from catanatron.game import Game
from catanatron.gym.board_tensor_features import create_board_tensor, is_graph_feature
from catanatron.models.player import Color, RandomPlayer
from catanrl.utils.catanatron_map import build_catan_map

COLOR_ORDER: Tuple[Color, ...] = (
    Color.RED,
    Color.BLUE,
    Color.ORANGE,
    Color.WHITE,
)
ObservationLevel = Literal["private", "public", "full"]
ActorObservationLevel = ObservationLevel
CriticObservationLevel = ObservationLevel
RESOURCE_IN_HAND_SUFFIXES = frozenset(
    {
        "BRICK_IN_HAND",
        "ORE_IN_HAND",
        "SHEEP_IN_HAND",
        "WHEAT_IN_HAND",
        "WOOD_IN_HAND",
    }
)


def _build_dummy_game(
    num_players: int, map_type: Literal["BASE", "MINI", "TOURNAMENT"]
) -> Tuple[Tuple[Color, ...], Game]:
    """Construct a deterministic dummy game for feature metadata helpers."""
    if num_players <= 0:
        raise ValueError("num_players must be positive")
    colors = COLOR_ORDER[:num_players]
    dummy_players = [RandomPlayer(color) for color in colors]
    dummy_game = Game(dummy_players, catan_map=build_catan_map(map_type))
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


def _validate_observation_level(level: ObservationLevel, *, context: str) -> None:
    if level not in ("private", "public", "full"):
        raise ValueError(
            f"Unknown {context} '{level}'. "
            "Expected one of: private, public, full."
        )


@lru_cache(maxsize=None)
def get_observation_numeric_feature_names(
    num_players: int,
    map_type: Literal["BASE", "MINI", "TOURNAMENT"],
    level: ObservationLevel = "private",
) -> Tuple[str, ...]:
    """Return numeric feature names for the selected information level.

    Ordering always follows the full privileged numeric vector so these names can
    be used to index directly into `full_game_to_features`.
    """
    _validate_observation_level(level, context="actor_observation_level")
    if level == "private":
        return get_numeric_feature_names(num_players, map_type)
    if level == "full":
        return get_full_numeric_feature_names(num_players, map_type)

    if num_players != 2:
        raise ValueError("observation level 'public' is currently supported only for 1v1.")

    private_names = set(get_numeric_feature_names(num_players, map_type))
    public_opponent_resource_names = {
        f"P1_{suffix}"
        for suffix in _private_feature_suffixes(num_players, map_type)
        if suffix in RESOURCE_IN_HAND_SUFFIXES
    }
    selected = private_names | public_opponent_resource_names
    return tuple(
        name for name in get_full_numeric_feature_names(num_players, map_type) if name in selected
    )


def compute_feature_vector_dim(
    num_players: int, map_type: Literal["BASE", "MINI", "TOURNAMENT"]
) -> int:
    """Return total flattened feature vector length (numeric + board tensor)."""
    numeric_len = len(get_numeric_feature_names(num_players, map_type))
    return numeric_len + _board_tensor_size(num_players, map_type)


def compute_observation_feature_vector_dim(
    num_players: int,
    map_type: Literal["BASE", "MINI", "TOURNAMENT"],
    level: ObservationLevel = "private",
) -> int:
    """Return flattened feature length for the selected observation level."""
    numeric_len = len(get_observation_numeric_feature_names(num_players, map_type, level))
    return numeric_len + _board_tensor_size(num_players, map_type)


def get_observation_indices_from_full(
    num_players: int,
    map_type: Literal["BASE", "MINI", "TOURNAMENT"],
    level: ObservationLevel = "private",
) -> np.ndarray:
    """Return indices into the full feature vector for an observation level.

    All flattened observations use [numeric, board]. Numeric features are selected
    by name for the requested level; the board tail is always preserved.
    """
    full_names = get_full_numeric_feature_names(num_players, map_type)
    full_name_to_index = {name: idx for idx, name in enumerate(full_names)}
    observation_numeric_names = get_observation_numeric_feature_names(num_players, map_type, level)
    full_numeric_dim = len(get_full_numeric_feature_names(num_players, map_type))
    board_dim = _board_tensor_size(num_players, map_type)

    numeric_indices = np.array(
        [full_name_to_index[name] for name in observation_numeric_names], dtype=np.int64
    )
    board_indices = np.arange(full_numeric_dim, full_numeric_dim + board_dim)
    return np.concatenate([numeric_indices, board_indices])


def get_actor_indices_from_full(
    num_players: int,
    map_type: Literal["BASE", "MINI", "TOURNAMENT"],
    level: ActorObservationLevel = "private",
) -> np.ndarray:
    """Return indices into the full feature vector that yield actor features."""
    return get_observation_indices_from_full(num_players, map_type, level)


def get_actor_indices_from_critic(
    num_players: int, map_type: Literal["BASE", "MINI", "TOURNAMENT"]
) -> np.ndarray:
    """Return private actor indices from the full critic feature vector."""
    return get_actor_indices_from_full(num_players, map_type, level="private")


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
