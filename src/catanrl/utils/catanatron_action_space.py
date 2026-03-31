from __future__ import annotations

from functools import lru_cache
from typing import Dict, Literal, Tuple

import numpy as np
from catanatron.gym.envs.action_space import (
    ACTION_TYPES,
    from_action_space as upstream_from_action_space,
    get_action_array as upstream_get_action_array,
    to_action_space as upstream_to_action_space,
)
from catanatron.models.actions import Action
from catanatron.models.enums import ActionType
from catanatron.models.player import Color

MapType = Literal["BASE", "TOURNAMENT", "MINI"]

# catanrl consistently uses Blue as the controlled player first.
PLAYER_COLOR_ORDER: Tuple[Color, ...] = (
    Color.BLUE,
    Color.RED,
    Color.WHITE,
    Color.ORANGE,
)


def get_player_colors(num_players: int) -> Tuple[Color, ...]:
    if not 1 <= num_players <= len(PLAYER_COLOR_ORDER):
        raise ValueError(f"num_players must be in [1, {len(PLAYER_COLOR_ORDER)}], got {num_players}")
    return PLAYER_COLOR_ORDER[:num_players]


@lru_cache(maxsize=None)
def get_action_array(
    num_players: int,
    map_type: MapType,
) -> Tuple[tuple[ActionType, object | None], ...]:
    return tuple(upstream_get_action_array(get_player_colors(num_players), map_type))


@lru_cache(maxsize=None)
def get_action_space_size(num_players: int, map_type: MapType) -> int:
    return len(get_action_array(num_players, map_type))


def to_action_space(action: Action, num_players: int, map_type: MapType) -> int:
    return upstream_to_action_space(action, get_player_colors(num_players), map_type)


def from_action_space(action_int: int, color: Color, num_players: int, map_type: MapType) -> Action:
    return upstream_from_action_space(action_int, color, get_player_colors(num_players), map_type)


@lru_cache(maxsize=None)
def get_end_turn_index(num_players: int, map_type: MapType) -> int:
    action_array = get_action_array(num_players, map_type)
    for idx, (action_type, _) in enumerate(action_array):
        if action_type == ActionType.END_TURN:
            return idx
    raise ValueError("END_TURN action not found in action space")


@lru_cache(maxsize=None)
def build_action_type_metadata(
    num_players: int,
    map_type: MapType,
) -> tuple[Dict[str, list[int]], np.ndarray, list[str]]:
    action_array = get_action_array(num_players, map_type)
    action_type_to_indices: Dict[str, list[int]] = {action_type.name: [] for action_type in ACTION_TYPES}
    action_to_type_idx = np.zeros(len(action_array), dtype=np.int32)
    for action_idx, (action_type, _) in enumerate(action_array):
        action_type_to_indices[action_type.name].append(action_idx)
        action_to_type_idx[action_idx] = ACTION_TYPES.index(action_type)
    action_type_names = [action_type.name for action_type in ACTION_TYPES]
    return action_type_to_indices, action_to_type_idx, action_type_names
