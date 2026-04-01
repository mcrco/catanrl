from __future__ import annotations

from functools import lru_cache
from typing import Dict, Literal, Tuple

import numpy as np
from catanatron.gym.envs.action_space import ACTION_TYPES
from catanatron.models.actions import Action
from catanatron.models.board import get_edges
from catanatron.models.enums import RESOURCES, ActionType
from catanatron.models.map import build_map
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


def _relative_opponent_slot(actor: Color, victim: Color, game_colors: Tuple[Color, ...]) -> int:
    """Map absolute victim color to 1..n-1 matching P1..P{n-1} like we do in feature extractors."""
    n = len(game_colors)
    start = game_colors.index(actor)
    for offset in range(1, n):
        if game_colors[(start + offset) % n] == victim:
            return offset
    raise ValueError(
        f"victim {victim} is not an opponent of {actor} in seating order {game_colors}"
    )


def _absolute_victim_from_slot(actor: Color, slot: int, game_colors: Tuple[Color, ...]) -> Color:
    if slot < 1 or slot >= len(game_colors):
        raise ValueError(f"invalid opponent slot {slot} for {len(game_colors)} players")
    return game_colors[(game_colors.index(actor) + slot) % len(game_colors)]


def _action_to_key(action: Action, game_colors: Tuple[Color, ...]) -> tuple:
    """Map a Catanatron Action to a key in the catanrl flat action array."""
    if action.action_type != ActionType.MOVE_ROBBER:
        return (action.action_type, action.value)
    coord, victim = action.value
    if victim is None:
        return (ActionType.MOVE_ROBBER, (coord, None))
    rel = _relative_opponent_slot(action.color, victim, game_colors)
    return (ActionType.MOVE_ROBBER, (coord, rel))


@lru_cache(maxsize=None)
def get_action_array(
    num_players: int,
    map_type: MapType,
) -> Tuple[tuple[ActionType, object | None], ...]:
    """
    Instead of the regular Catanatron env action array, we make the robber actions use relative
    player indices like we do in feature extractors instead of absolute victim colors.
    """
    if not 1 <= num_players <= len(PLAYER_COLOR_ORDER):
        raise ValueError(f"num_players must be in [1, {len(PLAYER_COLOR_ORDER)}], got {num_players}")
    catan_map = build_map(map_type)
    num_nodes = len(catan_map.land_nodes)
    robber_slots: list[object] = [None, *range(1, num_players)]

    actions_array = sorted(
        [
            (ActionType.ROLL, None),
            *[(ActionType.DISCARD_RESOURCE, resource) for resource in RESOURCES],
            *[
                (ActionType.BUILD_ROAD, tuple(sorted(edge)))
                for edge in get_edges(catan_map.land_nodes)
            ],
            *[(ActionType.BUILD_SETTLEMENT, node_id) for node_id in range(num_nodes)],
            *[(ActionType.BUILD_CITY, node_id) for node_id in range(num_nodes)],
            (ActionType.BUY_DEVELOPMENT_CARD, None),
            (ActionType.PLAY_KNIGHT_CARD, None),
            *[
                (ActionType.PLAY_YEAR_OF_PLENTY, (first_card, RESOURCES[j]))
                for i, first_card in enumerate(RESOURCES)
                for j in range(i, len(RESOURCES))
            ],
            *[
                (ActionType.PLAY_YEAR_OF_PLENTY, (first_card,))
                for first_card in RESOURCES
            ],
            (ActionType.PLAY_ROAD_BUILDING, None),
            *[(ActionType.PLAY_MONOPOLY, r) for r in RESOURCES],
            *[
                (ActionType.MOVE_ROBBER, (coordinates, slot))
                for coordinates in catan_map.land_tiles.keys()
                for slot in robber_slots
            ],
            *[
                (ActionType.MARITIME_TRADE, tuple(4 * [i] + [j]))
                for i in RESOURCES
                for j in RESOURCES
                if i != j
            ],
            *[
                (ActionType.MARITIME_TRADE, tuple(3 * [i] + [None, j]))  # type: ignore[misc]
                for i in RESOURCES
                for j in RESOURCES
                if i != j
            ],
            *[
                (ActionType.MARITIME_TRADE, tuple(2 * [i] + [None, None, j]))  # type: ignore[misc]
                for i in RESOURCES
                for j in RESOURCES
                if i != j
            ],
            (ActionType.END_TURN, None),
        ],
        key=lambda a: str(a),
    )
    return tuple(actions_array)


@lru_cache(maxsize=None)
def get_action_space_size(num_players: int, map_type: MapType) -> int:
    return len(get_action_array(num_players, map_type))


def to_action_space(
    action: Action,
    num_players: int,
    map_type: MapType,
    game_colors: Tuple[Color, ...],
) -> int:
    """Map a Catanatron action to a flat index (relative robber victim encoding)."""
    if len(game_colors) != num_players:
        raise ValueError(
            f"game_colors length {len(game_colors)} must match num_players {num_players}"
        )
    actions_array = get_action_array(num_players, map_type)
    key = _action_to_key(action, game_colors)
    return actions_array.index(key)


def from_action_space(
    action_int: int,
    color: Color,
    num_players: int,
    map_type: MapType,
    game_colors: Tuple[Color, ...],
) -> Action:
    """Map flat index to Action; MOVE_ROBBER values use absolute victim colors."""
    if len(game_colors) != num_players:
        raise ValueError(
            f"game_colors length {len(game_colors)} must match num_players {num_players}"
        )
    actions_array = get_action_array(num_players, map_type)
    action_type, value = actions_array[action_int]
    if action_type != ActionType.MOVE_ROBBER:
        return Action(color, action_type, value)
    coord, slot = value  # type: ignore[misc]
    if slot is None:
        return Action(color, action_type, (coord, None))
    if not isinstance(slot, int):
        raise TypeError(f"MOVE_ROBBER slot must be int or None, got {type(slot)}")
    victim = _absolute_victim_from_slot(color, slot, game_colors)
    return Action(color, action_type, (coord, victim))


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
