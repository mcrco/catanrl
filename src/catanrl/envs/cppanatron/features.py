from __future__ import annotations

import numpy as np
from catanatron.gym.board_tensor_features import (
    HEIGHT,
    WIDTH,
    get_node_and_edge_maps,
    get_tile_coordinate_map,
)

from catanrl.features.catanatron_utils import get_full_numeric_feature_names

from .binding import MapType, NativeGame, NativePlayerState

_RESOURCES = ("WOOD", "BRICK", "SHEEP", "WHEAT", "ORE")
_DEVELOPMENT_CARDS = (
    "KNIGHT",
    "YEAR_OF_PLENTY",
    "MONOPOLY",
    "ROAD_BUILDING",
    "VICTORY_POINT",
)
_DICE_PROBABILITY = {
    number: (6 - abs(7 - number)) / 36.0 for number in range(2, 13)
}


def _add_player_features(
    features: dict[str, float],
    relative_index: int,
    state: NativePlayerState,
) -> None:
    prefix = f"P{relative_index}"
    features[f"{prefix}_ACTUAL_VPS"] = state.actual_victory_points
    features[f"{prefix}_PUBLIC_VPS"] = state.victory_points
    features[f"{prefix}_HAS_ARMY"] = state.has_army
    features[f"{prefix}_HAS_ROAD"] = state.has_road
    features[f"{prefix}_ROADS_LEFT"] = state.roads_available
    features[f"{prefix}_SETTLEMENTS_LEFT"] = state.settlements_available
    features[f"{prefix}_CITIES_LEFT"] = state.cities_available
    features[f"{prefix}_HAS_ROLLED"] = state.has_rolled
    features[f"{prefix}_LONGEST_ROAD_LENGTH"] = state.longest_road_length
    features[f"{prefix}_HAS_PLAYED_DEVELOPMENT_CARD_IN_TURN"] = (
        state.has_played_development_card_in_turn
    )
    features[f"{prefix}_NUM_RESOURCES_IN_HAND"] = sum(state.resources)
    features[f"{prefix}_NUM_DEVS_IN_HAND"] = sum(state.development_cards)
    features[f"{prefix}_TURNS_SINCE_LAST_KNIGHT"] = state.turns_since_last_knight
    features[f"{prefix}_TURNS_SINCE_LAST_DEV_BOUGHT"] = (
        state.turns_since_last_development_card_bought
    )

    for index, resource in enumerate(_RESOURCES):
        features[f"{prefix}_{resource}_IN_HAND"] = state.resources[index]
    for index, card in enumerate(_DEVELOPMENT_CARDS):
        features[f"{prefix}_{card}_IN_HAND"] = state.development_cards[index]
        if card == "VICTORY_POINT":
            continue
        features[f"{prefix}_{card}_PLAYED"] = state.played_development_cards[index]
        features[f"{prefix}_{card}_PLAYABLE"] = (
            state.development_cards[index] > 0
            and state.development_card_owned_at_start[index]
            and not state.has_played_development_card_in_turn
        )


def native_numeric_features(
    game: NativeGame,
    map_type: MapType,
    base_player: int,
) -> np.ndarray:
    if not 0 <= base_player < game.num_players:
        raise ValueError(f"base_player {base_player} is out of range")

    features: dict[str, float] = {}
    bank = game.resource_bank
    for index, resource in enumerate(_RESOURCES):
        features[f"BANK_{resource}"] = bank[index]
    features["BANK_DEV_CARDS"] = game.development_cards_remaining

    (
        is_initial_build_phase,
        is_discarding,
        is_moving_robber,
        _is_road_building,
        _current_player,
        _current_turn,
        _completed_turns,
    ) = game.flags
    features["IS_INITIAL_BUILD_PHASE"] = is_initial_build_phase
    features["IS_DISCARDING"] = is_discarding
    features["IS_MOVING_ROBBER"] = is_moving_robber
    features["TURN_NUMBER"] = game.num_turns

    for relative_index in range(game.num_players):
        absolute_index = (base_player + relative_index) % game.num_players
        _add_player_features(
            features,
            relative_index,
            game.player_state(absolute_index),
        )

    names = get_full_numeric_feature_names(game.num_players, map_type)
    return np.asarray([float(features.get(name, 0.0)) for name in names], dtype=np.float32)


def native_board_tensor(game: NativeGame, base_player: int) -> np.ndarray:
    channels = 2 * game.num_players + 5 + 1 + 6
    planes = np.zeros((channels, WIDTH, HEIGHT), dtype=np.float32)
    node_map, edge_map = get_node_and_edge_maps()

    for node, color, building in game.buildings():
        relative_player = (color - base_player) % game.num_players
        x, y = node_map[node]
        planes[2 * relative_player, x, y] = 1.0 if building == 0 else 2.0
    for a, b, color in game.roads():
        relative_player = (color - base_player) % game.num_players
        x, y = edge_map[(a, b)]
        planes[2 * relative_player + 1, x, y] = 1.0

    tile_map = get_tile_coordinate_map()
    tiles = game.tiles()
    for x_coord, y_coord, z_coord, _id, kind, resource, number, _direction, _nodes in tiles:
        if kind != 0 or resource < 0:
            continue
        y, x = tile_map[(x_coord, y_coord, z_coord)]
        probability = _DICE_PROBABILITY[number]
        channel = 2 * game.num_players + resource
        for x_delta in (0, 2, 4):
            for y_delta in (0, 2):
                planes[channel, x + x_delta, y + y_delta] += probability

    robber_y, robber_x = tile_map[game.robber_coordinate]
    robber_channel = 2 * game.num_players + 5
    for x_delta in (0, 2, 4):
        for y_delta in (0, 2):
            planes[robber_channel, robber_x + x_delta, robber_y + y_delta] = 1.0

    for _x, _y, _z, _id, kind, resource, _number, direction, nodes in tiles:
        if kind != 2:
            continue
        port_node_indices = {
            0: (2, 1),
            1: (3, 2),
            2: (4, 3),
            3: (5, 4),
            4: (0, 5),
            5: (1, 0),
        }[direction]
        port_channel_delta = 5 if resource < 0 else resource
        channel = 2 * game.num_players + 6 + port_channel_delta
        for node_index in port_node_indices:
            x, y = node_map[nodes[node_index]]
            planes[channel, x, y] = 1.0

    return np.transpose(planes, (1, 2, 0))


def full_native_features(
    game: NativeGame,
    map_type: MapType,
    base_player: int,
) -> np.ndarray:
    numeric = native_numeric_features(game, map_type, base_player)
    board = native_board_tensor(game, base_player).reshape(-1)
    return np.concatenate((numeric, board), dtype=np.float32)
