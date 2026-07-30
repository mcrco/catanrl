from __future__ import annotations

import random
from collections import Counter

import numpy as np
import pytest
from catanatron.game import Game
from catanatron.models.coordinate_system import Direction
from catanatron.models.enums import (
    DEVELOPMENT_CARDS,
    RESOURCES,
    ActionRecord,
    ActionType,
)
from catanatron.models.map import (
    BASE_MAP_TEMPLATE,
    MINI_MAP_TEMPLATE,
    CatanMap,
    build_map,
    initialize_tiles,
)
from catanatron.models.player import Color, RandomPlayer
from catanatron.models.tiles import LandTile, Port, Water
from catanatron.players.value import ValueFunctionPlayer, base_fn

from catanrl.envs.cppanatron import (
    NativeGame,
    find_cppanatron_library,
    full_native_features,
)
from catanrl.features.catanatron_utils import full_game_to_features
from catanrl.utils.catanatron_action_space import from_action_space, to_action_space
from catanrl.utils.catanatron_game import force_player_order


@pytest.fixture(scope="module", autouse=True)
def _require_native_library():
    try:
        find_cppanatron_library()
    except FileNotFoundError as exc:
        pytest.skip(str(exc))


def test_native_binding_runs_complete_initial_setup():
    with NativeGame(num_players=2, map_type="TOURNAMENT", seed=42) as game:
        assert game.action_space_size == 313
        assert game.current_player == 0
        assert game.current_prompt == 0
        assert game.valid_action_mask().sum() == 54

        played = []
        while game.current_prompt in (0, 1):
            action = int(np.flatnonzero(game.valid_action_mask())[0])
            played.append(action)
            game.step(action)

        assert played == [126, 54, 128, 59, 130, 61, 132, 64]
        assert game.current_player == 0
        assert game.current_prompt == 2
        assert game.num_turns == 2
        assert game.player_state(0).resources == (0, 1, 0, 1, 1)
        assert game.player_state(1).resources == (1, 0, 1, 0, 1)


def test_native_binding_reports_invalid_actions():
    with NativeGame(num_players=2, map_type="BASE", seed=3) as game:
        with pytest.raises(RuntimeError, match="not currently playable"):
            game.step(game.action_space_size - 1)


@pytest.mark.parametrize("map_type", ["MINI", "BASE"])
@pytest.mark.parametrize("seed", [0, 1, 42])
def test_native_random_maps_match_python_template_contract(map_type, seed):
    random.seed(seed)
    python_map = build_map(map_type)

    with NativeGame(num_players=2, map_type=map_type, seed=seed) as native_game:
        native_tiles = {
            (tile[0], tile[1], tile[2]): tile[3:] for tile in native_game.tiles()
        }
        assert set(native_tiles) == set(python_map.tiles)

        for coordinate, python_tile in python_map.tiles.items():
            (
                tile_id,
                kind,
                _resource,
                _number,
                direction,
                nodes,
            ) = native_tiles[coordinate]
            assert nodes == tuple(python_tile.nodes.values())
            if isinstance(python_tile, LandTile):
                assert kind == 0
                assert tile_id == python_tile.id
                assert direction == -1
            elif isinstance(python_tile, Water):
                assert kind == 1
                assert tile_id == -1
                assert direction == -1
            elif isinstance(python_tile, Port):
                assert kind == 2
                assert tile_id == python_tile.id
                assert direction == list(Direction).index(python_tile.direction)
            else:  # pragma: no cover - guards an upstream tile-type addition
                raise AssertionError(f"Unknown Python tile type: {type(python_tile)}")

        python_land_resources = Counter(
            None if tile.resource is None else RESOURCES.index(tile.resource)
            for tile in python_map.land_tiles.values()
        )
        native_land_resources = Counter(
            resource
            for _id, kind, resource, _number, _direction, _nodes in native_tiles.values()
            if kind == 0
        )
        assert native_land_resources == Counter(
            {-1 if resource is None else resource: count
             for resource, count in python_land_resources.items()}
        )

        python_numbers = Counter(
            tile.number for tile in python_map.land_tiles.values()
        )
        native_numbers = Counter(
            number
            for _id, kind, _resource, number, _direction, _nodes in native_tiles.values()
            if kind == 0
        )
        assert native_numbers == Counter(
            {-1 if number is None else number: count
             for number, count in python_numbers.items()}
        )

        python_ports = Counter(
            None if tile.resource is None else RESOURCES.index(tile.resource)
            for tile in python_map.tiles.values()
            if isinstance(tile, Port)
        )
        native_ports = Counter(
            resource
            for _id, kind, resource, _number, _direction, _nodes in native_tiles.values()
            if kind == 2
        )
        assert native_ports == Counter(
            {-1 if resource is None else resource: count
             for resource, count in python_ports.items()}
        )


@pytest.mark.parametrize("map_type", ["MINI", "BASE", "TOURNAMENT"])
def test_native_random_legal_rollouts_do_not_dead_end(map_type):
    rng = np.random.default_rng(173)
    for seed in range(4):
        with NativeGame(num_players=2, map_type=map_type, seed=seed) as game:
            for _ in range(4_000):
                if game.winner is not None or game.num_turns >= 1_000:
                    break
                valid = np.flatnonzero(game.valid_action_mask())
                assert valid.size > 0
                game.step(int(rng.choice(valid)))


def _python_player_tuple(game: Game, color: Color):
    index = game.state.color_to_index[color]
    key = f"P{index}"
    state = game.state.player_state
    return (
        state[f"{key}_VICTORY_POINTS"],
        state[f"{key}_ACTUAL_VICTORY_POINTS"],
        state[f"{key}_ROADS_AVAILABLE"],
        state[f"{key}_SETTLEMENTS_AVAILABLE"],
        state[f"{key}_CITIES_AVAILABLE"],
        bool(state[f"{key}_HAS_ROAD"]),
        bool(state[f"{key}_HAS_ARMY"]),
        bool(state[f"{key}_HAS_ROLLED"]),
        bool(state[f"{key}_HAS_PLAYED_DEVELOPMENT_CARD_IN_TURN"]),
        state[f"{key}_LONGEST_ROAD_LENGTH"],
        tuple(state[f"{key}_{resource}_IN_HAND"] for resource in RESOURCES),
        tuple(state[f"{key}_{card}_IN_HAND"] for card in DEVELOPMENT_CARDS),
        tuple(state[f"{key}_PLAYED_{card}"] for card in DEVELOPMENT_CARDS),
        tuple(
            bool(state[f"{key}_{card}_OWNED_AT_START"])
            for card in DEVELOPMENT_CARDS[:-1]
        ),
    )


def _native_player_tuple(game: NativeGame, index: int):
    state = game.player_state(index)
    return (
        state.victory_points,
        state.actual_victory_points,
        state.roads_available,
        state.settlements_available,
        state.cities_available,
        state.has_road,
        state.has_army,
        state.has_rolled,
        state.has_played_development_card_in_turn,
        state.longest_road_length,
        state.resources,
        state.development_cards,
        state.played_development_cards,
        state.development_card_owned_at_start,
    )


def _python_map_matching_native(
    native_game: NativeGame,
    map_type: str,
) -> CatanMap:
    if map_type == "TOURNAMENT":
        return build_map("TOURNAMENT")

    template = MINI_MAP_TEMPLATE if map_type == "MINI" else BASE_MAP_TEMPLATE
    native_tiles = {
        (tile[0], tile[1], tile[2]): tile[3:] for tile in native_game.tiles()
    }
    placed_land_resources = []
    placed_port_resources = []
    for coordinate, tile_type in template.topology.items():
        _id, kind, resource, _number, _direction, _nodes = native_tiles[coordinate]
        if tile_type is LandTile:
            assert kind == 0
            placed_land_resources.append(
                None if resource < 0 else RESOURCES[resource]
            )
        elif isinstance(tile_type, tuple):
            assert kind == 2
            placed_port_resources.append(
                None if resource < 0 else RESOURCES[resource]
            )

    tiles = initialize_tiles(
        template,
        shuffled_port_resources_param=list(reversed(placed_port_resources)),
        shuffled_tile_resources_param=list(reversed(placed_land_resources)),
        number_placement="official_spiral",
    )
    return CatanMap.from_tiles(tiles)


@pytest.mark.parametrize("map_type", ["MINI", "BASE", "TOURNAMENT"])
@pytest.mark.parametrize("num_players", [2, 3, 4])
def test_replayed_python_and_native_transitions_match(map_type, num_players):
    colors = (Color.RED, Color.BLUE, Color.WHITE, Color.ORANGE)[:num_players]
    players = [RandomPlayer(color) for color in colors]
    rng = np.random.default_rng(991)

    with NativeGame(
        num_players=num_players,
        map_type=map_type,
        seed=777,
        vps_to_win=6,
    ) as native_game:
        python_game = Game(
            players,
            seed=777,
            catan_map=_python_map_matching_native(native_game, map_type),
            vps_to_win=6,
        )
        force_player_order(python_game, players)
        for step in range(2_000):
            python_mask = np.zeros(native_game.action_space_size, dtype=np.bool_)
            for action in python_game.playable_actions:
                python_mask[
                    to_action_space(
                        action,
                        num_players,
                        map_type,
                        tuple(python_game.state.colors),
                    )
                ] = True
            native_mask = native_game.valid_action_mask()
            np.testing.assert_array_equal(
                native_mask,
                python_mask,
                err_msg=f"legal-action mismatch before replay step {step}",
            )
            assert native_game.current_player == python_game.state.current_player_index
            assert native_game.current_prompt == list(type(python_game.state.current_prompt)).index(
                python_game.state.current_prompt
            )
            assert native_game.num_turns == python_game.state.num_turns
            for index, color in enumerate(colors):
                assert _native_player_tuple(native_game, index) == _python_player_tuple(
                    python_game, color
                )
            np.testing.assert_allclose(
                full_native_features(native_game, map_type, base_player=0),
                full_game_to_features(
                    python_game,
                    num_players,
                    map_type,
                    base_color=Color.RED,
                ),
                rtol=0,
                atol=1e-7,
                err_msg=f"feature mismatch before replay step {step}",
            )
            for index, color in enumerate(colors):
                assert native_game.value_score(index) == pytest.approx(
                    base_fn()(python_game, color),
                    rel=1e-14,
                    abs=1e-6,
                )
            stochastic_types = {
                ActionType.ROLL,
                ActionType.BUY_DEVELOPMENT_CARD,
                ActionType.MOVE_ROBBER,
            }
            if not any(
                action.action_type in stochastic_types
                for action in python_game.playable_actions
            ):
                python_expert = ValueFunctionPlayer(python_game.state.current_color())
                expert_action = python_expert.decide(
                    python_game, python_game.playable_actions
                )
                python_expert_index = to_action_space(
                    expert_action,
                    num_players,
                    map_type,
                    tuple(python_game.state.colors),
                )
                native_expert_index = native_game.value_action()
                if native_expert_index != python_expert_index:
                    native_expert_action = from_action_space(
                        native_expert_index,
                        python_game.state.current_color(),
                        num_players,
                        map_type,
                        tuple(python_game.state.colors),
                        python_game.playable_actions,
                    )
                    python_candidate = python_game.copy()
                    python_candidate.execute(expert_action)
                    native_candidate = python_game.copy()
                    native_candidate.execute(native_expert_action)
                    assert base_fn()(native_candidate, native_expert_action.color) == pytest.approx(
                        base_fn()(python_candidate, expert_action.color),
                        rel=1e-14,
                        abs=1e-6,
                    )

            if python_game.winning_color() is not None or python_game.state.num_turns >= 1_000:
                break

            valid = np.flatnonzero(python_mask)
            action_index = int(rng.choice(valid))
            action = from_action_space(
                action_index,
                python_game.state.current_color(),
                num_players,
                map_type,
                tuple(python_game.state.colors),
                python_game.playable_actions,
            )

            result = None
            dice = None
            development_card = None
            stolen_resource = None
            if action.action_type == ActionType.ROLL:
                dice = (int(rng.integers(1, 7)), int(rng.integers(1, 7)))
                result = dice
            elif action.action_type == ActionType.BUY_DEVELOPMENT_CARD:
                result = python_game.state.development_listdeck[0]
                development_card = DEVELOPMENT_CARDS.index(result)
            elif action.action_type == ActionType.MOVE_ROBBER and action.value[1] is not None:
                victim = action.value[1]
                victim_index = python_game.state.color_to_index[victim]
                victim_key = f"P{victim_index}"
                result = next(
                    resource
                    for resource in RESOURCES
                    if python_game.state.player_state[
                        f"{victim_key}_{resource}_IN_HAND"
                    ]
                    > 0
                )
                stolen_resource = RESOURCES.index(result)

            python_game.execute(
                action,
                action_record=ActionRecord(action=action, result=result),
            )
            native_game.step(
                action_index,
                dice=dice,
                development_card=development_card,
                stolen_resource=stolen_resource,
            )
