from __future__ import annotations

from collections import Counter

import numpy as np
import pytest
from catanatron.game import Game
from catanatron.models.coordinate_system import Direction
from catanatron.models.enums import (
    DEVELOPMENT_CARDS,
    RESOURCES,
    SETTLEMENT,
    ActionRecord,
    ActionType,
)
from catanatron.models.map import MINI_MAP_TEMPLATE, build_map
from catanatron.models.player import Color, RandomPlayer
from catanatron.models.tiles import LandTile, Port, Water

from catanrl.envs.cudanatron import NativeGame, find_cudanatron_library, full_native_features
from catanrl.envs.cudanatron.map_bridge import build_catan_map_from_native_game
from catanrl.features.catanatron_utils import full_game_to_features
from catanrl.utils.catanatron_action_space import from_action_space, get_action_array, to_action_space
from catanrl.utils.catanatron_game import force_player_order


@pytest.fixture(scope="module", autouse=True)
def _require_native_library():
    try:
        find_cudanatron_library()
    except FileNotFoundError as exc:
        pytest.skip(str(exc))


def test_native_binding_runs_complete_initial_setup():
    with NativeGame(num_players=2, map_type="TOURNAMENT", seed=42) as game:
        assert game.action_space_size == 313
        assert game.current_player == 0
        assert game.current_prompt == 0
        assert game.valid_action_mask().sum() == 54

        while game.current_prompt in (0, 1):
            action = int(np.flatnonzero(game.valid_action_mask())[0])
            game.step(action)

        assert game.current_player == 0
        assert game.current_prompt == 2
        assert game.num_turns == 2


@pytest.mark.parametrize("map_type,num_players", [("MINI", 2), ("BASE", 2), ("TOURNAMENT", 4)])
def test_native_action_space_keys_match_python(map_type, num_players):
    python_keys = [str(action) for action in get_action_array(num_players, map_type)]
    with NativeGame(num_players=num_players, map_type=map_type, seed=0) as game:
        assert game.action_space_size == len(python_keys)
        native_keys = [game.action_key(index) for index in range(game.action_space_size)]
    assert native_keys == python_keys


def test_native_binding_reports_invalid_actions():
    with NativeGame(num_players=2, map_type="BASE", seed=3) as game:
        with pytest.raises(RuntimeError, match="not currently playable"):
            game.step(game.action_space_size - 1)


def test_native_binding_supports_independent_map_and_game_seeds():
    with (
        NativeGame(num_players=2, map_type="BASE", seed=11, map_seed=101) as first,
        NativeGame(num_players=2, map_type="BASE", seed=22, map_seed=101) as same_map,
        NativeGame(num_players=2, map_type="BASE", seed=11, map_seed=202) as other_map,
    ):
        assert first.tiles() == same_map.tiles()
        assert first.tiles() != other_map.tiles()

        original_tiles = first.tiles()
        first.reset(seed=33, map_seed=101)
        assert first.tiles() == original_tiles


def test_native_binding_selects_number_placement():
    with (
        NativeGame(
            num_players=2,
            map_type="BASE",
            seed=29,
            map_seed=17,
            number_placement="official_spiral",
        ) as official,
        NativeGame(
            num_players=2,
            map_type="BASE",
            seed=29,
            map_seed=17,
            number_placement="random",
        ) as randomized,
    ):
        official_tiles = {tile[:3]: (tile[5], tile[6]) for tile in official.tiles() if tile[4] == 0}
        randomized_tiles = {
            tile[:3]: (tile[5], tile[6]) for tile in randomized.tiles() if tile[4] == 0
        }
        assert {
            coordinate: resource for coordinate, (resource, _number) in official_tiles.items()
        } == {coordinate: resource for coordinate, (resource, _number) in randomized_tiles.items()}
        assert {
            coordinate: number for coordinate, (_resource, number) in official_tiles.items()
        } != {coordinate: number for coordinate, (_resource, number) in randomized_tiles.items()}


@pytest.mark.parametrize("map_type", ["MINI", "BASE"])
@pytest.mark.parametrize("seed", [0, 1, 42])
def test_native_random_maps_match_python_template_contract(map_type, seed):
    python_map = build_map(map_type, number_placement="random")
    with NativeGame(num_players=2, map_type=map_type, seed=seed) as native_game:
        native_tiles = {(tile[0], tile[1], tile[2]): tile[3:] for tile in native_game.tiles()}
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
            else:  # pragma: no cover
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
            {
                -1 if resource is None else resource: count
                for resource, count in python_land_resources.items()
            }
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
        tuple(bool(state[f"{key}_{card}_OWNED_AT_START"]) for card in DEVELOPMENT_CARDS[:-1]),
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


@pytest.mark.parametrize("map_type", ["MINI", "BASE", "TOURNAMENT"])
@pytest.mark.parametrize("num_players", [2, 3, 4])
def test_replayed_python_and_native_transitions_match(map_type, num_players):
    colors = (Color.RED, Color.BLUE, Color.WHITE, Color.ORANGE)[:num_players]
    players = [RandomPlayer(color) for color in colors]
    rng = np.random.default_rng(991)

    with NativeGame(num_players=num_players, map_type=map_type, seed=777, vps_to_win=6) as native_game:
        python_game = Game(
            players,
            seed=777,
            catan_map=build_catan_map_from_native_game(native_game, map_type),
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
                native_player = _native_player_tuple(native_game, index)
                python_player = _python_player_tuple(python_game, color)
                assert native_player == python_player, (
                    f"player-state mismatch for player {index} before replay step {step}; "
                    f"native_roads={native_game.roads()}; "
                    f"python_roads={python_game.state.board.roads}; "
                    f"buildings={python_game.state.board.buildings}"
                )
            python_winner = python_game.winning_color()
            assert native_game.winner == (
                None if python_winner is None else colors.index(python_winner)
            )
            assert native_game.resource_bank == tuple(python_game.state.resource_freqdeck)
            assert native_game.development_cards_remaining == len(
                python_game.state.development_listdeck
            )
            flags = native_game.flags
            assert flags[:6] == (
                python_game.state.is_initial_build_phase,
                python_game.state.is_discarding,
                python_game.state.current_prompt.name == "MOVE_ROBBER",
                python_game.state.is_road_building,
                python_game.state.current_player_index,
                python_game.state.current_turn_index,
            )
            assert flags[6] == sum(
                record.action.action_type == ActionType.END_TURN
                for record in python_game.state.action_records
            )
            python_buildings = {
                (
                    node,
                    colors.index(color),
                    0 if building == SETTLEMENT else 1,
                )
                for node, (color, building) in python_game.state.board.buildings.items()
            }
            assert set(native_game.buildings()) == python_buildings
            python_roads = {
                (min(edge), max(edge), colors.index(color))
                for edge, color in python_game.state.board.roads.items()
            }
            assert set(native_game.roads()) == python_roads
            np.testing.assert_allclose(
                native_game.observation(base_player=0),
                full_native_features(native_game, map_type, base_player=0),
                rtol=0,
                atol=1e-6,
                err_msg=f"native observation writer mismatch before replay step {step}",
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
                result = python_game.state.development_listdeck[-1]
                development_card = DEVELOPMENT_CARDS.index(result)
            elif action.action_type == ActionType.MOVE_ROBBER and action.value[1] is not None:
                victim = action.value[1]
                victim_index = python_game.state.color_to_index[victim]
                victim_key = f"P{victim_index}"
                result = next(
                    resource
                    for resource in RESOURCES
                    if python_game.state.player_state[f"{victim_key}_{resource}_IN_HAND"] > 0
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


def test_search_pool_selects_and_evaluates_leaf_batches():
    from catanrl.envs.cudanatron import NativeSearchPool

    games = [
        NativeGame(num_players=2, map_type="MINI", seed=seed)
        for seed in (1, 2, 3, 4)
    ]
    try:
        for game in games:
            while game.current_prompt in (0, 1):
                game.step(int(np.flatnonzero(game.valid_action_mask())[0]))
        with NativeSearchPool(games, c_puct=1.5, seed=11) as pool:
            policy = np.zeros((pool.size, pool.action_space_size), dtype=np.float32)
            pool.initialize_roots(policy)
            pool.set_root_wdls(np.tile(np.array([0.4, 0.2, 0.4]), (pool.size, 1)))
            pool.add_simulations_all(16)
            evaluations = 0
            while pool.remaining_simulations > 0:
                observations, players, tokens = pool.select_leaves(capacity=8)
                if tokens.size == 0:
                    break
                assert observations.shape[0] == tokens.size
                assert np.all(players >= 0)
                leaf_policy = np.zeros((tokens.size, pool.action_space_size), dtype=np.float32)
                leaf_wdl = np.tile(np.array([0.5, 0.0, 0.5]), (tokens.size, 1))
                pool.evaluate_leaves(tokens, leaf_policy, leaf_wdl)
                evaluations += int(tokens.size)
                assert evaluations <= 64
            assert pool.remaining_simulations == 0
            visits = pool.root_visits(0)
            assert int(visits.sum()) == 16
    finally:
        for game in games:
            game.close()


def _batch_observation_dtype(action_space_size: int, observation_size: int) -> np.dtype:
    pad = (4 - (action_space_size % 4)) % 4
    fields = [("action_mask", np.uint8, (action_space_size,))]
    if pad:
        fields.append(("_pad", np.uint8, (pad,)))
    fields.append(("observation", np.float32, (observation_size,)))
    return np.dtype(fields)


def test_cuda_batch_reset_matches_native_game_observation():
    from catanrl.envs.cudanatron import NativeGameBatch

    num_envs = 4
    num_players = 2
    with NativeGame(num_players=num_players, map_type="MINI", seed=21, map_seed=21) as probe:
        action_n = probe.action_space_size
        obs_n = probe.observation_size()
        expected = probe.observation(base_player=0)

    dtype = _batch_observation_dtype(action_n, obs_n)
    rows = num_envs * num_players
    observations = np.zeros((rows, dtype.itemsize), dtype=np.uint8)
    actions = np.zeros(rows, dtype=np.int32)
    rewards = np.zeros(rows, dtype=np.float32)
    terminals = np.zeros(rows, dtype=np.bool_)
    truncations = np.zeros(rows, dtype=np.bool_)
    masks = np.ones(rows, dtype=np.bool_)
    batch = NativeGameBatch(
        num_envs=num_envs,
        num_players=num_players,
        map_type="MINI",
        discard_limit=7,
        vps_to_win=10,
        reward_function="shaped",
        turns_limit=1000,
        observations=observations,
        obs_dtype=dtype,
        actions=actions,
        rewards=rewards,
        terminals=terminals,
        truncations=truncations,
        masks=masks,
    )
    try:
        seeds = np.full(num_envs, 21, dtype=np.uint64)
        batch.reset_all(seeds, seeds)
        offset = dtype.fields["observation"][1]
        got = np.frombuffer(observations[0], dtype=np.float32, offset=offset, count=obs_n)
        np.testing.assert_allclose(got, expected, rtol=0, atol=1e-6)

        for env_index in range(num_envs):
            row = env_index * num_players
            valid = np.flatnonzero(observations[row, :action_n])
            assert valid.size > 0
            actions[row] = int(valid[0])
        batch.step()
        assert bool(masks[0])
    finally:
        batch.close()


