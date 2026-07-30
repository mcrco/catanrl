from __future__ import annotations

import numpy as np
import pytest

from catanrl.envs.cppanatron import (
    NativeGame,
    NativeVectorCppanatronPufferEnv,
    find_cppanatron_library,
    full_native_features,
    make_cppanatron_native_marl_vectorized_envs,
)
from catanrl.envs.cppanatron.puffer_env import _native_production_sum


@pytest.fixture(scope="module", autouse=True)
def _require_native_library():
    try:
        find_cppanatron_library()
    except FileNotFoundError as exc:
        pytest.skip(str(exc))


def _make_env(
    *,
    num_envs: int,
    num_players: int,
    map_type: str,
) -> NativeVectorCppanatronPufferEnv:
    return NativeVectorCppanatronPufferEnv(
        {
            "num_envs": num_envs,
            "num_players": num_players,
            "map_type": map_type,
            "vps_to_win": 6,
            "discard_limit": 7,
            "shared_critic": True,
            "reward_function": "shaped",
        }
    )


@pytest.mark.parametrize("map_type", ["MINI", "BASE", "TOURNAMENT"])
@pytest.mark.parametrize("num_players", [2, 3, 4])
def test_native_vector_reset_matches_scalar_features_and_masks(
    map_type: str,
    num_players: int,
):
    env = _make_env(
        num_envs=1,
        num_players=num_players,
        map_type=map_type,
    )
    game = NativeGame(
        num_players,
        map_type,
        seed=67_890,
        map_seed=12_345,
        number_placement="random",
        vps_to_win=6,
    )
    try:
        env._batch.reset_all(
            np.asarray([12_345], dtype=np.uint64),
            np.asarray([67_890], dtype=np.uint64),
        )
        structured = env.observations.view(env.obs_dtype).reshape(-1)
        for player in range(num_players):
            np.testing.assert_array_equal(
                structured["observation"][player],
                full_native_features(game, map_type, player),
            )
            expected_mask = np.zeros(env.action_space_size, dtype=np.int8)
            if player == game.current_player:
                expected_mask = game.valid_action_mask().astype(np.int8, copy=False)
            else:
                expected_mask[-1] = 1
            np.testing.assert_array_equal(
                structured["action_mask"][player],
                expected_mask,
            )
    finally:
        game.close()
        env.close()


def test_native_vector_steps_match_scalar_state_and_shaped_rewards():
    num_envs = 2
    num_players = 2
    map_seeds = np.asarray([101, 202], dtype=np.uint64)
    game_seeds = np.asarray([303, 404], dtype=np.uint64)
    env = _make_env(
        num_envs=num_envs,
        num_players=num_players,
        map_type="MINI",
    )
    games = [
        NativeGame(
            num_players,
            "MINI",
            seed=int(game_seeds[index]),
            map_seed=int(map_seeds[index]),
            number_placement="random",
            vps_to_win=6,
        )
        for index in range(num_envs)
    ]
    previous_vps = np.zeros((num_envs, num_players), dtype=np.int32)
    previous_production = np.zeros((num_envs, num_players), dtype=np.float64)
    try:
        env._batch.reset_all(map_seeds, game_seeds)
        env._initialized = True
        for _ in range(64):
            actions = np.full(env.num_agents, env.action_space_size - 1, dtype=np.int32)
            for env_index, game in enumerate(games):
                action = int(np.flatnonzero(game.valid_action_mask())[0])
                actions[env_index * num_players + game.current_player] = action
                game.step(action)

            _, rewards, terminals, truncations, _ = env.step(actions)
            structured = env.observations.view(env.obs_dtype).reshape(-1)
            for env_index, game in enumerate(games):
                for player in range(num_players):
                    row = env_index * num_players + player
                    state = game.player_state(player)
                    production = _native_production_sum(game, player)
                    expected_reward = 0.01 * (
                        state.actual_victory_points - previous_vps[env_index, player]
                    ) / 6 + 0.0025 * (production - previous_production[env_index, player])
                    assert rewards[row] == pytest.approx(expected_reward, abs=1e-7)
                    previous_vps[env_index, player] = state.actual_victory_points
                    previous_production[env_index, player] = production
                    assert not terminals[row]
                    assert not truncations[row]
                    np.testing.assert_array_equal(
                        structured["observation"][row],
                        full_native_features(game, "MINI", player),
                    )
    finally:
        for game in games:
            game.close()
        env.close()


def test_native_vector_factory_exposes_internal_environment_layout():
    env = make_cppanatron_native_marl_vectorized_envs(
        num_players=2,
        map_type="MINI",
        vps_to_win=6,
        discard_limit=7,
        shared_critic=True,
        reward_function="win",
        num_envs=3,
    )
    try:
        observations, infos = env.reset(seed=19)
        assert observations.shape[0] == 6
        assert env.agents_per_env == [2, 2, 2]
        assert len(infos) == 6
    finally:
        env.close()
