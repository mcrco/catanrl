"""Transition, edge-case, and integration tests for native Puffer Catan environments."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
from pufferlib.emulation import nativize

from catanatron.game import TURNS_LIMIT
from catanatron.models.player import Color
from catanatron.state_functions import get_state_index

from catanrl.envs.puffer.common import compute_single_agent_dims, create_expert, create_opponents
from catanrl.envs.puffer.multi_agent_env import (
    MultiAgentCatanatronEnvConfig,
    ParallelCatanatronPufferEnv,
    compute_multiagent_input_dim,
    get_valid_actions,
    make_vectorized_envs as make_multi_puffer_vectorized_envs,
)
from catanrl.envs.puffer.rewards import ShapedReward, WinReward
from catanrl.envs.puffer.rollout_utils import decode_puffer_batch
from catanrl.envs.puffer.single_agent_env import (
    SingleAgentCatanatronPufferEnv,
    make_puffer_vectorized_envs,
)
from catanrl.features.catanatron_utils import get_observation_indices_from_full
from catanrl.utils.catanatron_action_space import get_end_turn_index, to_action_space


def _make_single_env(**overrides) -> SingleAgentCatanatronPufferEnv:
    config = {
        "map_type": "BASE",
        "vps_to_win": 10,
        "discard_limit": 7,
        "enemies": create_opponents(["R"]),
        "reward_function": WinReward(),
        "shared_critic": True,
        "expert_player": create_expert("F"),
        "nn_seat": "first",
    }
    config.update(overrides)
    return SingleAgentCatanatronPufferEnv(config=config)


def _make_multi_env(**overrides) -> ParallelCatanatronPufferEnv:
    config = MultiAgentCatanatronEnvConfig(
        num_players=2,
        map_type="BASE",
        vps_to_win=10,
        discard_limit=7,
        shared_critic=True,
        reward_function="win",
    )
    for key, value in overrides.items():
        setattr(config, key, value)
    return ParallelCatanatronPufferEnv(config=config)


def _decode_single(env: SingleAgentCatanatronPufferEnv, observation: np.ndarray | None = None) -> dict:
    obs = env.observations if observation is None else observation
    return nativize(obs[0], env.env_single_observation_space, env.obs_dtype)


def _decode_multi_row(env: ParallelCatanatronPufferEnv, row: int) -> dict:
    return nativize(env.observations[row], env.env_single_observation_space, env.obs_dtype)


def _mask_indices(structured_obs: dict) -> np.ndarray:
    return np.where(np.asarray(structured_obs["action_mask"], dtype=np.int8) > 0)[0]


def _assert_mask_matches_info(structured_obs: dict, info: dict) -> None:
    mask_indices = set(_mask_indices(structured_obs).tolist())
    valid_actions = set(np.asarray(info["valid_actions"], dtype=np.int64).tolist())
    assert mask_indices == valid_actions


def _first_invalid_action(valid_actions: list[int], action_space_size: int) -> int:
    valid = set(valid_actions)
    for candidate in range(action_space_size):
        if candidate not in valid:
            return candidate
    raise AssertionError("Could not find an invalid action index")


def _step_single_valid(env: SingleAgentCatanatronPufferEnv, info: dict):
    action = int(info["valid_actions"][0])
    return env.step(np.asarray([action], dtype=np.int32))


def _multi_actions_for_current(env: ParallelCatanatronPufferEnv, current_action: int) -> np.ndarray:
    end_turn = get_end_turn_index(env.num_players, env.map_type)
    actions = np.full(env.num_agents, end_turn, dtype=np.int32)
    current_color = env.game.state.current_color()
    current_agent_idx = env.possible_agents.index(env.color_to_agent_name[current_color])
    actions[current_agent_idx] = current_action
    return actions


# ---------------------------------------------------------------------------
# Single-agent lifecycle and invariants
# ---------------------------------------------------------------------------


def test_single_agent_reset_places_p0_on_decision_with_valid_mask():
    env = _make_single_env()
    try:
        _, infos = env.reset(seed=123)
        info = infos[0]
        structured = _decode_single(env)

        assert env.game is not None
        assert env.game.state.current_color() == env.p0.color
        assert len(info["valid_actions"]) >= 2
        assert int(info["expert_action"]) in info["valid_actions"]
        _assert_mask_matches_info(structured, info)
        assert env.rewards[0] == 0.0
        assert not bool(env.terminals[0])
        assert not bool(env.truncations[0])
        assert bool(env.masks[0])
        assert env._episode_done is False
    finally:
        env.close()


def test_single_agent_reset_is_reproducible_with_seed():
    first = _make_single_env()
    second = _make_single_env()
    try:
        first.reset(seed=999)
        second.reset(seed=999)
        assert first.game is not None and second.game is not None
        assert get_state_index(first.game.state) == get_state_index(second.game.state)
        assert first.get_valid_actions() == second.get_valid_actions()
    finally:
        first.close()
        second.close()


def test_single_agent_observation_fields_have_expected_shapes():
    env = _make_single_env()
    try:
        env.reset(seed=7)
        structured = _decode_single(env)
        assert structured["observation"]["numeric"].shape == (env.actor_numeric_dim,)
        assert structured["observation"]["board"].shape == env.board_tensor_shape
        assert structured["critic"].shape == (env.critic_vector_dim,)
        assert structured["action_mask"].shape == (env.action_space_size,)
    finally:
        env.close()


def test_single_agent_critic_is_full_state_actor_is_restricted():
    env = _make_single_env(actor_observation_level="private")
    try:
        env.reset(seed=11)
        structured = _decode_single(env)
        assert structured["critic"].shape[0] > structured["observation"]["numeric"].shape[0]
    finally:
        env.close()


# ---------------------------------------------------------------------------
# Single-agent transitions
# ---------------------------------------------------------------------------


def test_single_agent_valid_step_changes_state_and_returns_zero_win_reward():
    env = _make_single_env(reward_function=WinReward())
    try:
        _, infos = env.reset(seed=123)
        before_index = get_state_index(env.game.state)
        before_obs = _decode_single(env)["observation"]["numeric"].copy()
        _, rewards, terminals, truncations, infos = _step_single_valid(env, infos[0])
        structured = _decode_single(env)

        assert get_state_index(env.game.state) != before_index
        assert not np.array_equal(structured["observation"]["numeric"], before_obs)
        assert float(rewards[0]) == 0.0
        assert not bool(terminals[0])
        assert not bool(truncations[0])
        _assert_mask_matches_info(structured, infos[0])
    finally:
        env.close()


def test_single_agent_expert_action_always_matches_playable_actions():
    env = _make_single_env()
    try:
        _, infos = env.reset(seed=321)
        info = infos[0]
        for _ in range(8):
            expert_action = int(info["expert_action"])
            assert expert_action in info["valid_actions"]
            _, _, terminals, truncations, infos = _step_single_valid(env, info)
            if bool(terminals[0] or truncations[0]):
                break
            info = infos[0]
    finally:
        env.close()


def test_single_agent_shaped_reward_can_be_nonzero_after_settlement():
    env = _make_single_env(reward_function=ShapedReward())
    try:
        _, infos = env.reset(seed=123)
        _, rewards, _, _, _ = _step_single_valid(env, infos[0])
        assert float(rewards[0]) > 0.0
    finally:
        env.close()


def test_single_agent_truncation_when_turn_limit_reached():
    env = _make_single_env()
    try:
        _, infos = env.reset(seed=77)
        assert env.game is not None
        env.game.state.num_turns = TURNS_LIMIT
        action = int(infos[0]["valid_actions"][0])
        _, rewards, terminals, truncations, _ = env.step(np.asarray([action], dtype=np.int32))
        assert float(rewards[0]) == 0.0
        assert not bool(terminals[0])
        assert bool(truncations[0])
        assert bool(env.masks[0])
        assert env._episode_done is False
        assert env.game is not None
        assert env.game.state.num_turns < TURNS_LIMIT
    finally:
        env.close()


def test_single_agent_step_after_done_auto_resets():
    env = _make_single_env()
    try:
        env.reset(seed=123)
        env._episode_done = True
        observations, rewards, terminals, truncations, infos = env.step(
            np.asarray([0], dtype=np.int32)
        )
        assert env._episode_done is False
        assert env.game is not None
        assert env.game.state.current_color() == env.p0.color
        assert len(infos[0]["valid_actions"]) >= 1
        assert float(rewards[0]) == 0.0
        assert not bool(terminals[0])
        assert not bool(truncations[0])
        assert observations.shape == env.observations.shape
    finally:
        env.close()


def test_single_agent_nn_seat_ordering():
    first_env = _make_single_env(nn_seat="first")
    second_env = _make_single_env(nn_seat="second")
    try:
        first_env.reset(seed=123)
        second_env.reset(seed=123)

        assert first_env.game is not None
        assert second_env.game is not None
        assert first_env.game.state.colors[0] == first_env.p0.color
        assert second_env.game.state.colors[1] == second_env.p0.color
        assert second_env.game.state.colors[0] != second_env.p0.color
        assert second_env.game.state.current_color() == second_env.p0.color
    finally:
        first_env.close()
        second_env.close()


def test_single_agent_opponent_auto_play_advances_past_enemy_turns():
    env = _make_single_env(nn_seat="first")
    try:
        _, infos = env.reset(seed=202)
        assert env.game is not None
        info = infos[0]
        for _ in range(4):
            assert env.game.state.current_color() == env.p0.color
            _, _, terminals, truncations, infos = _step_single_valid(env, info)
            if bool(terminals[0] or truncations[0]):
                break
            info = infos[0]
    finally:
        env.close()


def test_single_agent_get_valid_actions_matches_playable_action_encoding():
    env = _make_single_env()
    try:
        _, infos = env.reset(seed=5)
        encoded = env.get_valid_actions()
        for action in env.game.playable_actions:
            idx = to_action_space(
                action,
                env.num_players,
                env.map_type,
                tuple(env.game.state.colors),
            )
            assert idx in encoded
    finally:
        env.close()


# ---------------------------------------------------------------------------
# Multi-agent lifecycle and invariants
# ---------------------------------------------------------------------------


def test_multi_agent_reset_exposes_all_agents_with_masks():
    env = _make_multi_env()
    try:
        _, infos = env.reset(seed=123)
        end_turn = get_end_turn_index(env.num_players, env.map_type)
        current_agent = env.color_to_agent_name[env.game.state.current_color()]

        for idx, agent in enumerate(env.possible_agents):
            structured = _decode_multi_row(env, idx)
            _assert_mask_matches_info(structured, infos[idx])
            valid = set(infos[idx]["valid_actions"].tolist())
            if agent == current_agent:
                assert len(valid) >= 2
            else:
                assert valid == {end_turn}
            assert "critic_observation" in infos[idx]
    finally:
        env.close()


def test_multi_agent_inactive_agent_action_is_ignored():
    env = _make_multi_env()
    try:
        _, infos = env.reset(seed=88)
        end_turn = get_end_turn_index(env.num_players, env.map_type)
        current_color = env.game.state.current_color()
        current_agent = env.color_to_agent_name[current_color]
        inactive_idx = 1 - env.possible_agents.index(current_agent)

        valid_current = int(infos[env.possible_agents.index(current_agent)]["valid_actions"][0])
        invalid_for_current = _first_invalid_action(
            infos[env.possible_agents.index(current_agent)]["valid_actions"].tolist(),
            env.action_space_size,
        )
        actions = np.full(env.num_agents, end_turn, dtype=np.int32)
        actions[inactive_idx] = invalid_for_current
        actions[env.possible_agents.index(current_agent)] = valid_current

        before_index = get_state_index(env.game.state)
        env.step(actions)
        assert get_state_index(env.game.state) != before_index
    finally:
        env.close()


def test_multi_agent_valid_step_updates_all_agent_rewards():
    env = _make_multi_env(reward_function="win")
    try:
        _, infos = env.reset(seed=44)
        current_idx = env.possible_agents.index(
            env.color_to_agent_name[env.game.state.current_color()]
        )
        action = int(infos[current_idx]["valid_actions"][0])
        _, rewards, terminals, truncations, _ = env.step(_multi_actions_for_current(env, action))
        assert rewards.shape == (env.num_agents,)
        assert np.all(rewards == 0.0)
        assert not np.any(terminals)
        assert not np.any(truncations)
    finally:
        env.close()


def test_multi_agent_shaped_reward_can_differ_between_players():
    env = _make_multi_env(reward_function="shaped")
    try:
        _, infos = env.reset(seed=123)
        current_idx = env.possible_agents.index(
            env.color_to_agent_name[env.game.state.current_color()]
        )
        action = int(infos[current_idx]["valid_actions"][0])
        _, rewards, _, _, _ = env.step(_multi_actions_for_current(env, action))
        assert np.any(rewards != 0.0)
    finally:
        env.close()


def test_multi_agent_truncation_zeros_masks_and_marks_truncated():
    env = _make_multi_env()
    try:
        env.reset(seed=66)
        env.game.state.num_turns = TURNS_LIMIT
        current_idx = env.possible_agents.index(
            env.color_to_agent_name[env.game.state.current_color()]
        )
        action = int(get_valid_actions(env._get_info(env.possible_agents[current_idx]))[0])
        _, _, terminals, truncations, _ = env.step(_multi_actions_for_current(env, int(action)))
        assert not np.any(terminals)
        assert np.all(truncations)
        assert np.all(~env.masks)
        assert env.done is True
    finally:
        env.close()


def test_multi_agent_step_after_done_auto_resets():
    env = _make_multi_env()
    try:
        env.reset(seed=5)
        env._all_done = True
        env.agents = []
        observations, rewards, terminals, truncations, infos = env.step(
            np.zeros(env.num_agents, dtype=np.int32)
        )
        assert env.done is False
        assert len(env.agents) == env.num_agents
        assert np.all(rewards == 0.0)
        assert not np.any(terminals)
        assert not np.any(truncations)
        assert observations.shape == env.observations.shape
        assert len(infos) == env.num_agents
    finally:
        env.close()


def test_multi_agent_three_players_have_expected_agent_count():
    env = _make_multi_env(num_players=3)
    try:
        _, infos = env.reset(seed=3)
        assert env.num_agents == 3
        assert len(infos) == 3
        end_turn = get_end_turn_index(3, "BASE")
        current_agent = env.color_to_agent_name[env.game.state.current_color()]
        for idx, agent in enumerate(env.possible_agents):
            valid = set(infos[idx]["valid_actions"].tolist())
            if agent == current_agent:
                assert len(valid) >= 2
            else:
                assert valid == {end_turn}
    finally:
        env.close()


def test_multi_agent_terminal_step_reports_winner_metadata():
    env = _make_multi_env(reward_function="win", vps_to_win=1)
    try:
        env.reset(seed=7)
        winner_color = Color.RED
        with patch.object(type(env.game), "winning_color", return_value=winner_color):
            end_turn = get_end_turn_index(env.num_players, env.map_type)
            current_color = env.game.state.current_color()
            current_idx = env.possible_agents.index(env.color_to_agent_name[current_color])
            action = int(env._get_info(env.possible_agents[current_idx])["valid_actions"][0])
            actions = _multi_actions_for_current(env, action)
            _, rewards, terminals, _, infos = env.step(actions)

        assert np.all(terminals)
        assert env.done is True
        assert infos[1]["winner_agent"] == "player_1"
        assert infos[1]["winner_agent_index"] == 1
        assert infos[1]["player_0_won"] is False
        assert float(rewards[1]) == 1.0
        assert float(rewards[0]) == -1.0
        assert np.all(~env.masks)
    finally:
        env.close()


def test_multi_agent_playable_actions_match_valid_actions_helper():
    env = _make_multi_env()
    try:
        _, infos = env.reset(seed=12)
        for idx, agent in enumerate(env.possible_agents):
            structured = _decode_multi_row(env, idx)
            helper_valid = get_valid_actions(infos[idx], structured)
            mask_valid = _mask_indices(structured)
            assert np.array_equal(helper_valid, mask_valid)
    finally:
        env.close()


def test_multi_agent_reset_is_reproducible_with_seed():
    first = _make_multi_env()
    second = _make_multi_env()
    try:
        first.reset(seed=808)
        second.reset(seed=808)
        assert get_state_index(first.game.state) == get_state_index(second.game.state)
        for idx in range(first.num_agents):
            assert np.array_equal(first.observations[idx], second.observations[idx])
    finally:
        first.close()
        second.close()


def test_single_agent_mini_map_reset_has_smaller_action_space():
    env = _make_single_env(map_type="MINI")
    try:
        base_env = _make_single_env(map_type="BASE")
        try:
            assert env.action_space_size < base_env.action_space_size
            _, infos = env.reset(seed=1)
            assert len(infos[0]["valid_actions"]) >= 2
        finally:
            base_env.close()
    finally:
        env.close()


def test_multi_agent_without_shared_critic_omits_critic_info():
    env = _make_multi_env(shared_critic=False)
    try:
        _, infos = env.reset(seed=2)
        for info in infos:
            assert "critic_observation" not in info
            structured = _decode_multi_row(env, 0)
            assert "critic" in structured
    finally:
        env.close()


def test_single_agent_win_reward_on_forced_terminal_state():
    env = _make_single_env(reward_function=WinReward())
    try:
        env.reset(seed=1)
        with patch.object(env.game, "winning_color", return_value=env.p0.color):
            reward = env.reward_function.reward(env, env.game, env.p0.color)
        assert reward == 1.0
    finally:
        env.close()


def test_single_agent_terminal_step_autoresets_for_next_transition():
    env = _make_single_env(reward_function=WinReward())
    try:
        _, infos = env.reset(seed=1)
        with patch.object(env.game, "winning_color", return_value=env.p0.color):
            action = int(infos[0]["valid_actions"][0])
            _, rewards, terminals, truncations, next_infos = env.step(
                np.asarray([action], dtype=np.int32)
            )

        assert float(rewards[0]) == 1.0
        assert bool(terminals[0])
        assert not bool(truncations[0])
        assert bool(env.masks[0])
        assert env._episode_done is False
        assert next_infos[0]["nn_won"] is True
        assert next_infos[0]["final_info"]["nn_won"] is True
        assert int(next_infos[0]["expert_action"]) in next_infos[0]["valid_actions"]
        assert env.game is not None
        assert env.game.state.current_color() == env.p0.color
    finally:
        env.close()


# ---------------------------------------------------------------------------
# Vectorized env and observation-dimension integration
# ---------------------------------------------------------------------------


def test_vectorized_puffer_envs_decode_batches():
    single_envs = make_puffer_vectorized_envs(
        reward_function="win",
        map_type="BASE",
        opponent_configs=["F"],
        num_envs=2,
        vps_to_win=10,
        discard_limit=7,
        expert_config="F",
    )
    multi_envs = make_multi_puffer_vectorized_envs(
        num_players=2,
        map_type="BASE",
        vps_to_win=10,
        discard_limit=7,
        shared_critic=True,
        reward_function="win",
        num_envs=1,
    )

    try:
        single_obs, _ = single_envs.reset(seed=[101, 202])
        single_driver = single_envs.driver_env
        single_dims = compute_single_agent_dims(num_players=2, map_type="BASE")
        single_actor, single_critic, single_masks = decode_puffer_batch(
            flat_obs=single_obs,
            obs_space=single_driver.env_single_observation_space,
            obs_dtype=single_driver.obs_dtype,
            actor_dim=single_dims["actor_dim"],
            critic_dim=single_dims["critic_dim"],
        )
        assert single_actor.shape == (2, single_dims["actor_dim"])
        assert single_critic is not None
        assert single_critic.shape == (2, single_dims["critic_dim"])
        assert single_masks.shape[0] == 2

        multi_obs, _ = multi_envs.reset(seed=[303])
        multi_driver = multi_envs.driver_env
        actor_dim, board_shape, actor_numeric_dim = compute_multiagent_input_dim(2, "BASE")
        assert board_shape is not None
        critic_dim = compute_single_agent_dims(2, "BASE")["critic_dim"]
        multi_actor, multi_critic, multi_masks = decode_puffer_batch(
            flat_obs=multi_obs,
            obs_space=multi_driver.env_single_observation_space,
            obs_dtype=multi_driver.obs_dtype,
            actor_dim=actor_dim,
            critic_dim=critic_dim,
        )
        assert multi_actor.shape == (2, actor_dim)
        assert multi_critic is not None
        assert multi_critic.shape == (2, critic_dim)
        assert multi_masks.shape[0] == 2
        assert actor_numeric_dim > 0
    finally:
        single_envs.close()
        multi_envs.close()


def test_actor_observation_level_shapes_match_computed_dims():
    for actor_level in ("private", "public", "full"):
        _assert_observation_level_shapes(actor_level)


def _assert_observation_level_shapes(actor_level: str):
    single_env = _make_single_env(actor_observation_level=actor_level)
    multi_env = _make_multi_env(actor_observation_level=actor_level)
    try:
        single_obs, _ = single_env.reset(seed=101)
        single_dims = compute_single_agent_dims(
            2,
            "BASE",
            actor_observation_level=actor_level,
        )
        single_actor, single_critic, _ = decode_puffer_batch(
            flat_obs=single_obs,
            obs_space=single_env.env_single_observation_space,
            obs_dtype=single_env.obs_dtype,
            actor_dim=single_dims["actor_dim"],
            critic_dim=single_dims["critic_dim"],
        )
        assert single_actor.shape == (1, single_dims["actor_dim"])
        assert single_critic is not None
        assert single_critic.shape == (1, single_dims["critic_dim"])

        multi_obs, _ = multi_env.reset(seed=202)
        actor_dim, board_shape, actor_numeric_dim = compute_multiagent_input_dim(
            2,
            "BASE",
            actor_observation_level=actor_level,
        )
        assert board_shape is not None
        critic_dim = single_dims["critic_dim"]
        multi_actor, multi_critic, _ = decode_puffer_batch(
            flat_obs=multi_obs,
            obs_space=multi_env.env_single_observation_space,
            obs_dtype=multi_env.obs_dtype,
            actor_dim=actor_dim,
            critic_dim=critic_dim,
        )
        assert multi_actor.shape == (2, actor_dim)
        assert multi_critic is not None
        assert multi_critic.shape == (2, critic_dim)
        assert actor_numeric_dim > 0
    finally:
        single_env.close()
        multi_env.close()


def test_critic_observation_levels_are_indexed_from_full_observation():
    full_dims = compute_single_agent_dims(2, "BASE", critic_observation_level="full")
    full_dim = full_dims["critic_dim"]
    full_batch = np.zeros((3, full_dim), dtype=np.float32)

    for critic_level in ("private", "public", "full"):
        dims = compute_single_agent_dims(2, "BASE", critic_observation_level=critic_level)
        indices = get_observation_indices_from_full(2, "BASE", critic_level)
        assert full_batch[:, indices].shape == (3, dims["critic_dim"])
