from __future__ import annotations

import numpy as np

from catanrl.envs.puffer.common import compute_single_agent_dims, create_expert, create_opponents
from catanrl.envs.puffer.multi_agent_env import (
    MultiAgentCatanatronEnvConfig,
    ParallelCatanatronPufferEnv,
    compute_multiagent_input_dim,
    make_vectorized_envs as make_multi_puffer_vectorized_envs,
)
from catanrl.envs.puffer.rewards import WinReward
from catanrl.envs.puffer.rollout_utils import decode_puffer_batch
from catanrl.envs.puffer.single_agent_env import (
    SingleAgentCatanatronPufferEnv,
    make_puffer_vectorized_envs,
)
from catanrl.features.catanatron_utils import get_observation_indices_from_full


def test_vectorized_native_puffer_envs_decode_batches():
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
    single_env = SingleAgentCatanatronPufferEnv(
        config={
            "map_type": "BASE",
            "vps_to_win": 10,
            "discard_limit": 7,
            "enemies": create_opponents(["F"]),
            "reward_function": WinReward(),
            "shared_critic": True,
            "expert_player": create_expert("F"),
            "actor_observation_level": actor_level,
        }
    )
    multi_env = ParallelCatanatronPufferEnv(
        config=MultiAgentCatanatronEnvConfig(
            num_players=2,
            map_type="BASE",
            vps_to_win=10,
            discard_limit=7,
            shared_critic=True,
            reward_function="win",
            actor_observation_level=actor_level,
        )
    )
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


def test_single_agent_step_ignores_stale_action_after_done():
    env = SingleAgentCatanatronPufferEnv(
        config={
            "map_type": "BASE",
            "vps_to_win": 10,
            "discard_limit": 7,
            "enemies": create_opponents(["F"]),
            "reward_function": WinReward(),
            "shared_critic": True,
            "expert_player": create_expert("F"),
        }
    )

    try:
        observations, infos = env.reset(seed=123)
        env._episode_done = True
        reset_observations, reset_rewards, reset_terminals, reset_truncations, reset_infos = env.step(
            np.asarray([97], dtype=np.int32)
        )
        assert reset_observations.shape == observations.shape
        assert float(reset_rewards[0]) == 0.0
        assert len(reset_infos) == 1
        assert bool(reset_terminals[0]) is False
        assert bool(reset_truncations[0]) is False
    finally:
        env.close()


def test_single_agent_fixed_seat_reset_respects_requested_order():
    first_env = SingleAgentCatanatronPufferEnv(
        config={
            "map_type": "BASE",
            "vps_to_win": 10,
            "discard_limit": 7,
            "enemies": create_opponents(["F"]),
            "reward_function": WinReward(),
            "shared_critic": True,
            "nn_seat": "first",
        }
    )
    second_env = SingleAgentCatanatronPufferEnv(
        config={
            "map_type": "BASE",
            "vps_to_win": 10,
            "discard_limit": 7,
            "enemies": create_opponents(["F"]),
            "reward_function": WinReward(),
            "shared_critic": True,
            "nn_seat": "second",
        }
    )

    try:
        first_env.reset(seed=123)
        second_env.reset(seed=123)

        assert first_env.game is not None
        assert second_env.game is not None
        assert first_env.game.state.colors[0] == first_env.p0.color
        assert second_env.game.state.colors[1] == second_env.p0.color
        assert second_env.game.state.colors[0] != second_env.p0.color
    finally:
        first_env.close()
        second_env.close()
