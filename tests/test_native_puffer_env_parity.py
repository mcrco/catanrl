from __future__ import annotations

import numpy as np
from pufferlib.emulation import nativize

from catanrl.envs.gym.rewards import WinReward
from catanrl.envs.gym.single_env import SingleAgentCatanatronEnv
from catanrl.envs.gym.single_env import create_expert as create_legacy_expert
from catanrl.envs.gym.single_env import create_opponents as create_legacy_opponents
from catanrl.envs.puffer.common import compute_single_agent_dims, create_expert, create_opponents
from catanrl.envs.puffer.multi_agent_env import (
    MultiAgentCatanatronEnvConfig,
    ParallelCatanatronPufferEnv,
    compute_multiagent_input_dim,
    make_vectorized_envs as make_multi_puffer_vectorized_envs,
)
from catanrl.envs.puffer.rollout_utils import decode_puffer_batch, flatten_puffer_observation
from catanrl.envs.puffer.single_agent_env import (
    SingleAgentCatanatronPufferEnv,
    make_puffer_vectorized_envs,
)
from catanrl.envs.zoo.parallel_env import ParallelCatanatronEnv
from catanrl.features.catanatron_utils import get_full_numeric_feature_names


def _decode_single_observation(
    env: SingleAgentCatanatronPufferEnv,
    observation: np.ndarray,
) -> dict:
    return nativize(observation[0], env.env_single_observation_space, env.obs_dtype)


def _decode_multi_observations(
    env: ParallelCatanatronPufferEnv,
    observations: np.ndarray,
) -> dict[str, dict]:
    return {
        agent: nativize(observations[idx], env.env_single_observation_space, env.obs_dtype)
        for idx, agent in enumerate(env.possible_agents)
    }


def test_single_agent_native_puffer_matches_legacy_env_for_first_few_steps():
    legacy_env = SingleAgentCatanatronEnv(
        config={
            "map_type": "BASE",
            "vps_to_win": 10,
            "discard_limit": 7,
            "enemies": create_legacy_opponents(["F"]),
            "reward_function": WinReward(),
            "shared_critic": True,
            "expert_player": create_legacy_expert("F"),
        }
    )
    native_env = SingleAgentCatanatronPufferEnv(
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
        legacy_obs, legacy_info = legacy_env.reset(seed=123)
        native_obs, native_infos = native_env.reset(seed=123)
        native_info = native_infos[0]

        for _ in range(4):
            native_structured = _decode_single_observation(native_env, native_obs)
            native_actor, native_critic = flatten_puffer_observation(native_structured)
            legacy_actor, legacy_critic = flatten_puffer_observation(legacy_obs)

            assert np.array_equal(native_actor, legacy_actor)
            assert np.array_equal(native_critic, legacy_critic)
            assert np.array_equal(native_structured["action_mask"], legacy_obs["action_mask"])
            assert native_info["valid_actions"] == legacy_info["valid_actions"]
            assert int(native_info["expert_action"]) == int(legacy_info["expert_action"])
            assert bool(native_info["nn_won"]) == bool(legacy_info["nn_won"])

            action = int(legacy_info["expert_action"])
            legacy_obs, legacy_reward, legacy_terminated, legacy_truncated, legacy_info = legacy_env.step(
                action
            )
            native_obs, native_rewards, native_terminated, native_truncated, native_infos = (
                native_env.step(np.asarray([action], dtype=np.int32))
            )
            native_info = native_infos[0]

            assert float(native_rewards[0]) == float(legacy_reward)
            assert bool(native_terminated[0]) == bool(legacy_terminated)
            assert bool(native_truncated[0]) == bool(legacy_truncated)

            if legacy_terminated or legacy_truncated:
                break
    finally:
        legacy_env.close()
        native_env.close()


def test_multi_agent_native_puffer_matches_legacy_parallel_env_for_first_few_steps():
    config = MultiAgentCatanatronEnvConfig(
        num_players=2,
        map_type="BASE",
        vps_to_win=10,
        discard_limit=7,
        shared_critic=True,
        reward_function="win",
    )
    legacy_env = ParallelCatanatronEnv(config=config)
    native_env = ParallelCatanatronPufferEnv(config=config)

    try:
        legacy_obs, legacy_infos = legacy_env.reset(seed=321)
        native_obs, native_infos = native_env.reset(seed=321)

        for _ in range(4):
            native_structured = _decode_multi_observations(native_env, native_obs)

            for idx, agent in enumerate(native_env.possible_agents):
                assert np.array_equal(
                    native_structured[agent]["action_mask"],
                    legacy_obs[agent]["action_mask"],
                )
                assert np.array_equal(
                    native_structured[agent]["critic"],
                    legacy_obs[agent]["critic"],
                )
                native_actor, _ = flatten_puffer_observation(native_structured[agent])
                legacy_actor, _ = flatten_puffer_observation(legacy_obs[agent])
                assert np.array_equal(native_actor, legacy_actor)
                assert np.array_equal(
                    np.asarray(native_infos[idx]["valid_actions"], dtype=np.int64),
                    np.asarray(legacy_infos[agent]["valid_actions"], dtype=np.int64),
                )

            actions_by_agent = {
                agent: int(np.asarray(legacy_infos[agent]["valid_actions"], dtype=np.int64)[0])
                for agent in native_env.possible_agents
            }
            native_actions = np.asarray(
                [actions_by_agent[agent] for agent in native_env.possible_agents],
                dtype=np.int32,
            )

            legacy_obs, legacy_rewards, legacy_dones, legacy_truncateds, legacy_infos = legacy_env.step(
                actions_by_agent
            )
            native_obs, native_rewards, native_dones, native_truncateds, native_infos = native_env.step(
                native_actions
            )

            if not legacy_obs:
                break

            for idx, agent in enumerate(native_env.possible_agents):
                assert float(native_rewards[idx]) == float(legacy_rewards[agent])
                assert bool(native_dones[idx]) == bool(legacy_dones[agent])
                assert bool(native_truncateds[idx]) == bool(legacy_truncateds[agent])
    finally:
        legacy_env.close()
        native_env.close()


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
        actor_dim, board_shape, numeric_dim = compute_multiagent_input_dim(2, "BASE")
        assert board_shape is not None
        board_dim = int(np.prod(board_shape))
        critic_dim = len(get_full_numeric_feature_names(2, "BASE")) + board_dim
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
        assert numeric_dim > 0
    finally:
        single_envs.close()
        multi_envs.close()


def test_actor_observation_level_shapes_match_computed_dims():
    for level in ("private", "public", "full"):
        single_env = SingleAgentCatanatronPufferEnv(
            config={
                "map_type": "BASE",
                "vps_to_win": 10,
                "discard_limit": 7,
                "enemies": create_opponents(["F"]),
                "reward_function": WinReward(),
                "shared_critic": True,
                "expert_player": create_expert("F"),
                "actor_observation_level": level,
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
                actor_observation_level=level,
            )
        )
        try:
            single_obs, _ = single_env.reset(seed=101)
            single_dims = compute_single_agent_dims(2, "BASE", actor_observation_level=level)
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
            actor_dim, board_shape, numeric_dim = compute_multiagent_input_dim(
                2,
                "BASE",
                actor_observation_level=level,
            )
            assert board_shape is not None
            critic_dim = len(get_full_numeric_feature_names(2, "BASE")) + int(np.prod(board_shape))
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
            assert numeric_dim > 0
        finally:
            single_env.close()
            multi_env.close()


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
