from __future__ import annotations

import numpy as np
import pytest
from pufferlib.emulation import nativize

from catanrl.envs.cppanatron import (
    SingleAgentCppanatronPufferEnv,
    find_cppanatron_library,
    make_cppanatron_vectorized_envs,
)
from catanrl.envs.puffer.common import compute_single_agent_dims
from catanrl.envs.puffer.rollout_utils import decode_puffer_batch
from catanrl.features.catanatron_utils import get_actor_indices_from_full


@pytest.fixture(scope="module", autouse=True)
def _require_native_library():
    try:
        find_cppanatron_library()
    except FileNotFoundError as exc:
        pytest.skip(str(exc))


def _make_env() -> SingleAgentCppanatronPufferEnv:
    return SingleAgentCppanatronPufferEnv(
        config={
            "map_type": "MINI",
            "vps_to_win": 6,
            "discard_limit": 7,
            "opponent_configs": ["F"],
            "reward_function": "shaped",
            "expert_config": "F",
            "nn_seat": "first",
        }
    )


def _decode(env: SingleAgentCppanatronPufferEnv) -> dict:
    return nativize(
        env.observations[0],
        env.env_single_observation_space,
        env.obs_dtype,
    )


def test_native_puffer_reset_and_steps_preserve_contract():
    env = _make_env()
    try:
        observations, infos = env.reset(seed=123)
        info = infos[0]
        structured = _decode(env)

        assert env.game is not None
        assert env.game.current_player == env.controlled_player
        assert int(info["expert_action"]) in info["valid_actions"]
        np.testing.assert_array_equal(
            np.flatnonzero(structured["action_mask"]),
            info["valid_actions"],
        )

        dims = compute_single_agent_dims(2, "MINI")
        actor, critic, masks = decode_puffer_batch(
            observations,
            env.env_single_observation_space,
            env.obs_dtype,
            actor_dim=dims["actor_dim"],
            critic_dim=dims["critic_dim"],
            actor_indices=get_actor_indices_from_full(2, "MINI"),
        )
        assert actor.shape == (1, dims["actor_dim"])
        assert critic is not None
        assert critic.shape == (1, dims["critic_dim"])
        assert masks.shape == (1, env.action_space_size)

        for _ in range(32):
            action = int(info["expert_action"])
            observations, rewards, terminals, truncations, infos = env.step(
                np.asarray([action], dtype=np.int32)
            )
            info = infos[0]
            assert observations.shape[0] == 1
            assert rewards.shape == (1,)
            assert terminals.shape == (1,)
            assert truncations.shape == (1,)
            assert int(info["expert_action"]) in info["valid_actions"]
            assert env.game is not None
            assert env.game.current_player == env.controlled_player
    finally:
        env.close()


def test_native_puffer_reset_is_seed_reproducible():
    env = _make_env()
    try:
        first_observations, first_infos = env.reset(seed=456)
        first_observations = first_observations.copy()
        second_observations, second_infos = env.reset(seed=456)

        np.testing.assert_array_equal(second_observations, first_observations)
        assert second_infos == first_infos
    finally:
        env.close()


def test_native_puffer_vectorized_factory_decodes_batch():
    envs = make_cppanatron_vectorized_envs(
        reward_function="win",
        map_type="MINI",
        opponent_configs=["F"],
        num_envs=2,
        vps_to_win=6,
        discard_limit=7,
        expert_config="F",
    )
    try:
        observations, infos = envs.reset(seed=[101, 202])
        driver = envs.driver_env
        dims = compute_single_agent_dims(2, "MINI")
        actor, critic, masks = decode_puffer_batch(
            observations,
            driver.env_single_observation_space,
            driver.obs_dtype,
            actor_dim=dims["actor_dim"],
            critic_dim=dims["critic_dim"],
            actor_indices=get_actor_indices_from_full(2, "MINI"),
        )

        assert actor.shape == (2, dims["actor_dim"])
        assert critic is not None
        assert critic.shape == (2, dims["critic_dim"])
        assert masks.shape == (2, driver.action_space_size)
        assert all(int(info["expert_action"]) in info["valid_actions"] for info in infos)
    finally:
        envs.close()
