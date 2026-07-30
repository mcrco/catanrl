from __future__ import annotations

import numpy as np
import pytest
from pufferlib.emulation import nativize

from catanrl.algorithms.imitation_learning.dagger import _resolve_env_factory
from catanrl.algorithms.ppo.marl_ppo_central_critic import (
    _resolve_marl_env_factory,
)
from catanrl.envs.cppanatron import (
    ParallelCppanatronPufferEnv,
    SingleAgentCppanatronPufferEnv,
    find_cppanatron_library,
    make_cppanatron_marl_vectorized_envs,
    make_cppanatron_vectorized_envs,
)
from catanrl.envs.cppanatron.puffer_env import _native_production_sum
from catanrl.envs.puffer.common import (
    compute_multiagent_input_dim,
    compute_single_agent_dims,
)
from catanrl.envs.puffer.multi_agent_env import (
    make_vectorized_envs as make_marl_vectorized_envs,
)
from catanrl.envs.puffer.rollout_utils import decode_puffer_batch
from catanrl.envs.puffer.single_agent_env import make_puffer_vectorized_envs
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


def test_native_puffer_completes_and_autoresets_episode():
    env = SingleAgentCppanatronPufferEnv(
        config={
            "map_type": "MINI",
            "vps_to_win": 3,
            "discard_limit": 7,
            "opponent_configs": ["F"],
            "reward_function": "shaped",
            "expert_config": "F",
            "nn_seat": "first",
        }
    )
    try:
        _, infos = env.reset(seed=123)

        for _ in range(2_000):
            action = int(infos[0]["expert_action"])
            _, _, terminals, truncations, infos = env.step(np.asarray([action], dtype=np.int32))
            if terminals[0] or truncations[0]:
                break
        else:
            pytest.fail("native expert episode did not finish")

        assert bool(terminals[0])
        assert not bool(truncations[0])
        assert "final_info" in infos[0]
        assert "nn_won" in infos[0]["final_info"]
        assert int(infos[0]["expert_action"]) in infos[0]["valid_actions"]
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


def test_dagger_backend_factory_selection():
    assert _resolve_env_factory("python") is make_puffer_vectorized_envs
    assert _resolve_env_factory("cppanatron") is make_cppanatron_vectorized_envs
    with pytest.raises(ValueError, match="Unknown DAgger environment backend"):
        _resolve_env_factory("other")  # type: ignore[arg-type]


def test_native_puffer_rejects_unsupported_player_types():
    with pytest.raises(ValueError, match="Native opponents"):
        SingleAgentCppanatronPufferEnv(
            config={
                "map_type": "MINI",
                "opponent_configs": ["AB:2"],
                "expert_config": "F",
            }
        )


def test_native_production_sum_is_scoped_to_player():
    class FakeGame:
        robber_coordinate = (99, 99, 99)

        @staticmethod
        def buildings():
            return [(0, 0, 0), (1, 1, 1)]

        @staticmethod
        def tiles():
            return [(0, 0, 0, 0, 0, 0, 6, 0, (0, 1, 2, 3, 4, 5))]

    probability_six = 5.0 / 36.0
    assert _native_production_sum(FakeGame(), 0) == probability_six
    assert _native_production_sum(FakeGame(), 1) == 2.0 * probability_six


def test_native_random_seating_preserves_mixed_opponent_identities():
    env = SingleAgentCppanatronPufferEnv(
        config={
            "map_type": "MINI",
            "vps_to_win": 6,
            "discard_limit": 7,
            "opponent_configs": ["F", "RANDOM"],
            "reward_function": "win",
            "expert_config": "F",
            "nn_seat": "random",
        }
    )
    try:
        for seed in range(8):
            env.reset(seed=seed)
            assignments = env._opponent_configs_by_player
            assert assignments[env.controlled_player] is None
            assert sorted(config for config in assignments if config is not None) == ["F", "RANDOM"]
    finally:
        env.close()


def test_native_marl_reset_exposes_one_active_player_mask():
    env = ParallelCppanatronPufferEnv(
        config={
            "num_players": 2,
            "map_type": "MINI",
            "vps_to_win": 6,
            "discard_limit": 7,
            "shared_critic": True,
            "reward_function": "shaped",
            "actor_observation_level": "public",
        }
    )
    try:
        observations, infos = env.reset(seed=17)
        assert observations.shape[0] == 2
        assert len(infos) == 2
        assert env.game is not None
        for player, info in enumerate(infos):
            valid_actions = np.asarray(info["valid_actions"])
            if player == env.game.current_player:
                assert len(valid_actions) > 1
            else:
                assert valid_actions.tolist() == [env.end_turn_idx]
            assert info["critic_observation"].shape == (env.critic_vector_dim,)
    finally:
        env.close()


def test_native_marl_vectorized_factory_decodes_batch():
    envs = make_cppanatron_marl_vectorized_envs(
        num_players=2,
        map_type="MINI",
        vps_to_win=6,
        discard_limit=7,
        shared_critic=True,
        reward_function="shaped",
        num_envs=2,
        actor_observation_level="public",
    )
    try:
        observations, _ = envs.reset(seed=[101, 202])
        driver = envs.driver_env
        actor_dim, _, _ = compute_multiagent_input_dim(
            2,
            "MINI",
            actor_observation_level="public",
        )
        critic_dim = compute_single_agent_dims(2, "MINI")["critic_dim"]
        actor, critic, masks = decode_puffer_batch(
            observations,
            driver.env_single_observation_space,
            driver.obs_dtype,
            actor_dim=actor_dim,
            critic_dim=critic_dim,
            actor_indices=get_actor_indices_from_full(
                2,
                "MINI",
                level="public",
            ),
        )
        assert actor.shape == (4, actor_dim)
        assert critic is not None
        assert critic.shape == (4, critic_dim)
        assert masks.shape == (4, driver.action_space_size)
    finally:
        envs.close()


def test_marl_backend_factory_selection():
    assert _resolve_marl_env_factory("python") is make_marl_vectorized_envs
    assert _resolve_marl_env_factory("cppanatron") is make_cppanatron_marl_vectorized_envs
    with pytest.raises(ValueError, match="Unknown MARL environment backend"):
        _resolve_marl_env_factory("other")  # type: ignore[arg-type]
