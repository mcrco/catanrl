import numpy as np
import pytest

from catanrl.envs import compute_single_agent_dims
from catanrl.envs.cppanatron import make_cppanatron_native_marl_vectorized_envs
from catanrl.envs.puffer import rollout_utils
from catanrl.features.catanatron_utils import (
    get_actor_indices_from_full,
    get_observation_indices_from_full,
)


@pytest.mark.parametrize("level", ["full", "public", "private"])
def test_structured_decode_matches_per_row_decode(level, monkeypatch):
    envs = make_cppanatron_native_marl_vectorized_envs(
        num_players=2,
        map_type="BASE",
        vps_to_win=15,
        discard_limit=9,
        shared_critic=True,
        reward_function="shaped",
        num_envs=6,
        actor_observation_level="full",
    )
    observations, _ = envs.reset(seed=3)
    driver = envs.driver_env
    rng = np.random.default_rng(0)
    for _ in range(40):
        masks = observations.view(driver.obs_dtype).reshape(-1)["action_mask"] > 0
        actions = np.array([rng.choice(np.flatnonzero(row)) for row in masks])
        observations, *_ = envs.step(actions)

    dims = compute_single_agent_dims(
        2, "BASE", actor_observation_level=level, critic_observation_level="full"
    )
    kwargs = dict(
        flat_obs=observations,
        obs_space=driver.env_single_observation_space,
        obs_dtype=driver.obs_dtype,
        actor_dim=dims["actor_dim"],
        critic_dim=dims["critic_dim"],
        actor_indices=get_actor_indices_from_full(2, "BASE", level=level),
        critic_indices=get_observation_indices_from_full(2, "BASE", level="full"),
    )
    fast = rollout_utils.decode_puffer_batch(**kwargs)
    monkeypatch.setattr(rollout_utils, "_decode_flat_structured_batch", lambda *a, **k: None)
    slow = rollout_utils.decode_puffer_batch(**kwargs)

    for fast_part, slow_part in zip(fast, slow):
        assert fast_part.dtype == slow_part.dtype
        np.testing.assert_array_equal(fast_part, slow_part)
