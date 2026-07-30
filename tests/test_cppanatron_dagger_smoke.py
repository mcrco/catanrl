from __future__ import annotations

import numpy as np
import torch
from torch import nn

from catanrl.algorithms.common import PolicyAgent
from catanrl.algorithms.imitation_learning.dagger import (
    _collect_dagger_rollouts_vectorized,
)
from catanrl.algorithms.imitation_learning.dataset import AggregatedDataset
from catanrl.envs.cppanatron import make_cppanatron_vectorized_envs
from catanrl.envs.puffer.common import compute_single_agent_dims
from catanrl.features.catanatron_utils import (
    get_actor_indices_from_full,
    get_observation_indices_from_full,
)


class _ZeroPolicy(nn.Module):
    def __init__(self, action_space_size: int):
        super().__init__()
        self.action_space_size = action_space_size

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.zeros(
            (inputs.shape[0], self.action_space_size),
            dtype=inputs.dtype,
            device=inputs.device,
        )


class _ZeroValue(nn.Module):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.zeros(
            (inputs.shape[0], 1),
            dtype=inputs.dtype,
            device=inputs.device,
        )


def test_native_backend_runs_through_dagger_collector():
    num_players = 2
    map_type = "MINI"
    num_envs = 2
    num_steps = 4
    dims = compute_single_agent_dims(num_players, map_type)
    envs = make_cppanatron_vectorized_envs(
        reward_function="shaped",
        map_type=map_type,
        opponent_configs=["F"],
        num_envs=num_envs,
        vps_to_win=6,
        discard_limit=7,
        expert_config="F",
    )
    driver = envs.driver_env
    dataset = AggregatedDataset(
        full_state_dim=dims["critic_dim"],
        num_players=num_players,
        map_type=map_type,
        max_size=num_envs * num_steps,
    )
    policy_agent = PolicyAgent(
        _ZeroPolicy(driver.action_space_size),
        model_type="flat",
        device="cpu",
    )
    critic = _ZeroValue()

    try:
        stats, rollout = _collect_dagger_rollouts_vectorized(
            envs=envs,
            policy_agent=policy_agent,
            critic_model=critic,
            dataset=dataset,
            num_steps=num_steps,
            beta=1.0,
            gamma=0.99,
            device=torch.device("cpu"),
            actor_dim=dims["actor_dim"],
            full_state_dim=dims["critic_dim"],
            actor_observation_indices=get_actor_indices_from_full(
                num_players,
                map_type,
            ),
            critic_observation_indices=get_observation_indices_from_full(
                num_players,
                map_type,
                "full",
            ),
            seed=123,
        )
    finally:
        envs.close()

    expected_samples = num_envs * num_steps
    assert stats.steps == expected_samples
    assert stats.dataset_size == expected_samples
    assert stats.expert_fraction == 1.0
    assert len(dataset) == expected_samples
    assert rollout.full_states.shape == (expected_samples, dims["critic_dim"])
    assert rollout.action_masks.shape == (
        expected_samples,
        driver.action_space_size,
    )
    np.testing.assert_array_equal(rollout.played_actions, rollout.expert_actions)
    assert np.all(
        rollout.action_masks[
            np.arange(expected_samples),
            rollout.expert_actions,
        ]
    )
    assert np.isfinite(dataset.returns[:expected_samples]).all()
