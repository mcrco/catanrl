from __future__ import annotations

import numpy as np

from catanrl.algorithms.imitation_learning.dagger import _compute_discounted_returns


def test_win_loss_returns_propagate_within_each_episode_only() -> None:
    rewards = np.array([[0.0], [1.0], [0.0], [-1.0]], dtype=np.float32)
    dones = np.array([[False], [True], [False], [True]])

    returns = _compute_discounted_returns(
        rewards,
        dones,
        bootstrap_values=np.array([123.0], dtype=np.float32),
        gamma=1.0,
    )

    np.testing.assert_array_equal(
        returns,
        np.array([[1.0], [1.0], [-1.0], [-1.0]], dtype=np.float32),
    )


def test_unfinished_episode_uses_bootstrap_value() -> None:
    rewards = np.zeros((2, 2), dtype=np.float32)
    dones = np.zeros((2, 2), dtype=np.bool_)

    returns = _compute_discounted_returns(
        rewards,
        dones,
        bootstrap_values=np.array([0.25, -0.5], dtype=np.float32),
        gamma=1.0,
    )

    np.testing.assert_array_equal(
        returns,
        np.array([[0.25, -0.5], [0.25, -0.5]], dtype=np.float32),
    )
