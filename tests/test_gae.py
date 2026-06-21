"""GAE math: Monte-Carlo equivalence, the lambda=0 TD(0) case, episode-boundary
bootstrapping, and per-env independence. These are the calculations that quietly
corrupt PPO targets if they regress."""

from __future__ import annotations

import numpy as np

from catanrl.algorithms.ppo.gae import compute_gae_batched


def _reference_gae(rewards, values, dones, next_values, gamma, lam):
    """Straightforward [T, E] reference implementation used as ground truth."""
    rewards = np.asarray(rewards, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    dones = np.asarray(dones, dtype=bool)
    next_values = np.asarray(next_values, dtype=np.float64)
    T, E = rewards.shape
    adv = np.zeros((T, E), dtype=np.float64)
    last = np.zeros(E, dtype=np.float64)
    for t in reversed(range(T)):
        nonterminal = (~dones[t]).astype(np.float64)
        boot = next_values if t == T - 1 else values[t + 1]
        delta = rewards[t] + gamma * nonterminal * boot - values[t]
        last = delta + gamma * lam * nonterminal * last
        adv[t] = last
    return adv, adv + values


def test_returns_are_monte_carlo_with_gamma_lambda_one_no_dones():
    # gamma=lambda=1 and no terminations => returns[t] == sum of all future
    # rewards in the rollout plus the final bootstrap (values cancel out).
    rewards = np.array([[1.0], [2.0], [3.0]], dtype=np.float32)
    values = np.array([[5.0], [7.0], [9.0]], dtype=np.float32)
    dones = np.zeros((3, 1), dtype=bool)
    next_values = np.array([4.0], dtype=np.float32)

    advantages, returns = compute_gae_batched(
        rewards, values, dones, next_values, gamma=1.0, gae_lambda=1.0
    )

    expected_returns = np.array([[1 + 2 + 3 + 4], [2 + 3 + 4], [3 + 4]], dtype=np.float32)
    np.testing.assert_allclose(returns, expected_returns, atol=1e-5)
    np.testing.assert_allclose(advantages, expected_returns - values, atol=1e-5)


def test_lambda_zero_is_one_step_td_residual():
    # lambda=0 => advantage[t] == r[t] + gamma * V(next) - V[t] (TD(0)).
    rewards = np.array([[1.0], [0.5]], dtype=np.float32)
    values = np.array([[2.0], [3.0]], dtype=np.float32)
    dones = np.zeros((2, 1), dtype=bool)
    next_values = np.array([10.0], dtype=np.float32)
    gamma = 0.9

    advantages, _ = compute_gae_batched(
        rewards, values, dones, next_values, gamma=gamma, gae_lambda=0.0
    )

    expected = np.array(
        [[1.0 + gamma * 3.0 - 2.0], [0.5 + gamma * 10.0 - 3.0]], dtype=np.float32
    )
    np.testing.assert_allclose(advantages, expected, atol=1e-5)


def test_terminal_step_does_not_bootstrap_across_episode_boundary():
    # done at t=0 means values[1] belongs to a freshly reset, unrelated episode.
    # The t=0 advantage must NOT include any of values[1].
    rewards = np.array([[1.0], [2.0]], dtype=np.float32)
    values = np.array([[0.5], [100.0]], dtype=np.float32)  # values[1] is huge on purpose
    dones = np.array([[True], [False]], dtype=bool)
    next_values = np.array([0.0], dtype=np.float32)
    gamma, lam = 0.99, 0.95

    advantages, _ = compute_gae_batched(
        rewards, values, dones, next_values, gamma=gamma, gae_lambda=lam
    )

    # t=0 terminated: advantage = r0 - V0, with no bootstrap and no future leakage.
    assert advantages[0, 0] == np.float32(1.0 - 0.5)
    # t=1 (last, not done): advantage = r1 + gamma*next_value - V1.
    assert advantages[1, 0] == np.float32(2.0 + gamma * 0.0 - 100.0)


def test_last_step_done_zeros_the_final_bootstrap():
    rewards = np.array([[3.0]], dtype=np.float32)
    values = np.array([[1.0]], dtype=np.float32)
    next_values = np.array([50.0], dtype=np.float32)

    done_adv, _ = compute_gae_batched(
        rewards, values, np.array([[True]]), next_values, gamma=0.9, gae_lambda=0.95
    )
    open_adv, _ = compute_gae_batched(
        rewards, values, np.array([[False]]), next_values, gamma=0.9, gae_lambda=0.95
    )

    assert done_adv[0, 0] == np.float32(3.0 - 1.0)
    assert open_adv[0, 0] == np.float32(3.0 + 0.9 * 50.0 - 1.0)


def test_matches_reference_on_random_rollouts_with_dones():
    rng = np.random.default_rng(0)
    T, E = 12, 5
    rewards = rng.standard_normal((T, E)).astype(np.float32)
    values = rng.standard_normal((T, E)).astype(np.float32)
    dones = rng.random((T, E)) < 0.25
    next_values = rng.standard_normal(E).astype(np.float32)

    advantages, returns = compute_gae_batched(
        rewards, values, dones, next_values, gamma=0.99, gae_lambda=0.95
    )
    ref_adv, ref_ret = _reference_gae(
        rewards, values, dones, next_values, gamma=0.99, lam=0.95
    )

    np.testing.assert_allclose(advantages, ref_adv, atol=1e-4)
    np.testing.assert_allclose(returns, ref_ret, atol=1e-4)


def test_envs_are_independent():
    # Two envs processed together must equal each processed alone.
    rewards = np.array([[1.0, -1.0], [2.0, 0.0], [0.5, 3.0]], dtype=np.float32)
    values = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], dtype=np.float32)
    dones = np.array([[False, True], [True, False], [False, False]], dtype=bool)
    next_values = np.array([1.0, 2.0], dtype=np.float32)

    adv_both, _ = compute_gae_batched(rewards, values, dones, next_values)
    for e in range(2):
        adv_single, _ = compute_gae_batched(
            rewards[:, e : e + 1],
            values[:, e : e + 1],
            dones[:, e : e + 1],
            next_values[e : e + 1],
        )
        np.testing.assert_allclose(adv_both[:, e : e + 1], adv_single, atol=1e-6)
