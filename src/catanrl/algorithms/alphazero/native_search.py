"""Batched native MCTS: C++ owns the trees, Python only evaluates leaves."""

from __future__ import annotations

from typing import Callable

import numpy as np

from catanrl.envs.cudanatron import NativeGame, NativeSearchPool

LeafEvaluator = Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]


def run_search_pool(
    pool: NativeSearchPool,
    *,
    simulations: int,
    batch_capacity: int,
    evaluate: LeafEvaluator,
    dirichlet_alpha: float | None = None,
    dirichlet_fraction: float = 0.25,
) -> None:
    """Run `simulations` per search using contiguous leaf batches.

    `evaluate(observations, players)` must return `(policy_logits, wdl)` with
    shapes `(N, A)` and `(N, 3)`. Terminal simulations are consumed inside the
    pool and never appear in `observations`.
    """
    if simulations < 0:
        raise ValueError("simulations must be non-negative")
    if batch_capacity < 1:
        raise ValueError("batch_capacity must be positive")

    if dirichlet_alpha is not None:
        pool.add_dirichlet_noise(dirichlet_alpha, dirichlet_fraction)
    pool.add_simulations_all(simulations)
    while pool.remaining_simulations > 0:
        observations, players, tokens = pool.select_leaves(batch_capacity)
        if tokens.size == 0:
            break
        policy, wdl = evaluate(observations, players)
        pool.evaluate_leaves(tokens, policy, wdl)


def play_native_self_play_decision(
    game: NativeGame,
    *,
    simulations: int,
    batch_capacity: int,
    evaluate: LeafEvaluator,
    c_puct: float = 1.5,
    seed: int = 0,
    temperature: float = 1.0,
) -> tuple[int, np.ndarray]:
    """Search one decision from `game` and return `(action, visit_policy)`."""
    with NativeSearchPool([game], c_puct=c_puct, seed=seed) as pool:
        root_obs = game.observation()
        policy, wdl = evaluate(root_obs[None, :], np.asarray([game.current_player]))
        pool.initialize_roots(policy)
        pool.set_root_wdls(wdl)
        run_search_pool(
            pool,
            simulations=simulations,
            batch_capacity=batch_capacity,
            evaluate=evaluate,
        )
        visits = pool.root_visits(0).astype(np.float64)
        if visits.sum() <= 0:
            mask = game.valid_action_mask()
            policy_out = mask.astype(np.float64)
            policy_out /= policy_out.sum()
            return int(np.argmax(policy_out)), policy_out
        if temperature <= 1e-8:
            action = int(np.argmax(visits))
            policy_out = np.zeros_like(visits)
            policy_out[action] = 1.0
            return action, policy_out
        logits = np.log(np.maximum(visits, 1e-8)) / temperature
        logits -= logits.max()
        policy_out = np.exp(logits)
        policy_out /= policy_out.sum()
        action = int(np.random.choice(policy_out.size, p=policy_out))
        return action, policy_out
