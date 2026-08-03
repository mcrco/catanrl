"""Native neural MCTS execution with search-effect diagnostics."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Literal

import numpy as np

from catanrl.envs.cppanatron import NativeGame, NativeMCTSSearch, full_native_features
from catanrl.players.nn_mcts_player import _NNMCTSInferenceBackend, _visit_distribution

MapType = Literal["BASE", "MINI", "TOURNAMENT"]


@dataclass(frozen=True)
class NativeSearchDiagnostics:
    simulations: int
    principal_variation_depth: int
    maximum_depth: int
    mean_depth: float
    legal_actions: int
    prior_top1_action: int
    search_top1_action: int
    top1_agreement: float
    policy_kl: float
    policy_js: float
    prior_entropy: float
    search_entropy: float
    network_value: float
    search_value: float
    value_shift: float
    value_scale: float
    backed_up_network_value: float
    retained_root_visits: int
    tree_reused: float
    pruned_actions: int
    coalesced_outcomes: int
    neural_evaluations: int
    elapsed_seconds: float


@dataclass(frozen=True)
class NativeSearchResult:
    policy: np.ndarray
    action: int
    full_state: np.ndarray
    diagnostics: NativeSearchDiagnostics


def step_game_and_reconcile_search(
    *,
    game: NativeGame,
    map_type: MapType,
    action: int,
    search: NativeMCTSSearch | None,
) -> NativeMCTSSearch | None:
    """Play an action and retain a subtree only when it exactly follows the game."""
    retained = search is not None and search.advance(action)
    if search is not None and not retained:
        search.close()
        search = None

    game.step(action)

    if search is not None:
        root_observation, root_player = search.root_observation()
        source_observation = full_native_features(game, map_type, game.current_player)
        if root_player != game.current_player or not np.array_equal(
            root_observation, source_observation
        ):
            # A canonicalized path can invalidate an otherwise reusable child.
            # A fresh tree is exact and preferable to aborting a self-play batch.
            search.close()
            search = None
    return search


def _softmax(values: np.ndarray) -> np.ndarray:
    finite_values = np.where(np.isfinite(values), values, -np.inf)
    maximum = float(np.max(finite_values))
    if not np.isfinite(maximum):
        return np.full(values.shape, 1.0 / values.size, dtype=np.float64)
    weights = np.exp(finite_values.astype(np.float64) - maximum)
    total = float(weights.sum())
    if not np.isfinite(total) or total <= 0.0:
        return np.full(values.shape, 1.0 / values.size, dtype=np.float64)
    return weights / total


def _entropy(probabilities: np.ndarray) -> float:
    positive = probabilities > 0.0
    return float(-(probabilities[positive] * np.log(probabilities[positive])).sum())


def _policy_divergences(search_policy: np.ndarray, prior: np.ndarray) -> tuple[float, float]:
    epsilon = 1e-12
    policy_kl = float(
        (search_policy * (np.log(search_policy + epsilon) - np.log(prior + epsilon))).sum()
    )
    midpoint = 0.5 * (search_policy + prior)
    policy_js = 0.5 * float(
        (search_policy * (np.log(search_policy + epsilon) - np.log(midpoint + epsilon))).sum()
    ) + 0.5 * float((prior * (np.log(prior + epsilon) - np.log(midpoint + epsilon))).sum())
    return policy_kl, policy_js


def run_native_search_policy(
    *,
    game: NativeGame,
    map_type: MapType,
    inference_backend: _NNMCTSInferenceBackend,
    actor_indices: np.ndarray,
    critic_indices: np.ndarray,
    num_simulations: int,
    c_puct: float,
    search_seed: int,
    add_noise: bool,
    dirichlet_alpha: float,
    dirichlet_frac: float,
    action_temperature: float,
    target_temperature: float | None,
    rng: np.random.Generator,
    value_scale: float = 1.0,
    canonical_pruning: bool = False,
    search: NativeMCTSSearch | None = None,
) -> NativeSearchResult:
    """Run one search and report how much it changed the root network prediction."""
    if num_simulations < 1:
        raise ValueError("num_simulations must be at least 1")
    if not np.isfinite(value_scale) or value_scale < 0.0:
        raise ValueError("value_scale must be finite and non-negative")

    started = time.perf_counter()
    owns_search = search is None
    if search is None:
        search = NativeMCTSSearch(
            game,
            map_type,
            c_puct=c_puct,
            seed=search_seed,
            canonical_pruning=canonical_pruning,
        )
    try:
        search.reset_metrics()
        full_state, root_player = search.root_observation()
        if root_player != game.current_player:
            raise RuntimeError("Native MCTS root player does not match its source game")
        source_state = full_native_features(game, map_type, game.current_player)
        if not np.array_equal(full_state, source_state):
            raise RuntimeError("Reused native MCTS tree diverged from its source game")
        root_evaluation = inference_backend.evaluate_leaf(
            full_state[actor_indices],
            full_state[critic_indices],
        )
        neural_evaluations = 1
        if not search.root_expanded:
            search.initialize_root(root_evaluation.policy_logits)
        if add_noise and dirichlet_frac > 0.0:
            search.add_root_dirichlet_noise(
                dirichlet_alpha,
                dirichlet_frac,
            )
        for _ in range(num_simulations):
            leaf = search.select_leaf()
            if leaf is None:
                continue
            full_leaf_state, _player = leaf
            evaluation = inference_backend.evaluate_leaf(
                full_leaf_state[actor_indices],
                full_leaf_state[critic_indices],
            )
            neural_evaluations += 1
            search.evaluate_leaf(
                evaluation.policy_logits,
                float(evaluation.value) * value_scale,
            )
        visits = search.root_visits().astype(np.float64)
        search_metrics = search.metrics()
    finally:
        if owns_search:
            search.close()

    valid_indices = np.flatnonzero(game.valid_action_mask())
    if valid_indices.size < 2:
        raise ValueError("Native MCTS diagnostics require at least two legal actions")
    valid_visits = visits[valid_indices]
    target_probabilities = _visit_distribution(
        valid_visits,
        action_temperature if target_temperature is None else target_temperature,
    )
    action_probabilities = _visit_distribution(valid_visits, action_temperature)
    policy = np.zeros(game.action_space_size, dtype=np.float32)
    policy[valid_indices] = target_probabilities.astype(np.float32, copy=False)
    action = int(rng.choice(valid_indices, p=action_probabilities))

    prior = _softmax(root_evaluation.policy_logits[valid_indices])
    policy_kl, policy_js = _policy_divergences(target_probabilities, prior)
    prior_top1_action = int(valid_indices[int(np.argmax(prior))])
    search_top1_action = int(valid_indices[int(np.argmax(valid_visits))])
    network_value = float(root_evaluation.value)
    search_value = float(search_metrics.root_value)
    diagnostics = NativeSearchDiagnostics(
        simulations=search_metrics.simulations,
        principal_variation_depth=search_metrics.principal_variation_depth,
        maximum_depth=search_metrics.maximum_depth,
        mean_depth=search_metrics.mean_depth,
        legal_actions=int(valid_indices.size),
        prior_top1_action=prior_top1_action,
        search_top1_action=search_top1_action,
        top1_agreement=float(prior_top1_action == search_top1_action),
        policy_kl=policy_kl,
        policy_js=policy_js,
        prior_entropy=_entropy(prior),
        search_entropy=_entropy(target_probabilities),
        network_value=network_value,
        search_value=search_value,
        value_shift=search_value - network_value,
        value_scale=value_scale,
        backed_up_network_value=float(np.clip(network_value * value_scale, -1.0, 1.0)),
        retained_root_visits=search_metrics.retained_root_visits,
        tree_reused=float(search_metrics.tree_reused),
        pruned_actions=search_metrics.pruned_actions,
        coalesced_outcomes=search_metrics.coalesced_outcomes,
        neural_evaluations=neural_evaluations,
        elapsed_seconds=time.perf_counter() - started,
    )
    return NativeSearchResult(
        policy=policy,
        action=action,
        full_state=full_state,
        diagnostics=diagnostics,
    )


__all__ = [
    "NativeSearchDiagnostics",
    "NativeSearchResult",
    "run_native_search_policy",
    "step_game_and_reconcile_search",
]
