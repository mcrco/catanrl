"""2FFS core: identification on small trees, chance nodes, budgets."""

from __future__ import annotations

import random

import pytest

from catanrl.algorithms.twoffs.core import (
    CHANCE,
    LEAF,
    MIN,
    ChildSpec,
    TwoFFS,
    TwoFFSConfig,
)
from catanrl.algorithms.twoffs.synthetic import SyntheticTreeOracle, reference_config


def _synthetic_tree(depth: int, branching: int, seed: int, beta: float) -> dict:
    """A full alternating tree in the reference generator's JSON schema."""
    rng = random.Random(seed)
    nodes: list[dict] = []

    def build(node_id: str, depth_index: int, player: str) -> float:
        remaining = depth - depth_index
        if remaining == 0:
            value = rng.uniform(0.2, 0.8)
            nodes.append(
                {
                    "id": node_id,
                    "depth": depth_index,
                    "remaining_depth": 0,
                    "player": "leaf",
                    "children": [],
                    "value": value,
                    "slow_mean": value,
                    "fast_value": value,
                    "fast_bound": 0.0,
                }
            )
            return value
        children = [f"{node_id}.{index}" for index in range(branching)]
        child_player = "min" if player == "max" else "max"
        values = [build(child, depth_index + 1, child_player) for child in children]
        value = max(values) if player == "max" else min(values)
        bound = beta * remaining / depth
        nodes.append(
            {
                "id": node_id,
                "depth": depth_index,
                "remaining_depth": remaining,
                "player": player,
                "children": children,
                "value": value,
                "slow_mean": value,
                "fast_value": value + rng.uniform(-bound, bound),
                "fast_bound": bound,
            }
        )
        return value

    root_values = [build(f"a{index}", 1, "min") for index in range(branching)]
    best = max(range(branching), key=lambda index: root_values[index])
    nodes.append(
        {
            "id": "r",
            "depth": 0,
            "remaining_depth": depth,
            "player": "max",
            "children": [f"a{index}" for index in range(branching)],
            "value": root_values[best],
        }
    )
    return {
        "tree_id": f"t{seed}",
        "root": "r",
        "root_children": [f"a{index}" for index in range(branching)],
        "depth": depth,
        "nodes": nodes,
        "optimal_root_child": f"a{best}",
        "root_gap": 0.0,
        "fast_oracle": {"cost": 1.0},
        "slow_oracle": {"cost": 2.0, "sigma": 0.02},
    }


@pytest.mark.parametrize("seed", range(6))
def test_identifies_the_minimax_optimal_root_action(seed: int) -> None:
    tree = _synthetic_tree(depth=3, branching=3, seed=seed, beta=0.4)
    result = TwoFFS(SyntheticTreeOracle(tree, seed=seed), reference_config(tree)).run()
    assert result.stopped
    assert result.recommendation_key == tree["optimal_root_child"]


def test_latching_only_changed_paths_identifies_the_same_action() -> None:
    for seed in range(6):
        tree = _synthetic_tree(depth=3, branching=3, seed=seed, beta=0.4)
        result = TwoFFS(
            SyntheticTreeOracle(tree, seed=seed), reference_config(tree, latch_all=False)
        ).run()
        assert result.stopped
        assert result.recommendation_key == tree["optimal_root_child"]


def test_budget_stops_early_and_falls_back_to_the_leader() -> None:
    tree = _synthetic_tree(depth=4, branching=4, seed=3, beta=0.6)
    result = TwoFFS(SyntheticTreeOracle(tree), reference_config(tree, budget=6.0)).run()
    assert not result.stopped
    assert result.stop_reason == "budget"
    lowers = [low for low, _ in result.root_intervals]
    assert result.recommendation_index == lowers.index(max(lowers))


class _ChanceOracle:
    """Root chooses between a sure 0.3 and a coin flip between 1.0 and -0.2.

    The gamble is worth 0.4, so the right answer requires the chance backup
    rather than either outcome alone. Fast values are exact but carry a wide
    bound until the chance node's children are revealed.
    """

    fast_query_cost = 1.0
    slow_sigma = 0.1

    def __init__(self) -> None:
        self.rng = random.Random(0)

    def root(self):
        return "root", 2

    def expand(self, payload, kind, remaining_depth):
        assert payload == "root"
        return [
            ChildSpec(LEAF, 1, "sure", exact_value=0.3, key="sure"),
            ChildSpec(
                CHANCE,
                1,
                "gamble",
                key="gamble",
                outcomes=[
                    ChildSpec(LEAF, 1, "win", probability=0.5, exact_value=1.0),
                    ChildSpec(LEAF, 1, "lose", probability=0.5, exact_value=-0.2),
                ],
            ),
        ]

    def fast_evaluate(self, specs):
        raise AssertionError("every child here is exact")

    def fast_bound(self, kind, remaining_depth):
        return 0.0

    def slow_sample(self, payload, kind, count):
        raise AssertionError("exact children never need slow samples")

    def slow_cost_estimate(self):
        return 10.0


def test_chance_nodes_back_up_expectations() -> None:
    search = TwoFFS(_ChanceOracle(), TwoFFSConfig(confidence_nodes=10))
    result = search.run()
    assert result.stopped
    assert result.recommendation_key == "gamble"
    gamble = result.root_intervals[result.root_keys.index("gamble")]
    assert gamble == pytest.approx((0.4, 0.4))


class _NoisyChanceOracle(_ChanceOracle):
    """Same game, but the chance outcomes are depth-limited leaves with fast
    bias, so certifying the gamble needs slow samples at its children."""

    def expand(self, payload, kind, remaining_depth):
        specs = super().expand(payload, kind, remaining_depth)
        for outcome in specs[1].outcomes:
            outcome.exact_value = None
            outcome.kind = MIN
            outcome.remaining_depth = 0
        return specs

    def fast_evaluate(self, specs):
        values = {"win": 1.0, "lose": -0.2}
        return [(values[spec.payload] + 0.05, 0.5) for spec in specs]

    def slow_sample(self, payload, kind, count):
        means = {"win": 1.0, "lose": -0.2, "gamble": 0.4}
        return [self.rng.gauss(means[payload], self.slow_sigma) for _ in range(count)], 10.0 * count


def test_chance_children_are_certified_by_slow_samples() -> None:
    result = TwoFFS(_NoisyChanceOracle(), TwoFFSConfig(confidence_nodes=10, delta=0.05)).run()
    assert result.stopped
    assert result.recommendation_key == "gamble"
    assert result.num_slow_queries > 0
