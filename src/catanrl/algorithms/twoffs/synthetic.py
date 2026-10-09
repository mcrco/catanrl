"""Oracle over the paper's generated two-fidelity minimax trees.

Reads the JSON trees written by ``env/generate_minimax_trees.py`` in the
authors' repository (schema ``2ffs-minimax-tree-v1``). Slow samples are drawn
in the same order and from the same ``random.Random`` stream as the reference
implementation, so identical decisions produce identical results.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from typing import Any

from .core import LEAF, MAX, MIN, ChildSpec, TwoFFSConfig

_KINDS = {"max": MAX, "min": MIN, "leaf": LEAF}


class SyntheticTreeOracle:
    def __init__(
        self,
        tree: dict[str, Any],
        *,
        seed: int = 0,
        slow_cost: float | None = None,
        slow_sigma: float | None = None,
    ) -> None:
        self.tree = tree
        self.nodes = {node["id"]: node for node in tree["nodes"]}
        self.rng = random.Random(seed)
        self.fast_query_cost = float(tree["fast_oracle"]["cost"])
        self._slow_cost = float(slow_cost if slow_cost is not None else tree["slow_oracle"]["cost"])
        self.slow_sigma = float(
            slow_sigma if slow_sigma is not None else tree["slow_oracle"]["sigma"]
        )

    def root(self) -> tuple[Any, int]:
        root = self.nodes[self.tree["root"]]
        return root["id"], int(root["remaining_depth"])

    def _spec(self, node_id: str) -> ChildSpec:
        node = self.nodes[node_id]
        return ChildSpec(
            kind=_KINDS[node["player"]],
            remaining_depth=int(node["remaining_depth"]),
            payload=node_id,
            key=node_id,
        )

    def expand(self, payload: Any, kind: int, remaining_depth: int) -> list[ChildSpec]:
        if payload == self.tree["root"]:
            return [self._spec(child) for child in self.tree["root_children"]]
        return [self._spec(child) for child in self.nodes[payload]["children"]]

    def fast_evaluate(self, specs: Sequence[ChildSpec]) -> list[tuple[float, float]]:
        return [
            (float(self.nodes[spec.payload]["fast_value"]), float(self.nodes[spec.payload]["fast_bound"]))
            for spec in specs
        ]

    def fast_bound(self, kind: int, remaining_depth: int) -> float:
        raise NotImplementedError("synthetic trees have no chance nodes")

    def slow_sample(self, payload: Any, kind: int, count: int) -> tuple[list[float], float]:
        mean = float(self.nodes[payload]["slow_mean"])
        samples = [self.rng.gauss(mean, self.slow_sigma) for _ in range(count)]
        return samples, self._slow_cost * count

    def slow_cost_estimate(self) -> float:
        return self._slow_cost


def reference_config(tree: dict[str, Any], **overrides: Any) -> TwoFFSConfig:
    """The reference runner's confidence allocation: delta over all non-root nodes."""
    return TwoFFSConfig(confidence_nodes=max(1, len(tree["nodes"]) - 1), **overrides)


__all__ = ["SyntheticTreeOracle", "reference_config"]
