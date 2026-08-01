from __future__ import annotations

import pytest

from catanrl.algorithms.alphazero.native_search import NativeSearchDiagnostics
from catanrl.eval.native_mcts_budget import SearchDiagnosticsAccumulator


def _diagnostics(*, depth: int, agreement: float) -> NativeSearchDiagnostics:
    return NativeSearchDiagnostics(
        simulations=96,
        principal_variation_depth=depth,
        maximum_depth=depth + 2,
        mean_depth=depth + 0.5,
        legal_actions=8,
        prior_top1_action=3,
        search_top1_action=3 if agreement else 4,
        top1_agreement=agreement,
        policy_kl=0.2,
        policy_js=0.05,
        prior_entropy=1.5,
        search_entropy=1.2,
        network_value=0.1,
        search_value=0.3,
        value_shift=0.2,
        elapsed_seconds=0.4,
    )


def test_search_diagnostics_accumulator_merges_weighted_summaries():
    first = SearchDiagnosticsAccumulator()
    first.add(_diagnostics(depth=2, agreement=1.0))
    second = SearchDiagnosticsAccumulator()
    second.add(_diagnostics(depth=4, agreement=0.0))

    first.merge(second.payload())
    summary = first.summary()

    assert summary["searches"] == 2.0
    assert summary["mean_simulations"] == 96.0
    assert summary["mean_principal_variation_depth"] == 3.0
    assert summary["mean_top1_agreement"] == 0.5
    assert summary["maximum_observed_depth"] == 6.0
    assert summary["simulations_per_second"] == pytest.approx(240.0)
