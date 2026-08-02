from __future__ import annotations

import pytest

from catanrl.algorithms.alphazero.native_search import NativeSearchDiagnostics
from catanrl.eval.native_mcts_budget import (
    CriticCalibrationAccumulator,
    SearchDiagnosticsAccumulator,
)


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
        value_scale=1.0,
        backed_up_network_value=0.1,
        retained_root_visits=0,
        tree_reused=0.0,
        pruned_actions=0,
        coalesced_outcomes=0,
        neural_evaluations=97,
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


def test_critic_calibration_reports_scale_and_affine_fit():
    calibration = CriticCalibrationAccumulator()
    calibration.add(0.25, 1.0)
    calibration.add(-0.25, -1.0)

    summary = calibration.summary()

    assert summary["mse"] == pytest.approx(0.5625)
    assert summary["correlation"] == pytest.approx(1.0)
    assert summary["least_squares_scale"] == pytest.approx(4.0)
    assert summary["affine_scale"] == pytest.approx(4.0)
    assert summary["affine_bias"] == pytest.approx(0.0)
