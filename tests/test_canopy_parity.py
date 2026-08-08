from __future__ import annotations

import pytest

from catanrl.eval.canopy_parity import (
    compare_independent_win_rates,
    validate_matching_action_cap,
)


def test_canopy_noninferiority_passes_for_equal_well_measured_rates():
    comparison = compare_independent_win_rates(
        {"games": 1000, "wins": 800},
        {"games": 1000, "wins": 800},
        noninferiority_margin=0.05,
    )

    assert comparison.win_rate_difference == pytest.approx(0.0)
    assert comparison.difference_ci95_low > -0.05
    assert comparison.noninferior is True


def test_canopy_noninferiority_rejects_clearly_weaker_candidate():
    comparison = compare_independent_win_rates(
        {"games": 1000.0, "wins": 650.0},
        {"games": 1000.0, "wins": 800.0},
        noninferiority_margin=0.05,
    )

    assert comparison.win_rate_difference == pytest.approx(-0.15)
    assert comparison.difference_ci95_high < -0.05
    assert comparison.noninferior is False


@pytest.mark.parametrize(
    "candidate, reference, message",
    [
        ({"games": 0, "wins": 0}, {"games": 10, "wins": 5}, "at least one"),
        ({"games": 10, "wins": 11}, {"games": 10, "wins": 5}, "exceed"),
        ({"games": 10.5, "wins": 5}, {"games": 10, "wins": 5}, "integer"),
    ],
)
def test_canopy_noninferiority_rejects_invalid_counts(candidate, reference, message):
    with pytest.raises(ValueError, match=message):
        compare_independent_win_rates(candidate, reference)


def test_canopy_reference_contract_requires_matching_action_caps():
    assert (
        validate_matching_action_cap(
            {"max_actions": 2000},
            {"max_actions": 2000},
        )
        == 2000
    )

    with pytest.raises(ValueError, match="Incomparable max_actions"):
        validate_matching_action_cap(
            {"max_actions": 1000},
            {"max_actions": 2000},
        )


@pytest.mark.parametrize("value", [None, True, 0, -1, 2000.0])
def test_canopy_reference_contract_rejects_invalid_action_caps(value):
    with pytest.raises(ValueError, match="positive integer"):
        validate_matching_action_cap(
            {"max_actions": value},
            {"max_actions": 2000},
        )
