"""Statistical comparison for independent Canopy-reference tournaments."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Mapping

from catanrl.eval.reporting import wilson_interval


@dataclass(frozen=True)
class CanopyNoninferiorityResult:
    candidate_games: int
    candidate_wins: int
    candidate_win_rate: float
    reference_games: int
    reference_wins: int
    reference_win_rate: float
    win_rate_difference: float
    difference_ci95_low: float
    difference_ci95_high: float
    noninferiority_margin: float
    noninferior: bool

    def payload(self) -> dict[str, int | float | bool]:
        return asdict(self)


def _integer_count(summary: Mapping[str, object], key: str) -> int:
    raw_value = summary[key]
    if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
        raise ValueError(f"{key} must be numeric, got {raw_value!r}")
    value = float(raw_value)
    if not math.isfinite(value) or value < 0.0 or not value.is_integer():
        raise ValueError(f"{key} must be a finite non-negative integer, got {value!r}")
    return int(value)


def compare_independent_win_rates(
    candidate: Mapping[str, object],
    reference: Mapping[str, object],
    *,
    noninferiority_margin: float = 0.05,
) -> CanopyNoninferiorityResult:
    """Use Newcombe's Wilson-score interval for an independent rate difference."""
    if not math.isfinite(noninferiority_margin) or not 0.0 <= noninferiority_margin < 1.0:
        raise ValueError("noninferiority_margin must be finite and in [0, 1)")

    candidate_games = _integer_count(candidate, "games")
    candidate_wins = _integer_count(candidate, "wins")
    reference_games = _integer_count(reference, "games")
    reference_wins = _integer_count(reference, "wins")
    if candidate_games < 1 or reference_games < 1:
        raise ValueError("Candidate and reference must each contain at least one game")
    if candidate_wins > candidate_games or reference_wins > reference_games:
        raise ValueError("Wins cannot exceed games")

    candidate_rate = candidate_wins / candidate_games
    reference_rate = reference_wins / reference_games
    candidate_low, candidate_high = wilson_interval(candidate_wins, candidate_games)
    reference_low, reference_high = wilson_interval(reference_wins, reference_games)
    difference = candidate_rate - reference_rate
    # Newcombe interval method 10: combine the two Wilson score intervals
    # without the anti-conservative pooled-variance assumption.
    difference_low = difference - math.sqrt(
        (candidate_rate - candidate_low) ** 2 + (reference_high - reference_rate) ** 2
    )
    difference_high = difference + math.sqrt(
        (candidate_high - candidate_rate) ** 2 + (reference_rate - reference_low) ** 2
    )
    return CanopyNoninferiorityResult(
        candidate_games=candidate_games,
        candidate_wins=candidate_wins,
        candidate_win_rate=candidate_rate,
        reference_games=reference_games,
        reference_wins=reference_wins,
        reference_win_rate=reference_rate,
        win_rate_difference=difference,
        difference_ci95_low=difference_low,
        difference_ci95_high=difference_high,
        noninferiority_margin=noninferiority_margin,
        noninferior=difference_low >= -noninferiority_margin,
    )
