"""Matched comparison of released Canopy and native CatanRL search games."""

from __future__ import annotations

import math
from statistics import NormalDist
from typing import Any, Mapping, Sequence

from .paired_comparison import exact_mcnemar


def _require_equal(label: str, canopy: object, candidate: object) -> None:
    if canopy != candidate:
        raise ValueError(
            f"Parity contract differs for {label}: Canopy={canopy!r}, candidate={candidate!r}"
        )


def _seat_counts(games: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for game in games:
        seat = str(game["seat"])
        counts[seat] = counts.get(seat, 0) + 1
    return counts


def paired_win_rate_interval(
    candidate_games: Sequence[Mapping[str, Any]],
    canopy_games: Sequence[Mapping[str, Any]],
    *,
    confidence: float = 0.95,
) -> dict[str, float]:
    """Normal paired interval for candidate-minus-Canopy win rate.

    The per-scenario observations are in ``{-1, 0, +1}``, retaining the
    covariance obtained from using identical seat/map/game seeds.
    """

    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between zero and one")
    candidate = {
        (str(game["seat"]), int(game["episode_seed"])): bool(game["win"])
        for game in candidate_games
    }
    canopy = {
        (str(game["seat"]), int(game["episode_seed"])): bool(game["win"]) for game in canopy_games
    }
    if len(candidate) != len(candidate_games) or len(canopy) != len(canopy_games):
        raise ValueError("Duplicate seat/episode_seed in paired games")
    if candidate.keys() != canopy.keys():
        raise ValueError("Paired game keys differ")

    differences = [float(candidate[key]) - float(canopy[key]) for key in candidate]
    count = len(differences)
    if count < 2:
        raise ValueError("At least two paired games are required")
    mean = sum(differences) / count
    variance = sum((difference - mean) ** 2 for difference in differences) / (count - 1)
    standard_error = math.sqrt(variance / count)
    z = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    return {
        "difference": mean,
        "standard_error": standard_error,
        "confidence": confidence,
        "ci_low": mean - z * standard_error,
        "ci_high": mean + z * standard_error,
    }


def compare_canopy_to_native(
    canopy_payload: Mapping[str, Any],
    native_payload: Mapping[str, Any],
    *,
    budget: int | None = None,
    noninferiority_margin: float = 0.05,
) -> dict[str, Any]:
    """Validate a common contract and compare matched search-game outcomes."""

    if not 0.0 <= noninferiority_margin < 1.0:
        raise ValueError("noninferiority_margin must be in [0, 1)")
    if canopy_payload.get("implementation") != "cullback/canopy adapted into Catanatron":
        raise ValueError("Canopy payload was not produced by the Catanatron adapter")
    if canopy_payload.get("game_engine") != "Catanatron":
        raise ValueError("Released Canopy games were not governed by Catanatron")
    if canopy_payload.get("map_layout_source") != (
        "cppanatron layout imported into Catanatron"
    ):
        raise ValueError("Released Canopy games did not use the shared cppanatron map layout")

    config = native_payload["config"]
    if config.get("authoritative_engine") != "catanatron":
        raise ValueError("Candidate games were not governed by Catanatron")
    _require_equal(
        "map layout source",
        canopy_payload["map_layout_source"],
        config.get("map_layout_source"),
    )
    canopy_budget = int(canopy_payload["simulations"])
    selected_budget = canopy_budget if budget is None else int(budget)
    sweep = native_payload["game_sweeps"].get(str(selected_budget))
    if sweep is None:
        raise ValueError(f"Native result has no {selected_budget}-simulation game sweep")

    _require_equal("search simulations", canopy_budget, selected_budget)
    _require_equal(
        "opponent",
        canopy_payload["opponent"],
        "F" if config["game_opponent"] == "value" else config["game_opponent"],
    )
    _require_equal("map type", canopy_payload["map_type"], config["map_type"])
    _require_equal("VP target", int(canopy_payload["vps_to_win"]), int(config["vps_to_win"]))
    _require_equal(
        "discard threshold",
        int(canopy_payload["discard_limit"]),
        int(config["discard_limit"]),
    )
    _require_equal(
        "games per seat",
        int(canopy_payload["games_per_seat"]),
        int(config["games_per_seat"]),
    )
    _require_equal("base seed", int(canopy_payload["seed"]), int(config["seed"]))
    _require_equal(
        "maximum actions",
        int(canopy_payload["max_actions"]),
        int(config["max_actions"]),
    )
    _require_equal(
        "root noise",
        float(canopy_payload["root_noise"]),
        float(config["root_dirichlet_fraction"]),
    )

    canopy_games = list(canopy_payload["game_records"])
    candidate_games = list(sweep["game_records"])
    expected_seats = {
        "first": int(canopy_payload["games_per_seat"]),
        "second": int(canopy_payload["games_per_seat"]),
    }
    _require_equal("Canopy seat counts", _seat_counts(canopy_games), expected_seats)
    _require_equal("candidate seat counts", _seat_counts(candidate_games), expected_seats)

    paired = {"overall": exact_mcnemar(candidate_games, canopy_games)}
    for seat in ("first", "second"):
        paired[seat] = exact_mcnemar(
            [game for game in candidate_games if game["seat"] == seat],
            [game for game in canopy_games if game["seat"] == seat],
        )
    interval = paired_win_rate_interval(candidate_games, canopy_games)
    # A one-sided 95% lower bound uses z_0.95 rather than the two-sided
    # interval's z_0.975.
    noninferiority_low = (
        interval["difference"] - NormalDist().inv_cdf(0.95) * interval["standard_error"]
    )
    return {
        "scenario": {
            "game_engine": "Catanatron",
            "opponent": "F",
            "map_type": canopy_payload["map_type"],
            "number_placement": canopy_payload["number_placement"],
            "num_players": int(canopy_payload["num_players"]),
            "vps_to_win": int(canopy_payload["vps_to_win"]),
            "discard_limit": int(canopy_payload["discard_limit"]),
            "simulations": selected_budget,
            "root_noise": 0.0,
            "games_per_seat": int(canopy_payload["games_per_seat"]),
            "seed": int(canopy_payload["seed"]),
            "max_actions": int(canopy_payload["max_actions"]),
        },
        "candidate": {
            "agent": config["experiment"],
            "checkpoint": str(config["checkpoint"]),
            "wins": sum(bool(game["win"]) for game in candidate_games),
            "games": len(candidate_games),
        },
        "canopy": {
            "agent": "cullback/canopy nexus-v3",
            "checkpoint": canopy_payload["checkpoint_name"],
            "wins": sum(bool(game["win"]) for game in canopy_games),
            "games": len(canopy_games),
        },
        "paired": paired,
        "win_rate_difference_interval": interval,
        "noninferiority": {
            "margin": noninferiority_margin,
            "confidence": 0.95,
            "one_sided_ci_low": noninferiority_low,
            "passes": noninferiority_low > -noninferiority_margin,
        },
    }


__all__ = ["compare_canopy_to_native", "paired_win_rate_interval"]
