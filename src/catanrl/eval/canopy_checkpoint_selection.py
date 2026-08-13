"""Select CatanRL checkpoints from direct matches against released Canopy."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from catanrl.eval.paired_comparison import exact_mcnemar


def shortlist_from_search_screen(
    payload: Mapping[str, Any],
    *,
    top_k: int,
) -> list[str]:
    """Return the top checkpoint selectors from a completed cheap screen."""

    if top_k < 1:
        raise ValueError("top_k must be positive")
    ranking = list(payload.get("ranking", []))
    if not ranking:
        raise ValueError("Search screen has no completed ranking")
    selectors = [str(row["selector"]) for row in ranking[:top_k]]
    if len(selectors) < top_k:
        raise ValueError(
            f"Search screen has only {len(selectors)} ranked checkpoints; need {top_k}"
        )
    if len(set(selectors)) != len(selectors):
        raise ValueError("Search-screen shortlist contains duplicate selectors")
    return selectors


def rank_direct_canopy_results(
    sweeps: Mapping[str, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Rank equal-size direct matches and retain paired comparisons to the winner."""

    if not sweeps:
        raise ValueError("At least one direct Canopy result is required")
    counts = {len(result["game_records"]) for result in sweeps.values()}
    if len(counts) != 1:
        raise ValueError("Direct Canopy checkpoint results must have equal game counts")
    ranking = []
    for selector, result in sweeps.items():
        summary = result["summary"]
        ranking.append(
            {
                "selector": str(selector),
                "wins": int(summary["wins"]),
                "games": int(summary["games"]),
                "win_rate": float(summary["win_rate"]),
                "mean_vps": float(summary["mean_vps"]),
            }
        )
    ranking.sort(
        key=lambda row: (
            -row["win_rate"],
            -row["mean_vps"],
            (0, int(row["selector"])) if row["selector"].isdigit() else (1, row["selector"]),
        )
    )
    top_selector = str(ranking[0]["selector"])
    top_records: Sequence[Mapping[str, Any]] = sweeps[top_selector]["game_records"]
    paired = {
        str(selector): exact_mcnemar(top_records, result["game_records"])
        for selector, result in sweeps.items()
        if str(selector) != top_selector
    }
    return ranking, paired


__all__ = ["rank_direct_canopy_results", "shortlist_from_search_screen"]
