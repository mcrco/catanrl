"""Utilities for screening saved checkpoints by paired search-game strength."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def numeric_policy_selectors(registry: Any) -> list[int]:
    """Return unique numeric policy selectors in ascending training order."""
    selectors_by_file: dict[str, int] = {}
    for entry in registry.checkpoints:
        if entry.get("role") != "policy" or entry.get("step") is None or not entry.get("file"):
            continue
        step = int(entry["step"])
        filename = str(entry["file"])
        previous = selectors_by_file.get(filename)
        selectors_by_file[filename] = step if previous is None else min(previous, step)
    return sorted(selectors_by_file.values())


def rank_checkpoint_summaries(
    sweeps: Mapping[str, Mapping[str, Any]],
    *,
    top_k: int,
) -> list[dict[str, int | float | str]]:
    """Rank screening results by wins, then VP margin proxy, then recency."""
    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    rows: list[dict[str, int | float | str]] = []
    for selector, payload in sweeps.items():
        summary = payload.get("summary")
        if not isinstance(summary, Mapping):
            raise ValueError(f"Checkpoint {selector!r} has no summary")
        games = int(summary["games"])
        wins = int(summary["wins"])
        if games < 1 or wins < 0 or wins > games:
            raise ValueError(f"Checkpoint {selector!r} has invalid game counts")
        rows.append(
            {
                "selector": selector,
                "games": games,
                "wins": wins,
                "win_rate": wins / games,
                "mean_vps": float(summary.get("mean_vps", 0.0)),
                "win_rate_ci95_low": float(summary.get("win_rate_ci95_low", 0.0)),
                "win_rate_ci95_high": float(summary.get("win_rate_ci95_high", 1.0)),
            }
        )

    def _rank_key(row: Mapping[str, int | float | str]) -> tuple[float, float, int]:
        selector = str(row["selector"])
        numeric_selector = int(selector) if selector.isdigit() else -1
        return float(row["wins"]), float(row["mean_vps"]), numeric_selector

    return sorted(rows, key=_rank_key, reverse=True)[:top_k]
