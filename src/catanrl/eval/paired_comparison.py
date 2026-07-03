"""Persistence and exact McNemar statistics for paired game evaluations."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

SCHEMA_VERSION = 1
GAME_COLUMNS = ("seat", "episode_seed", "win", "vps", "total_vps", "turns")


def write_paired_results(
    path: str | Path,
    *,
    agent: str,
    checkpoint: str,
    base_seed: int,
    scenario: Mapping[str, Any],
    games: Sequence[Mapping[str, Any]],
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "agent": agent,
        "checkpoint": checkpoint,
        "base_seed": int(base_seed),
        "scenario": dict(scenario),
        "games": [dict(game) for game in games],
    }
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def load_paired_results(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text())
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Unsupported paired-results schema in {path}")
    if not isinstance(payload.get("games"), list) or not payload["games"]:
        raise ValueError(f"Paired-results file has no games: {path}")
    return payload


def _game_key(game: Mapping[str, Any]) -> tuple[str, int]:
    return str(game["seat"]), int(game["episode_seed"])


def _index_games(games: Sequence[Mapping[str, Any]]) -> dict[tuple[str, int], bool]:
    indexed: dict[tuple[str, int], bool] = {}
    for game in games:
        key = _game_key(game)
        if key in indexed:
            raise ValueError(f"Duplicate paired game key: seat={key[0]}, seed={key[1]}")
        indexed[key] = bool(game["win"])
    return indexed


def exact_mcnemar(
    games_a: Sequence[Mapping[str, Any]],
    games_b: Sequence[Mapping[str, Any]],
) -> dict[str, int | float]:
    """Return an exact two-sided McNemar comparison for matched game outcomes."""
    indexed_a = _index_games(games_a)
    indexed_b = _index_games(games_b)
    if indexed_a.keys() != indexed_b.keys():
        only_a = len(indexed_a.keys() - indexed_b.keys())
        only_b = len(indexed_b.keys() - indexed_a.keys())
        raise ValueError(f"Paired game keys differ (only A: {only_a}, only B: {only_b})")

    both_win = a_only = b_only = both_loss = 0
    for key, a_win in indexed_a.items():
        b_win = indexed_b[key]
        if a_win and b_win:
            both_win += 1
        elif a_win:
            a_only += 1
        elif b_win:
            b_only += 1
        else:
            both_loss += 1

    discordant = a_only + b_only
    if discordant == 0:
        p_value = 1.0
    else:
        lower_tail = sum(math.comb(discordant, index) for index in range(min(a_only, b_only) + 1))
        p_value = min(1.0, 2.0 * lower_tail / (1 << discordant))

    games = len(indexed_a)
    wins_a = both_win + a_only
    wins_b = both_win + b_only
    return {
        "games": games,
        "wins_a": wins_a,
        "wins_b": wins_b,
        "win_rate_a": wins_a / games,
        "win_rate_b": wins_b / games,
        "win_rate_difference": (wins_a - wins_b) / games,
        "both_win": both_win,
        "a_only_win": a_only,
        "b_only_win": b_only,
        "both_loss": both_loss,
        "discordant": discordant,
        "p_value_exact_two_sided": p_value,
    }


def compare_paired_payloads(
    payload_a: Mapping[str, Any], payload_b: Mapping[str, Any]
) -> dict[str, dict[str, int | float]]:
    if payload_a.get("scenario") != payload_b.get("scenario"):
        raise ValueError("Evaluation scenarios differ; refusing a paired comparison")
    games_a = payload_a["games"]
    games_b = payload_b["games"]
    comparisons = {"overall": exact_mcnemar(games_a, games_b)}
    seats = sorted({str(game["seat"]) for game in games_a})
    for seat in seats:
        seat_a = [game for game in games_a if str(game["seat"]) == seat]
        seat_b = [game for game in games_b if str(game["seat"]) == seat]
        comparisons[seat] = exact_mcnemar(seat_a, seat_b)
    return comparisons


def log_paired_games_to_wandb(
    run: Any, games: Sequence[Mapping[str, Any]], wandb_module: Any
) -> None:
    table = wandb_module.Table(
        columns=list(GAME_COLUMNS),
        data=[[game[column] for column in GAME_COLUMNS] for game in games],
    )
    run.log({"eval/paired_games": table})
