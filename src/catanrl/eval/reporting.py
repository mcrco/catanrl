"""Shared result aggregation and W&B presentation for evaluation scripts."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class EvalResult:
    wins: int
    vps: list[int]
    total_vps: list[int]
    turns: list[int]

    @classmethod
    def from_tuple(cls, result: tuple[int, list[int], list[int], list[int]]) -> "EvalResult":
        return cls(*result)

    @property
    def games(self) -> int:
        return len(self.turns)


def wilson_interval(wins: int, games: int, z: float = 1.96) -> tuple[float, float]:
    """Return a bounded approximate 95% binomial confidence interval."""
    if games < 1:
        raise ValueError("Cannot compute a confidence interval without games")
    rate = wins / games
    denominator = 1.0 + z**2 / games
    center = (rate + z**2 / (2.0 * games)) / denominator
    radius = z * math.sqrt(rate * (1.0 - rate) / games + z**2 / (4.0 * games**2))
    radius /= denominator
    return center - radius, center + radius


def summarize_eval_results(
    label: str,
    checkpoint: str,
    results: Mapping[str, EvalResult],
) -> dict[str, str | float]:
    """Build one combined row, retaining fixed-seat metrics when available."""
    if not results:
        raise ValueError("At least one evaluation result is required")
    if any(result.games < 1 for result in results.values()):
        raise ValueError("Each evaluation result must contain at least one game")

    wins = sum(result.wins for result in results.values())
    games = sum(result.games for result in results.values())
    all_vps = [value for result in results.values() for value in result.vps]
    all_total_vps = [value for result in results.values() for value in result.total_vps]
    all_turns = [value for result in results.values() for value in result.turns]
    ci_low, ci_high = wilson_interval(wins, games)
    row: dict[str, str | float] = {
        "agent": label,
        "checkpoint": checkpoint,
        "wins": float(wins),
        "games": float(games),
        "win_rate": wins / games,
        "ci95_low": ci_low,
        "ci95_high": ci_high,
        "avg_vps": sum(all_vps) / len(all_vps),
        "avg_total_vps": sum(all_total_vps) / len(all_total_vps),
        "avg_turns": sum(all_turns) / len(all_turns),
    }
    for seat in ("first", "second"):
        if seat in results:
            result = results[seat]
            row[f"{seat}_seat_win_rate"] = result.wins / result.games
            row[f"avg_vps_{seat}"] = sum(result.vps) / result.games
            row[f"avg_turns_{seat}"] = sum(result.turns) / result.games
    return row


def print_eval_rows(rows: Sequence[Mapping[str, str | float]]) -> None:
    """Print the common evaluation summary used by standalone scripts."""
    print("\nResults")
    for row in rows:
        print(f"{row['agent']} ({row['checkpoint']}):")
        print(
            f"  Overall: {float(row['win_rate']):.3%} "
            f"({int(float(row['wins']))}/{int(float(row['games']))}; "
            f"95% CI {float(row['ci95_low']):.3%}–{float(row['ci95_high']):.3%})"
        )
        if "first_seat_win_rate" in row:
            print(
                f"  First seat: {float(row['first_seat_win_rate']):.3%} | "
                f"Second seat: {float(row['second_seat_win_rate']):.3%}"
            )
        print(
            f"  Avg VPs: {float(row['avg_vps']):.2f} | "
            f"Avg total VPs: {float(row['avg_total_vps']):.2f} | "
            f"Avg turns: {float(row['avg_turns']):.1f}"
        )


def _metric_slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_")


def build_wandb_summary(
    rows: Sequence[Mapping[str, str | float]], namespace: str = "eval"
) -> dict[str, float]:
    """Build stable, named one-shot summary fields for evaluation rows."""
    summary: dict[str, float] = {}
    slugs = [_metric_slug(str(row["agent"])) for row in rows]
    if len(set(slugs)) != len(slugs):
        slugs = [f"{slug}_{_metric_slug(str(row['checkpoint']))}" for slug, row in zip(slugs, rows)]
    excluded = {"agent", "checkpoint"}
    for slug, row in zip(slugs, rows):
        for metric, value in row.items():
            if metric not in excluded:
                summary[f"{namespace}/{metric}_{slug}"] = float(value)
    if len(rows) == 1:
        for metric, value in rows[0].items():
            if metric not in excluded:
                summary[f"{namespace}/{metric}"] = float(value)
    return summary


def log_wandb_eval_results(
    run: Any,
    rows: Sequence[Mapping[str, str | float]],
    wandb_module: Any,
    namespace: str = "eval",
    chart_title: str = "Evaluation win rate",
) -> None:
    """Store one-shot metrics as summary fields plus a table and bar chart."""
    if not rows:
        raise ValueError("At least one evaluation row is required")
    run.summary.update(build_wandb_summary(rows, namespace=namespace))
    columns = list(rows[0])
    table = wandb_module.Table(
        columns=columns,
        data=[[row.get(column) for column in columns] for row in rows],
    )
    run.log(
        {
            f"{namespace}/results_table": table,
            f"{namespace}/win_rate_bar": wandb_module.plot.bar(
                table, "agent", "win_rate", title=chart_title
            ),
        }
    )
