"""Evaluate two saved experiment policies head-to-head in balanced 1v1 games."""

import argparse
import math
import re
from typing import Any

import torch
import wandb
from tqdm import tqdm

from catanrl.eval.vectorized_rollout import run_policy_h2h_eval_vectorized
from catanrl.experiment_store import Experiment, load_experiment
from catanrl.experiments.common_args import DEFAULT_WANDB_PROJECT


def validate_compatible_experiments(experiment_a: Experiment, experiment_b: Experiment) -> None:
    """Require both policies to target the same 1v1 game and action space."""
    errors = []
    if experiment_a.num_players != 2 or experiment_b.num_players != 2:
        errors.append("both experiments must be 2-player policies")
    if experiment_a.metadata.game != experiment_b.metadata.game:
        errors.append("game configurations differ")
    if experiment_a.policy_spec.num_actions != experiment_b.policy_spec.num_actions:
        errors.append("policy action counts differ")
    if errors:
        raise ValueError("Incompatible experiments: " + "; ".join(errors))


def summarize_seat_results(seat_results: dict[str, tuple[int, list[int]]]) -> dict[str, float]:
    """Create per-seat and combined metrics from experiment A's results."""
    metrics: dict[str, float] = {}
    total_wins = 0
    total_games = 0
    all_turns: list[int] = []
    for seat_mode in ("first", "second"):
        wins, turns = seat_results[seat_mode]
        games = len(turns)
        if games == 0:
            raise ValueError(f"No completed games for seat {seat_mode!r}")
        metrics[f"win_rate_a_{seat_mode}"] = wins / games
        metrics[f"avg_turns_{seat_mode}"] = sum(turns) / games
        total_wins += wins
        total_games += games
        all_turns.extend(turns)
    metrics["win_rate_a"] = total_wins / total_games
    metrics["win_rate_b"] = 1.0 - metrics["win_rate_a"]
    metrics["avg_turns"] = sum(all_turns) / total_games
    metrics["games"] = float(total_games)
    metrics["wins_a"] = float(total_wins)
    metrics["wins_b"] = float(total_games - total_wins)
    return metrics


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


def build_policy_rows(
    metrics: dict[str, float],
    policy_a: str,
    checkpoint_a: str,
    policy_b: str,
    checkpoint_b: str,
) -> list[dict[str, str | float]]:
    """Build symmetric per-policy rows for console and W&B presentation."""
    games = int(metrics["games"])
    a_interval = wilson_interval(int(metrics["wins_a"]), games)
    b_interval = wilson_interval(int(metrics["wins_b"]), games)
    return [
        {
            "policy": policy_a,
            "checkpoint": checkpoint_a,
            "wins": metrics["wins_a"],
            "games": metrics["games"],
            "win_rate": metrics["win_rate_a"],
            "ci95_low": a_interval[0],
            "ci95_high": a_interval[1],
            "first_seat_win_rate": metrics["win_rate_a_first"],
            "second_seat_win_rate": metrics["win_rate_a_second"],
            "avg_turns": metrics["avg_turns"],
            "avg_turns_first": metrics["avg_turns_first"],
            "avg_turns_second": metrics["avg_turns_second"],
        },
        {
            "policy": policy_b,
            "checkpoint": checkpoint_b,
            "wins": metrics["wins_b"],
            "games": metrics["games"],
            "win_rate": metrics["win_rate_b"],
            "ci95_low": b_interval[0],
            "ci95_high": b_interval[1],
            "first_seat_win_rate": 1.0 - metrics["win_rate_a_second"],
            "second_seat_win_rate": 1.0 - metrics["win_rate_a_first"],
            "avg_turns": metrics["avg_turns"],
            "avg_turns_first": metrics["avg_turns_second"],
            "avg_turns_second": metrics["avg_turns_first"],
        },
    ]


def _metric_slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_")


def build_wandb_summary(rows: list[dict[str, str | float]]) -> dict[str, float]:
    """Build named one-shot summary metrics without creating history curves."""
    summary = {
        "eval_h2h/games": float(rows[0]["games"]),
        "eval_h2h/avg_turns": float(rows[0]["avg_turns"]),
    }
    slugs = [_metric_slug(str(row["policy"])) for row in rows]
    if slugs[0] == slugs[1]:
        slugs = [
            f"{slugs[index]}_{_metric_slug(str(row['checkpoint']))}"
            for index, row in enumerate(rows)
        ]
    for slug, row in zip(slugs, rows):
        for metric in (
            "win_rate",
            "ci95_low",
            "ci95_high",
            "first_seat_win_rate",
            "second_seat_win_rate",
            "avg_turns_first",
            "avg_turns_second",
        ):
            summary[f"eval_h2h/{metric}_{slug}"] = float(row[metric])
    return summary


def log_wandb_results(run: Any, rows: list[dict[str, str | float]]) -> None:
    """Store one-shot metrics as summary fields plus a table and bar chart."""
    run.summary.update(build_wandb_summary(rows))
    columns = list(rows[0])
    table = wandb.Table(columns=columns, data=[[row[column] for column in columns] for row in rows])
    run.log(
        {
            "eval_h2h/results_table": table,
            "eval_h2h/win_rate_bar": wandb.plot.bar(
                table,
                "policy",
                "win_rate",
                title="Policy H2H overall win rate",
            ),
        }
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare two saved experiment policies in balanced 1v1 games."
    )
    parser.add_argument("--experiment-a", required=True, help="First experiment name or path.")
    parser.add_argument("--which-a", default="best", help="Checkpoint for experiment A.")
    parser.add_argument("--experiment-b", required=True, help="Second experiment name or path.")
    parser.add_argument("--which-b", default="best", help="Checkpoint for experiment B.")
    parser.add_argument(
        "--games-per-seat",
        type=int,
        default=500,
        help="Games with experiment A in each fixed seat.",
    )
    parser.add_argument("--num-envs", type=int, default=8, help="Parallel game environments.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None, help="cuda or cpu; defaults to CUDA if available.")
    parser.add_argument(
        "--sample-actions",
        action="store_true",
        help="Sample policy actions instead of taking deterministic argmax actions.",
    )
    parser.add_argument("--wandb", action="store_true", help="Log config and results to W&B.")
    parser.add_argument("--wandb-project", default=DEFAULT_WANDB_PROJECT)
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument(
        "--wandb-group",
        default="policy-h2h-eval",
        help="W&B group for policy H2H evaluations.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.games_per_seat < 1:
        raise ValueError("--games-per-seat must be at least 1")
    if args.num_envs < 1:
        raise ValueError("--num-envs must be at least 1")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    experiment_a = load_experiment(args.experiment_a)
    experiment_b = load_experiment(args.experiment_b)
    validate_compatible_experiments(experiment_a, experiment_b)

    policy_a = experiment_a.build_policy(which=args.which_a, device=device)
    policy_b = experiment_b.build_policy(which=args.which_b, device=device)
    game = experiment_a.metadata.game
    observation_level_a = experiment_a.policy_spec.observation_level or "private"
    observation_level_b = experiment_b.policy_spec.observation_level or "private"

    print(f"A: {experiment_a.metadata.name} ({args.which_a})")
    print(f"B: {experiment_b.metadata.name} ({args.which_b})")
    print(f"Observation levels: A={observation_level_a} | B={observation_level_b}")
    print(f"Device: {device} | Games per seat: {args.games_per_seat}")
    print(f"Action selection: {'sample' if args.sample_actions else 'argmax'}")

    seat_results: dict[str, tuple[int, list[int]]] = {}
    with tqdm(total=2 * args.games_per_seat, desc="Policy H2H games") as progress:
        for seat_mode in ("first", "second"):
            seat_results[seat_mode] = run_policy_h2h_eval_vectorized(
                policy_model=policy_a,
                champion_model=policy_b,
                model_type=experiment_a.model_type or "flat",
                map_type=game.map_type,
                num_players=game.num_players,
                num_envs=args.num_envs,
                num_games=args.games_per_seat,
                device=device,
                vps_to_win=game.vps_to_win or 15,
                discard_limit=game.discard_limit or 9,
                deterministic=not args.sample_actions,
                nn_seat=seat_mode,
                actor_observation_level=observation_level_a,
                champion_model_type=experiment_b.model_type or "flat",
                champion_actor_observation_level=observation_level_b,
                seed=args.seed,
                progress_callback=progress.update,
            )

    metrics = summarize_seat_results(seat_results)
    rows = build_policy_rows(
        metrics,
        experiment_a.metadata.name,
        args.which_a,
        experiment_b.metadata.name,
        args.which_b,
    )
    print("\nResults")
    for row in rows:
        print(f"{row['policy']} ({row['checkpoint']}):")
        print(
            f"  Overall: {row['win_rate']:.3%} "
            f"(95% CI {row['ci95_low']:.3%}–{row['ci95_high']:.3%})"
        )
        print(
            f"  First seat: {row['first_seat_win_rate']:.3%} | "
            f"Second seat: {row['second_seat_win_rate']:.3%}"
        )
        print(
            f"  Avg turns: {row['avg_turns']:.1f} "
            f"(first {row['avg_turns_first']:.1f}, second {row['avg_turns_second']:.1f})"
        )
    print(f"Games: {int(metrics['games'])}")

    if args.wandb:
        config: dict[str, Any] = vars(args).copy()
        config["experiment_a_name"] = experiment_a.metadata.name
        config["experiment_b_name"] = experiment_b.metadata.name
        run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            group=args.wandb_group,
            job_type="eval",
            config=config,
        )
        log_wandb_results(run, rows)
        run.finish()


if __name__ == "__main__":
    main()
