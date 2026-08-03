#!/usr/bin/env python3
"""Sweep native MCTS budgets against a raw policy or native F and fixed positions."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import torch

import wandb
from catanrl.eval.native_mcts_budget import (
    run_native_budget_games,
    run_native_budget_position_probes,
)
from catanrl.eval.paired_comparison import exact_mcnemar
from catanrl.experiment_store import KIND_POLICY_VALUE, load_experiment
from catanrl.features.catanatron_utils import ActorObservationLevel, CriticObservationLevel
from catanrl.models import PolicyNetworkWrapper, PolicyValueNetworkWrapper, ValueNetworkWrapper


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure native MCTS depth/search effect on identical frozen-policy positions, "
            "then play paired games against either the same raw policy or native F."
        )
    )
    parser.add_argument("--experiment", required=True, help="Frozen experiment checkpoint")
    parser.add_argument("--which", default="best", help="Checkpoint selector")
    parser.add_argument(
        "--budgets",
        type=int,
        nargs="+",
        default=[96, 256, 512, 1024, 1600],
        help="Simulation counts to evaluate",
    )
    parser.add_argument("--probe-games", type=int, default=8)
    parser.add_argument(
        "--probe-stride",
        type=int,
        default=8,
        help="Probe every Nth non-forced decision on frozen raw-policy trajectories",
    )
    parser.add_argument("--games-per-seat", type=int, default=32)
    parser.add_argument(
        "--game-opponent",
        choices=("raw", "value"),
        default="raw",
        help=(
            "Opponent for paired games: the frozen raw network (raw) or the native "
            "C++ Catanatron F/value player (value)"
        ),
    )
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--inference-batch-size", type=int, default=64)
    parser.add_argument("--inference-wait-ms", type=float, default=2.0)
    parser.add_argument("--c-puct", type=float, default=1.5)
    parser.add_argument(
        "--search-selection",
        choices=("puct", "completed-q"),
        default="puct",
        help="Ordinary PUCT or Canopy-style completed-Q interior allocation",
    )
    parser.add_argument("--c-visit", type=float, default=50.0)
    parser.add_argument("--c-scale", type=float, default=1.0)
    parser.add_argument(
        "--value-scale",
        type=float,
        default=1.0,
        help="Multiplier for non-terminal network values backed up by native MCTS",
    )
    parser.add_argument(
        "--canonical-pruning",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable exact successor/discard-order deduplication during search",
    )
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--turns-limit", type=int, default=1000)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--skip-probes", action="store_true")
    parser.add_argument("--skip-games", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="catan")
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--wandb-group", default="native-mcts-budget-sweep")
    args = parser.parse_args()
    if not args.budgets or any(budget < 1 for budget in args.budgets):
        parser.error("--budgets must contain positive integers")
    if len(set(args.budgets)) != len(args.budgets):
        parser.error("--budgets must not contain duplicates")
    if not math.isfinite(args.value_scale) or args.value_scale < 0.0:
        parser.error("--value-scale must be finite and non-negative")
    if not math.isfinite(args.c_visit) or args.c_visit < 0.0:
        parser.error("--c-visit must be finite and non-negative")
    if not math.isfinite(args.c_scale) or args.c_scale < 0.0:
        parser.error("--c-scale must be finite and non-negative")
    if args.skip_probes and args.skip_games:
        parser.error("Cannot combine --skip-probes and --skip-games")
    return args


def _write_results(output_path: Path, payload: dict[str, Any]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _print_summary(label: str, summary: dict[str, float]) -> None:
    formatted = " | ".join(f"{key}={value:.4g}" for key, value in summary.items())
    print(f"{label}: {formatted}", flush=True)


def main() -> None:
    args = _parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    experiment = load_experiment(args.experiment)
    if experiment.num_players != 2:
        raise ValueError("Native budget sweep currently requires a two-player checkpoint")
    if experiment.model_type is None:
        raise ValueError("Experiment metadata does not specify a policy model type")

    actor_level = cast(
        ActorObservationLevel,
        experiment.policy_spec.observation_level or "private",
    )
    critic_spec = experiment.metadata.networks.get("critic")
    critic_level = cast(
        CriticObservationLevel,
        (critic_spec.observation_level if critic_spec is not None else actor_level) or "full",
    )
    map_type = cast(Any, experiment.map_type)
    vps_to_win = experiment.metadata.game.vps_to_win or 15
    discard_limit = experiment.metadata.game.discard_limit or 9

    if experiment.policy_spec.kind == KIND_POLICY_VALUE:
        policy_model = cast(
            PolicyValueNetworkWrapper,
            experiment.build_policy(
                which=args.which,
                device=device,
                as_policy_only=False,
            ),
        )
        critic_model = None
    else:
        policy_model = cast(
            PolicyNetworkWrapper,
            experiment.build_policy(which=args.which, device=device),
        )
        critic_model = cast(
            ValueNetworkWrapper,
            experiment.build_critic(which=args.which, device=device),
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(
        args.output_dir
        or f"experiments/native-mcts-budget-sweep-{experiment.metadata.name}-{timestamp}"
    )
    output_path = output_dir / "results.json"
    config = {
        "experiment": experiment.metadata.name,
        "experiment_path": experiment.path,
        "checkpoint": args.which,
        "budgets": args.budgets,
        "probe_games": args.probe_games,
        "probe_stride": args.probe_stride,
        "games_per_seat": args.games_per_seat,
        "game_opponent": args.game_opponent,
        "num_workers": args.num_workers,
        "inference_batch_size": args.inference_batch_size,
        "inference_wait_ms": args.inference_wait_ms,
        "c_puct": args.c_puct,
        "search_selection": args.search_selection,
        "c_visit": args.c_visit,
        "c_scale": args.c_scale,
        "value_scale": args.value_scale,
        "canonical_pruning": args.canonical_pruning,
        "seed": args.seed,
        "turns_limit": args.turns_limit,
        "device": str(device),
        "map_type": experiment.map_type,
        "vps_to_win": vps_to_win,
        "discard_limit": discard_limit,
        "actor_observation_level": actor_level,
        "critic_observation_level": critic_level,
    }
    payload: dict[str, Any] = {
        "schema_version": 1,
        "config": config,
        "probe_summaries": {},
        "position_records": [],
        "game_sweeps": {},
        "paired_comparisons": {},
    }
    _write_results(output_path, payload)

    run = None
    if args.wandb:
        run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name
            or f"native-mcts-budget-sweep-{experiment.metadata.name}-{timestamp}",
            group=args.wandb_group,
            job_type="eval",
            config=config,
            tags=["native-mcts", "budget-sweep", "frozen-policy", args.game_opponent],
        )

    print(f"Frozen checkpoint: {experiment.metadata.name} ({args.which})", flush=True)
    print(f"Budgets: {args.budgets} | device={device}", flush=True)
    print(f"Incremental results: {output_path}", flush=True)

    try:
        if not args.skip_probes:
            print("\nRunning identical-position probes...", flush=True)
            probes = run_native_budget_position_probes(
                policy_model=policy_model,
                critic_model=critic_model,
                model_type=experiment.model_type,
                map_type=map_type,
                actor_observation_level=actor_level,
                critic_observation_level=critic_level,
                budgets=args.budgets,
                num_games=args.probe_games,
                probe_stride=args.probe_stride,
                num_workers=args.num_workers,
                inference_batch_size=args.inference_batch_size,
                inference_wait_ms=args.inference_wait_ms,
                c_puct=args.c_puct,
                search_selection=args.search_selection,
                c_visit=args.c_visit,
                c_scale=args.c_scale,
                value_scale=args.value_scale,
                canonical_pruning=args.canonical_pruning,
                seed=args.seed,
                vps_to_win=vps_to_win,
                discard_limit=discard_limit,
                device=device,
                turns_limit=args.turns_limit,
            )
            probe_summaries = probes.summaries(args.budgets)
            payload["probe_summaries"] = {
                str(budget): summary for budget, summary in probe_summaries.items()
            }
            payload["position_records"] = probes.position_records
            _write_results(output_path, payload)
            for budget in args.budgets:
                _print_summary(f"probe s{budget}", probe_summaries[budget])
                if run is not None:
                    run.log(
                        {
                            "budget": budget,
                            **{
                                f"probe/{key}": value
                                for key, value in probe_summaries[budget].items()
                            },
                        },
                        step=budget,
                    )

        if not args.skip_games:
            for budget in args.budgets:
                print(
                    f"\nRunning paired native games for s{budget} "
                    f"vs {args.game_opponent} ({args.games_per_seat} per seat)...",
                    flush=True,
                )
                games = run_native_budget_games(
                    policy_model=policy_model,
                    critic_model=critic_model,
                    model_type=experiment.model_type,
                    map_type=map_type,
                    actor_observation_level=actor_level,
                    critic_observation_level=critic_level,
                    budget=budget,
                    games_per_seat=args.games_per_seat,
                    num_workers=args.num_workers,
                    inference_batch_size=args.inference_batch_size,
                    inference_wait_ms=args.inference_wait_ms,
                    c_puct=args.c_puct,
                    search_selection=args.search_selection,
                    c_visit=args.c_visit,
                    c_scale=args.c_scale,
                    seed=args.seed,
                    vps_to_win=vps_to_win,
                    discard_limit=discard_limit,
                    device=device,
                    turns_limit=args.turns_limit,
                    game_opponent=args.game_opponent,
                    value_scale=args.value_scale,
                    canonical_pruning=args.canonical_pruning,
                )
                summary = games.summary()
                payload["game_sweeps"][str(budget)] = {
                    "summary": summary,
                    "game_records": games.game_records,
                }
                _write_results(output_path, payload)
                _print_summary(f"games s{budget}", summary)
                if run is not None:
                    run.log(
                        {
                            "budget": budget,
                            **{f"games/{key}": value for key, value in summary.items()},
                        },
                        step=budget,
                    )

            baseline_budget = args.budgets[0]
            baseline_games = payload["game_sweeps"][str(baseline_budget)]["game_records"]
            for budget in args.budgets[1:]:
                comparison = exact_mcnemar(
                    baseline_games,
                    payload["game_sweeps"][str(budget)]["game_records"],
                )
                payload["paired_comparisons"][f"{baseline_budget}_vs_{budget}"] = comparison
            _write_results(output_path, payload)

        if run is not None:
            run.summary["results_path"] = str(output_path)
    finally:
        if run is not None:
            run.finish()

    print(f"\nCompleted. Results: {output_path}", flush=True)


if __name__ == "__main__":
    main()
