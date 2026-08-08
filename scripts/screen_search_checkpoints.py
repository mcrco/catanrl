#!/usr/bin/env python3
"""Screen saved checkpoints with paired native MCTS games against Catanatron F."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, cast

import torch

from catanrl.eval.checkpoint_selection import (
    numeric_policy_selectors,
    rank_checkpoint_summaries,
)
from catanrl.eval.native_mcts_budget import run_native_budget_games
from catanrl.eval.paired_comparison import exact_mcnemar
from catanrl.experiment_store import KIND_POLICY_VALUE, load_experiment
from catanrl.experiments.canopy_contract import validate_canopy_experiment
from catanrl.features.catanatron_utils import ActorObservationLevel, CriticObservationLevel
from catanrl.models import PolicyNetworkWrapper, PolicyValueNetworkWrapper, ValueNetworkWrapper


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--selectors", nargs="+", help="Defaults to every numeric checkpoint")
    parser.add_argument("--budget", type=int, default=64)
    parser.add_argument("--games-per-seat", type=int, default=24)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--inference-batch-size", type=int, default=64)
    parser.add_argument("--inference-wait-ms", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=41051)
    parser.add_argument("--turns-limit", type=int, default=1000)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    for name in ("budget", "games_per_seat", "top_k", "num_workers"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be at least 1")
    return args


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main() -> None:
    args = _parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    experiment = load_experiment(args.experiment)
    validate_canopy_experiment(experiment)
    if experiment.num_players != 2:
        raise ValueError("Checkpoint screening requires a two-player experiment")
    if experiment.model_type is None:
        raise ValueError("Experiment metadata does not specify a policy model type")

    default_selectors = numeric_policy_selectors(experiment.registry)
    selectors = args.selectors or [str(selector) for selector in default_selectors]
    if not selectors:
        raise ValueError("Experiment has no numeric checkpoints to screen")
    # Resolve everything before spending GPU time so a stale registry fails fast.
    for selector in selectors:
        experiment.resolve_checkpoint(selector, "policy")

    actor_level = cast(
        ActorObservationLevel,
        experiment.policy_spec.observation_level or "private",
    )
    critic_spec = experiment.metadata.networks.get("critic")
    critic_level = cast(
        CriticObservationLevel,
        (critic_spec.observation_level if critic_spec is not None else actor_level) or "full",
    )
    output_path = args.output or Path(experiment.path) / "eval" / "search-checkpoint-screen.json"
    payload: dict[str, Any] = {
        "schema_version": 1,
        "config": {
            "experiment": experiment.metadata.name,
            "selectors": selectors,
            "budget": args.budget,
            "games_per_seat": args.games_per_seat,
            "top_k": args.top_k,
            "game_opponent": "value",
            "seed": args.seed,
            "device": str(device),
            "search_selection": "completed-q",
            "c_puct": 2.5,
            "c_visit": 50.0,
            "c_scale": 1.0,
            "root_noise": False,
        },
        "sweeps": {},
        "ranking": [],
        "paired_vs_top": {},
    }
    _write(output_path, payload)

    for selector in selectors:
        print(f"\nScreening checkpoint {selector} at s{args.budget}...", flush=True)
        if experiment.policy_spec.kind == KIND_POLICY_VALUE:
            policy_model = cast(
                PolicyValueNetworkWrapper,
                experiment.build_policy(which=selector, device=device, as_policy_only=False),
            )
            critic_model = None
        else:
            policy_model = cast(
                PolicyNetworkWrapper,
                experiment.build_policy(which=selector, device=device),
            )
            critic_model = cast(
                ValueNetworkWrapper,
                experiment.build_critic(which=selector, device=device),
            )

        games = run_native_budget_games(
            policy_model=policy_model,
            critic_model=critic_model,
            model_type=experiment.model_type,
            map_type=cast(Any, experiment.map_type),
            actor_observation_level=actor_level,
            critic_observation_level=critic_level,
            budget=args.budget,
            games_per_seat=args.games_per_seat,
            num_workers=args.num_workers,
            inference_batch_size=args.inference_batch_size,
            inference_wait_ms=args.inference_wait_ms,
            c_puct=2.5,
            search_selection="completed-q",
            c_visit=50.0,
            c_scale=1.0,
            seed=args.seed,
            vps_to_win=experiment.metadata.game.vps_to_win or 15,
            discard_limit=experiment.metadata.game.discard_limit or 9,
            device=device,
            turns_limit=args.turns_limit,
            game_opponent="value",
            value_scale=1.0,
            tree_reuse=True,
            canonical_pruning=True,
        )
        payload["sweeps"][str(selector)] = {
            "summary": games.summary(),
            "game_records": games.game_records,
        }
        _write(output_path, payload)
        del games, policy_model, critic_model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    ranking = rank_checkpoint_summaries(payload["sweeps"], top_k=args.top_k)
    payload["ranking"] = ranking
    top_selector = str(ranking[0]["selector"])
    top_games = payload["sweeps"][top_selector]["game_records"]
    payload["paired_vs_top"] = {
        selector: exact_mcnemar(top_games, result["game_records"])
        for selector, result in payload["sweeps"].items()
        if selector != top_selector
    }
    _write(output_path, payload)
    print("\nScreening ranking:")
    for index, row in enumerate(ranking, start=1):
        print(
            f"  {index}. {row['selector']}: {float(row['win_rate']):.1%} "
            f"({row['wins']}/{row['games']}), mean VP {float(row['mean_vps']):.3f}"
        )
    print(f"Independent full-budget confirmation candidates: {output_path}")


if __name__ == "__main__":
    main()
