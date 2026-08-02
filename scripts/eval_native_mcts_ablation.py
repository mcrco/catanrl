#!/usr/bin/env python3
"""Paired native-MCTS ablations for exploration, critic weighting, and tree reuse."""

from __future__ import annotations

import argparse
import json
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


ABLATIONS: dict[str, dict[str, Any]] = {
    "baseline": {
        "c_puct": 1.5,
        "value_scale": 1.0,
        "tree_reuse": False,
        "canonical_pruning": False,
    },
    "explore_2p5": {
        "c_puct": 2.5,
        "value_scale": 1.0,
        "tree_reuse": False,
        "canonical_pruning": False,
    },
    "explore_4": {
        "c_puct": 4.0,
        "value_scale": 1.0,
        "tree_reuse": False,
        "canonical_pruning": False,
    },
    "value_half": {
        "c_puct": 2.5,
        "value_scale": 0.5,
        "tree_reuse": False,
        "canonical_pruning": False,
    },
    "policy_only": {
        "c_puct": 2.5,
        "value_scale": 0.0,
        "tree_reuse": False,
        "canonical_pruning": False,
    },
    "reuse": {
        "c_puct": 2.5,
        "value_scale": 0.5,
        "tree_reuse": True,
        "canonical_pruning": False,
    },
    "reuse_prune": {
        "c_puct": 2.5,
        "value_scale": 0.5,
        "tree_reuse": True,
        "canonical_pruning": True,
    },
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--which", default="best")
    parser.add_argument("--budget", type=int, default=512)
    parser.add_argument(
        "--configs",
        nargs="+",
        choices=tuple(ABLATIONS),
        default=list(ABLATIONS),
    )
    parser.add_argument("--probe-games", type=int, default=4)
    parser.add_argument("--probe-stride", type=int, default=12)
    parser.add_argument("--games-per-seat", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--inference-batch-size", type=int, default=64)
    parser.add_argument("--inference-wait-ms", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--turns-limit", type=int, default=1000)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--skip-probes", action="store_true")
    parser.add_argument("--skip-games", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="catan")
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--wandb-group", default="native-mcts-ablation")
    args = parser.parse_args()
    if args.budget < 1:
        parser.error("--budget must be positive")
    if len(set(args.configs)) != len(args.configs):
        parser.error("--configs must not contain duplicates")
    if args.skip_probes and args.skip_games:
        parser.error("Cannot combine --skip-probes and --skip-games")
    return args


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _print_summary(name: str, summary: dict[str, float]) -> None:
    selected = (
        "win_rate",
        "first_win_rate",
        "second_win_rate",
        "search/mean_principal_variation_depth",
        "search/mean_top1_agreement",
        "search/mean_retained_root_visits",
        "search/mean_tree_reused",
        "search/mean_pruned_actions",
        "critic/mse",
        "critic/correlation",
        "critic/least_squares_scale",
    )
    values = " | ".join(f"{key}={summary[key]:.4g}" for key in selected if key in summary)
    print(f"{name}: {values}", flush=True)


def main() -> None:
    args = _parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    experiment = load_experiment(args.experiment)
    if experiment.num_players != 2:
        raise ValueError("Native MCTS ablations currently require a two-player checkpoint")
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
            experiment.build_policy(which=args.which, device=device, as_policy_only=False),
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
        or f"experiments/native-mcts-ablation-{experiment.metadata.name}-{timestamp}"
    )
    output_path = output_dir / "results.json"
    config = {
        "experiment": experiment.metadata.name,
        "experiment_path": experiment.path,
        "checkpoint": args.which,
        "budget": args.budget,
        "configs": {name: ABLATIONS[name] for name in args.configs},
        "probe_games": args.probe_games,
        "probe_stride": args.probe_stride,
        "games_per_seat": args.games_per_seat,
        "num_workers": args.num_workers,
        "inference_batch_size": args.inference_batch_size,
        "inference_wait_ms": args.inference_wait_ms,
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
        "ablations": {},
        "paired_comparisons": {},
    }
    _write(output_path, payload)

    run = None
    if args.wandb:
        run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name
            or f"native-mcts-ablation-{experiment.metadata.name}-{timestamp}",
            group=args.wandb_group,
            job_type="eval",
            config=config,
            tags=["native-mcts", "ablation", "frozen-policy"],
        )

    print(f"Frozen checkpoint: {experiment.metadata.name} ({args.which})", flush=True)
    print(f"Budget: {args.budget} | configs={args.configs} | device={device}", flush=True)
    print(f"Incremental results: {output_path}", flush=True)
    try:
        for config_index, name in enumerate(args.configs):
            ablation = ABLATIONS[name]
            result: dict[str, Any] = {"config": ablation}
            print(f"\nRunning {name}: {ablation}", flush=True)
            if not args.skip_probes:
                probes = run_native_budget_position_probes(
                    policy_model=policy_model,
                    critic_model=critic_model,
                    model_type=experiment.model_type,
                    map_type=map_type,
                    actor_observation_level=actor_level,
                    critic_observation_level=critic_level,
                    budgets=[args.budget],
                    num_games=args.probe_games,
                    probe_stride=args.probe_stride,
                    num_workers=args.num_workers,
                    inference_batch_size=args.inference_batch_size,
                    inference_wait_ms=args.inference_wait_ms,
                    c_puct=float(ablation["c_puct"]),
                    seed=args.seed,
                    vps_to_win=vps_to_win,
                    discard_limit=discard_limit,
                    device=device,
                    turns_limit=args.turns_limit,
                    value_scale=float(ablation["value_scale"]),
                    canonical_pruning=bool(ablation["canonical_pruning"]),
                )
                result["probe_summary"] = probes.summaries([args.budget])[args.budget]
                result["position_records"] = probes.position_records
            if not args.skip_games:
                games = run_native_budget_games(
                    policy_model=policy_model,
                    critic_model=critic_model,
                    model_type=experiment.model_type,
                    map_type=map_type,
                    actor_observation_level=actor_level,
                    critic_observation_level=critic_level,
                    budget=args.budget,
                    games_per_seat=args.games_per_seat,
                    num_workers=args.num_workers,
                    inference_batch_size=args.inference_batch_size,
                    inference_wait_ms=args.inference_wait_ms,
                    c_puct=float(ablation["c_puct"]),
                    seed=args.seed,
                    vps_to_win=vps_to_win,
                    discard_limit=discard_limit,
                    device=device,
                    turns_limit=args.turns_limit,
                    value_scale=float(ablation["value_scale"]),
                    tree_reuse=bool(ablation["tree_reuse"]),
                    canonical_pruning=bool(ablation["canonical_pruning"]),
                )
                result["game_summary"] = games.summary()
                result["game_records"] = games.game_records
                _print_summary(name, result["game_summary"])
            payload["ablations"][name] = result
            _write(output_path, payload)
            if run is not None:
                metrics = {"config_index": config_index, "config_name": name}
                metrics.update(
                    {f"games/{key}": value for key, value in result.get("game_summary", {}).items()}
                )
                metrics.update(
                    {
                        f"probe/{key}": value
                        for key, value in result.get("probe_summary", {}).items()
                    }
                )
                run.log(metrics, step=config_index)

        if not args.skip_games:
            baseline = args.configs[0]
            baseline_games = payload["ablations"][baseline]["game_records"]
            for name in args.configs[1:]:
                payload["paired_comparisons"][f"{baseline}_vs_{name}"] = exact_mcnemar(
                    baseline_games,
                    payload["ablations"][name]["game_records"],
                )
            _write(output_path, payload)
        if run is not None:
            run.summary["results_path"] = str(output_path)
    finally:
        if run is not None:
            run.finish()

    print(f"\nCompleted. Results: {output_path}", flush=True)


if __name__ == "__main__":
    main()
