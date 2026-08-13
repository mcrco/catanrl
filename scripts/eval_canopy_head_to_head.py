#!/usr/bin/env python3
"""Play CatanRL search directly against released Canopy in Catanatron."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, cast

import torch

from catanrl.eval.canopy_catanatron_bridge import CanopyBridgeProcess
from catanrl.eval.canopy_head_to_head import (
    run_canopy_head_to_head,
    summarize_canopy_head_to_head,
)
from catanrl.experiment_store import KIND_POLICY_VALUE, load_experiment
from catanrl.experiments.canopy_contract import validate_canopy_experiment
from catanrl.features.catanatron_utils import (
    ActorObservationLevel,
    CriticObservationLevel,
    get_observation_indices_from_full,
)
from catanrl.models import PolicyNetworkWrapper, PolicyValueNetworkWrapper, ValueNetworkWrapper
from catanrl.utils.seeding import derive_seed


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--which", default="best", help="CatanRL checkpoint selector")
    parser.add_argument("--canopy-binary", type=Path, required=True)
    parser.add_argument("--canopy-checkpoint", type=Path, required=True)
    parser.add_argument("--simulations", type=int, default=1600)
    parser.add_argument("--games-per-seat", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--inference-batch-size", type=int, default=64)
    parser.add_argument("--inference-wait-ms", type=float, default=2.0)
    parser.add_argument("--canopy-batch-size", type=int, default=16)
    parser.add_argument("--canopy-wait-ms", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=52043)
    parser.add_argument("--noninferiority-margin", type=float, default=0.05)
    parser.add_argument("--turns-limit", type=int, default=1000)
    parser.add_argument("--max-actions", type=int, default=2000)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for name in (
        "simulations",
        "games_per_seat",
        "num_workers",
        "inference_batch_size",
        "canopy_batch_size",
    ):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be at least 1")
    for name in ("inference_wait_ms", "canopy_wait_ms"):
        if getattr(args, name) < 0.0:
            parser.error(f"--{name.replace('_', '-')} cannot be negative")
    if args.turns_limit < 1:
        parser.error("--turns-limit must be at least 1")
    if args.max_actions < 0:
        parser.error("--max-actions cannot be negative")
    if not 0.0 <= args.noninferiority_margin < 0.5:
        parser.error("--noninferiority-margin must be in [0, 0.5)")
    return args


def main() -> None:
    args = _parse_args()
    if not args.canopy_binary.is_file():
        raise FileNotFoundError(args.canopy_binary)
    if not args.canopy_checkpoint.is_file():
        raise FileNotFoundError(args.canopy_checkpoint)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    experiment = load_experiment(args.experiment)
    validate_canopy_experiment(experiment)
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
    actor_indices = get_observation_indices_from_full(2, "BASE", actor_level)
    critic_indices = get_observation_indices_from_full(2, "BASE", critic_level)

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

    episode_seeds = [
        derive_seed(args.seed, "canopy_h2h_episode", game_index)
        for game_index in range(args.games_per_seat)
    ]
    scenarios = [(seat, episode_seed) for episode_seed in episode_seeds for seat in (0, 1)]
    config: dict[str, Any] = {
        "map_type": "BASE",
        "num_players": 2,
        "vps_to_win": experiment.metadata.game.vps_to_win or 15,
        "discard_limit": experiment.metadata.game.discard_limit or 9,
        "turns_limit": args.turns_limit,
        "max_actions": args.max_actions,
        "tree_reuse": True,
        "canonical_pruning": True,
        "search_selection": "completed-q",
        "c_puct": 2.5,
        "c_visit": 50.0,
        "c_scale": 1.0,
        "value_scale": 1.0,
        "root_dirichlet_alpha": 0.05,
        "root_dirichlet_fraction": 0.0,
    }
    with CanopyBridgeProcess(
        args.canopy_binary,
        args.canopy_checkpoint,
        simulations=args.simulations,
        seed=args.seed,
    ) as bridge:
        records = run_canopy_head_to_head(
            policy_model=policy_model,
            critic_model=critic_model,
            model_type=experiment.model_type,
            bridge=bridge,
            actor_indices=actor_indices,
            critic_indices=critic_indices,
            scenarios=scenarios,
            budget=args.simulations,
            args_dict=config,
            device=device,
            num_workers=args.num_workers,
            inference_batch_size=args.inference_batch_size,
            inference_wait_ms=args.inference_wait_ms,
            canopy_batch_size=args.canopy_batch_size,
            canopy_wait_ms=args.canopy_wait_ms,
        )
    records.sort(key=lambda record: (int(record["episode_seed"]), str(record["seat"])))

    payload = {
        "schema_version": 1,
        "comparison": "CatanRL native MCTS vs released cullback/canopy Nexus-v3 MCTS",
        "authoritative_engine": "Catanatron",
        "map_layout_source": "cppanatron layout imported into Catanatron",
        "candidate": {
            "experiment": experiment.metadata.name,
            "experiment_path": str(Path(experiment.path).resolve()),
            "selector": str(args.which),
            "policy_checkpoint": str(
                Path(experiment.resolve_checkpoint(args.which, "policy")).resolve()
            ),
            "critic_checkpoint": (
                None
                if critic_model is None
                else str(Path(experiment.resolve_checkpoint(args.which, "critic")).resolve())
            ),
        },
        "canopy": {
            "repository": "https://github.com/cullback/canopy",
            "release_commit": "6185983a88ba6802e7fa9893cef5a76a15de2595",
            "checkpoint": str(args.canopy_checkpoint.resolve()),
            "binary": str(args.canopy_binary.resolve()),
        },
        "config": {
            **config,
            "simulations_per_move_per_agent": args.simulations,
            "games_per_seat": args.games_per_seat,
            "seed": args.seed,
            "num_workers": args.num_workers,
            "inference_batch_size": args.inference_batch_size,
            "inference_wait_ms": args.inference_wait_ms,
            "canopy_batch_size": args.canopy_batch_size,
            "canopy_wait_ms": args.canopy_wait_ms,
            "root_noise": False,
            "noninferiority_margin": args.noninferiority_margin,
        },
        "summary": summarize_canopy_head_to_head(records, args.noninferiority_margin),
        "game_records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
