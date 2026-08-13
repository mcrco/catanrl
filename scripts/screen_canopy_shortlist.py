#!/usr/bin/env python3
"""Choose a checkpoint by playing a cheap shortlist directly against Canopy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, cast

import torch

from catanrl.eval.canopy_catanatron_bridge import CanopyBridgeProcess
from catanrl.eval.canopy_checkpoint_selection import (
    rank_direct_canopy_results,
    shortlist_from_search_screen,
)
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
    parser.add_argument("--search-screen", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--canopy-binary", type=Path, required=True)
    parser.add_argument("--canopy-checkpoint", type=Path, required=True)
    parser.add_argument("--simulations", type=int, default=64)
    parser.add_argument("--games-per-seat", type=int, default=24)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--inference-batch-size", type=int, default=64)
    parser.add_argument("--inference-wait-ms", type=float, default=2.0)
    parser.add_argument("--canopy-batch-size", type=int, default=8)
    parser.add_argument("--canopy-wait-ms", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=51051)
    parser.add_argument("--turns-limit", type=int, default=1000)
    parser.add_argument("--max-actions", type=int, default=2000)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selected-output", type=Path, required=True)
    args = parser.parse_args()
    for name in (
        "top_k",
        "simulations",
        "games_per_seat",
        "num_workers",
        "inference_batch_size",
        "canopy_batch_size",
    ):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be at least 1")
    if args.max_actions < 0:
        parser.error("--max-actions cannot be negative")
    return args


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main() -> None:
    args = _parse_args()
    # A failed selector must never leave a stale checkpoint that a downstream
    # expensive run can mistake for fresh direct-Canopy evidence.
    args.selected_output.unlink(missing_ok=True)
    screen_payload = json.loads(args.search_screen.read_text())
    selectors = shortlist_from_search_screen(screen_payload, top_k=args.top_k)
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
    scenarios = [
        (seat, derive_seed(args.seed, "canopy_h2h_episode", game_index))
        for game_index in range(args.games_per_seat)
        for seat in (0, 1)
    ]
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
    payload: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "selection_opponent": "released cullback/canopy nexus-v3",
        "authoritative_engine": "Catanatron",
        "shortlist_source": str(args.search_screen.resolve()),
        "selectors": selectors,
        "config": {
            **config,
            "simulations_per_move_per_agent": args.simulations,
            "games_per_seat": args.games_per_seat,
            "seed": args.seed,
            "root_noise": False,
        },
        "sweeps": {},
        "ranking": [],
        "paired_vs_top": {},
    }
    _write(args.output, payload)

    for selector in selectors:
        print(f"\nDirectly screening checkpoint {selector} against Canopy...", flush=True)
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
        records.sort(key=lambda row: (int(row["episode_seed"]), str(row["seat"])))
        payload["sweeps"][selector] = {
            "summary": summarize_canopy_head_to_head(records, 0.05),
            "game_records": records,
        }
        _write(args.output, payload)
        del policy_model, critic_model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    ranking, paired = rank_direct_canopy_results(payload["sweeps"])
    payload["ranking"] = ranking
    payload["paired_vs_top"] = paired
    selected = str(ranking[0]["selector"])
    payload["selected"] = selected
    payload["status"] = "complete"
    _write(args.output, payload)
    args.selected_output.parent.mkdir(parents=True, exist_ok=True)
    args.selected_output.write_text(selected + "\n")
    print(f"Selected checkpoint {selected} by direct Canopy play; wrote {args.output}")


if __name__ == "__main__":
    main()
