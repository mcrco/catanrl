#!/usr/bin/env python3
"""Paired native games between search agents (native MCTS, raw policy, F).

Example:
    eval_search_matchup.py --agents-a raw "mcts:sims=256" --agent-b F --games-per-seat 500
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

from catanrl.envs.cppanatron.puffer_env import TURNS_LIMIT
from catanrl.eval.batched_inference import value_transform_for_reward
from catanrl.eval.batched_native_mcts import run_batched_native_parallel
from catanrl.eval.paired_comparison import exact_mcnemar
from catanrl.eval.search_matchups import MatchupResult, parse_agent, run_matchup
from catanrl.experiment_store import load_experiment
from catanrl.utils.seeding import derive_seed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--experiment", default="m-graph-s-full-full-native-g1-from-warm-s42-20261001")
    parser.add_argument("--which", default="1500")
    parser.add_argument("--agents-a", nargs="+", required=True)
    parser.add_argument("--agent-b", required=True)
    parser.add_argument("--games-per-seat", type=int, default=250)
    parser.add_argument("--num-workers", type=int, default=15)
    parser.add_argument("--seed", type=int, default=67)
    parser.add_argument("--max-rows", type=int, default=2048)
    parser.add_argument("--max-wait-ms", type=float, default=1.0)
    parser.add_argument(
        "--native-processes",
        type=int,
        default=8,
        help="Processes for lockstep-batched raw/MCTS games against F (0 disables)",
    )
    parser.add_argument("--native-concurrency", type=int, default=64)
    parser.add_argument("--output", default=None)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="catan")
    parser.add_argument("--wandb-group", default="search-matchup")
    parser.add_argument("--wandb-run-name", default=None)
    args = parser.parse_args()
    for text in [*args.agents_a, args.agent_b]:
        parse_agent(text)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    experiment = load_experiment(args.experiment)
    policy = experiment.build_policy(which=args.which, device=device).eval()
    critic = experiment.build_critic(which=args.which, device=device).eval()
    reward_function = experiment.metadata.train_config.get("reward_function")
    settings = {
        "map_type": experiment.map_type,
        "number_placement": experiment.number_placement,
        "vps_to_win": experiment.metadata.game.vps_to_win or 15,
        "discard_limit": experiment.metadata.game.discard_limit or 9,
        "turns_limit": TURNS_LIMIT,
        "value_transform": value_transform_for_reward(reward_function),
    }
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output = Path(args.output or f"experiments/search-matchup-{timestamp}.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    config = {
        "experiment": args.experiment,
        "which": args.which,
        "agents_a": args.agents_a,
        "agent_b": args.agent_b,
        "games_per_seat": args.games_per_seat,
        "num_workers": args.num_workers,
        "seed": args.seed,
        "reward_function": reward_function,
        **settings,
    }
    payload: dict[str, Any] = {"config": config, "matchups": {}, "paired": {}}

    run = None
    if args.wandb:
        import wandb

        run = wandb.init(
            project=args.wandb_project,
            group=args.wandb_group,
            name=args.wandb_run_name or f"search-matchup-vs-{args.agent_b}-{timestamp}",
            job_type="eval",
            config=config,
            tags=["native-mcts", "search-matchup"],
        )

    def write() -> None:
        output.write_text(json.dumps(payload, indent=2, default=str) + "\n")

    try:
        for index, agent_a in enumerate(args.agents_a):
            print(f"\n=== {agent_a} vs {args.agent_b} ({args.games_per_seat} per seat) ===", flush=True)
            spec_a, spec_b = parse_agent(agent_a), parse_agent(args.agent_b)
            batched = (
                args.native_processes > 0
                and spec_a.kind in {"raw", "mcts"}
                and spec_b.kind == "value"
            )
            if batched:
                # Same seeds and per-decision search seeds as run_matchup.
                scenarios = [
                    (seat, derive_seed(args.seed, "search_matchup_episode", game_index))
                    for game_index in range(args.games_per_seat)
                    for seat in range(2)
                ]
                records, stats_a, throughput = run_batched_native_parallel(
                    experiment_name=args.experiment,
                    which=args.which,
                    value_transform=settings["value_transform"],
                    scenarios=scenarios,
                    simulations=int(spec_a.params.get("sims", 256)) if spec_a.kind == "mcts" else 0,
                    c_puct=float(spec_a.params.get("c_puct", 1.5)),
                    settings={**settings, "seed": args.seed},
                    processes=args.native_processes,
                    concurrency=args.native_concurrency,
                    label=agent_a,
                )
                result = MatchupResult(agent_a=agent_a, agent_b=args.agent_b, records=records, stats_a=stats_a)
                result.seconds = throughput["wall_seconds"]
                summary = result.summary()
                summary["server_rows_per_second"] = throughput["server_rows_per_second"]
                summary.pop("server_mean_batch_rows", None)
                summary["driver"] = "batched-native"
            else:
                result = run_matchup(
                    policy_model=policy,
                    critic_model=critic,
                    model_type=experiment.model_type or "flat",
                    device=device,
                    agent_a=agent_a,
                    agent_b=args.agent_b,
                    games_per_seat=args.games_per_seat,
                    num_workers=args.num_workers,
                    seed=args.seed,
                    settings=settings,
                    max_rows=args.max_rows,
                    max_wait_ms=args.max_wait_ms,
                )
                summary = result.summary()
            payload["matchups"][agent_a] = {
                "summary": summary,
                "records": sorted(result.records, key=lambda r: (r["episode_seed"], r["a_seat"])),
            }
            write()
            print(json.dumps(summary, indent=2), flush=True)
            if run is not None:
                run.log({"matchup_index": index, **{f"{agent_a}/{k}": v for k, v in summary.items()}})
                run.summary[f"{agent_a}/win_rate"] = summary["win_rate"]
                run.summary[f"{agent_a}/evals_per_search"] = summary["a/evals_per_search"]
        baseline = args.agents_a[0]
        for agent_a in args.agents_a[1:]:
            payload["paired"][f"{agent_a}__vs__{baseline}"] = exact_mcnemar(
                payload["matchups"][agent_a]["records"], payload["matchups"][baseline]["records"]
            )
        write()
        if payload["paired"]:
            print(json.dumps(payload["paired"], indent=2), flush=True)
        if run is not None:
            run.summary["results_path"] = str(output)
    finally:
        if run is not None:
            run.finish()
    print(f"wrote {output}", flush=True)


if __name__ == "__main__":
    main()
