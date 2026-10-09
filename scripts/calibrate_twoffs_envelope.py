#!/usr/bin/env python3
"""Calibrate the 2FFS fast-oracle bias envelope for a frozen critic.

2FFS treats the critic as a fast oracle with ``|V_F(v) - V*(v)| <= B(h)``. The
paper builds ``B`` from a Bellman-residual bound (Remark 2.2: ``B(h) = eta *
sum_{i<h} gamma^i``) plus, for non-terminal frontiers, the critic's error
against the slow oracle. This script measures both on positions from games of
the frozen policy against native F:

* leaf error: critic minus the mean of ``--rollouts`` argmax-policy self-play
  rollouts (the slow oracle), with the rollout noise reported alongside;
* one-ply residual: critic minus the best probability-weighted critic value
  over the mover's legal actions (the max backup the search applies).

It also reports how often the one-ply critic argmax agrees with the policy
argmax, and the root gap between the two best one-ply values.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from catanrl.algorithms.twoffs.catan import CatanOracle, CatanTwoFFSConfig
from catanrl.algorithms.twoffs.core import CHANCE
from catanrl.envs.cppanatron import FlatGame
from catanrl.eval.batched_inference import LocalEvaluator, value_transform_for_reward
from catanrl.experiment_store import load_experiment
from catanrl.utils.seeding import derive_map_and_game_seeds, derive_seed


def _quantiles(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {}
    return {
        "n": float(array.size),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "abs_mean": float(np.abs(array).mean()),
        **{f"abs_q{int(q * 100)}": float(np.quantile(np.abs(array), q)) for q in (0.5, 0.8, 0.9, 0.95)},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--experiment", default="m-graph-s-full-full-native-g1-from-warm-s42-20261001")
    parser.add_argument("--which", default="1500")
    parser.add_argument("--games", type=int, default=24)
    parser.add_argument("--stride", type=int, default=6, help="Probe every Nth policy decision")
    parser.add_argument("--rollouts", type=int, default=32)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    experiment = load_experiment(args.experiment)
    policy = experiment.build_policy(which=args.which, device=device).eval()
    critic = experiment.build_critic(which=args.which, device=device).eval()
    transform = value_transform_for_reward(experiment.metadata.train_config.get("reward_function"))
    evaluator = LocalEvaluator(
        policy, critic, experiment.model_type or "flat", device, 3770, value_transform=transform
    )
    print(f"critic value transform: {transform}", flush=True)
    game_settings = experiment.metadata.game
    config = CatanTwoFFSConfig(depth=1)

    leaf_errors: list[float] = []
    rollout_noise: list[float] = []
    residuals: list[float] = []
    root_gaps: list[float] = []
    agreements: list[float] = []
    rollout_costs: list[float] = []
    records = []
    started = time.perf_counter()
    for game_index in range(args.games):
        episode_seed = derive_seed(args.seed, "twoffs_calibration", game_index)
        map_seed, game_seed = derive_map_and_game_seeds(episode_seed)
        game = FlatGame(
            2,
            experiment.map_type,
            seed=game_seed,
            map_seed=map_seed,
            number_placement=experiment.number_placement,
            discard_limit=game_settings.discard_limit or 9,
            vps_to_win=game_settings.vps_to_win or 15,
        )
        policy_seat = game_index % 2
        rng = np.random.default_rng(episode_seed)
        decision = 0
        while game.winner is None and game.num_turns < config.turns_limit:
            legal = np.flatnonzero(game.action_mask())
            if legal.size == 1:
                game.step(int(legal[0]))
                continue
            if game.current_player != policy_seat:
                game.step(game.expert_action())
                continue
            logits, values = evaluator.evaluate(game.observation(game.current_player)[None])
            policy_action = int(legal[np.argmax(logits[0][legal])])
            if decision % args.stride == 0:
                oracle = CatanOracle(game.game.copy(), game.space, evaluator, config, rng)
                specs = oracle.expand(oracle.game, 0, 1)
                fast_specs = []
                for spec in specs:
                    fast_specs.extend(spec.outcomes if spec.kind == CHANCE else [spec])
                needed = [spec for spec in fast_specs if spec.exact_value is None]
                fast = dict(zip(map(id, needed), oracle.fast_evaluate(needed))) if needed else {}

                def value(spec, fast=fast):
                    return spec.exact_value if spec.exact_value is not None else fast[id(spec)][0]

                q_values = []
                for spec in specs:
                    if spec.kind == CHANCE:
                        q_values.append(sum(o.probability * value(o) for o in spec.outcomes))
                    else:
                        q_values.append(value(spec))
                q_values = np.asarray(q_values)
                order = np.argsort(-q_values)
                critic_value = float(values[0])
                before = oracle.network_evaluations
                samples = oracle.rollout([game.game.copy() for _ in range(args.rollouts)])
                rollout_costs.append((oracle.network_evaluations - before) / args.rollouts)
                mean = float(np.mean(samples))
                noise = float(np.std(samples) / np.sqrt(len(samples)))
                leaf_errors.append(critic_value - mean)
                rollout_noise.append(noise)
                residuals.append(critic_value - float(q_values[order[0]]))
                root_gaps.append(float(q_values[order[0]] - q_values[order[1]]))
                greedy_action = int(specs[int(order[0])].key)
                agreements.append(float(greedy_action == policy_action))
                records.append(
                    {
                        "game": game_index,
                        "turn": game.num_turns,
                        "legal": int(legal.size),
                        "critic": critic_value,
                        "rollout_mean": mean,
                        "rollout_se": noise,
                        "best_q": float(q_values[order[0]]),
                        "second_q": float(q_values[order[1]]),
                        "policy_action": policy_action,
                        "greedy_action": greedy_action,
                        "policy_action_q": float(q_values[[s.key for s in specs].index(policy_action)]),
                    }
                )
            game.step(policy_action)
            decision += 1
        print(
            f"game {game_index + 1}/{args.games}: positions={len(records)} "
            f"elapsed={time.perf_counter() - started:.0f}s",
            flush=True,
        )

    noise_var = float(np.mean(np.square(rollout_noise))) if rollout_noise else 0.0
    leaf_var = float(np.var(leaf_errors)) if leaf_errors else 0.0
    summary = {
        "experiment": args.experiment,
        "which": args.which,
        "positions": len(records),
        "rollouts_per_position": args.rollouts,
        "mean_rollout_evaluations": float(np.mean(rollout_costs)) if rollout_costs else 0.0,
        "leaf_error": _quantiles(leaf_errors),
        "rollout_standard_error_rms": float(np.sqrt(noise_var)),
        # Var(critic - mean) = Var(true error) + Var(noise): subtract the noise.
        "leaf_error_rms_denoised": float(
            np.sqrt(max(0.0, leaf_var + np.mean(leaf_errors) ** 2 - noise_var))
        )
        if leaf_errors
        else 0.0,
        "one_ply_residual": _quantiles(residuals),
        "root_gap": _quantiles(root_gaps),
        "greedy_policy_agreement": float(np.mean(agreements)) if agreements else 0.0,
        "policy_action_regret_vs_greedy_q": _quantiles(
            [r["best_q"] - r["policy_action_q"] for r in records]
        ),
    }
    print(json.dumps(summary, indent=2))
    output = Path(args.output or f"experiments/twoffs-calibration-{args.experiment}-{args.which}.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"summary": summary, "records": records}, indent=2) + "\n")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
