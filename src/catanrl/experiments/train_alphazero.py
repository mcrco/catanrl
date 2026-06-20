#!/usr/bin/env python3
"""
Experiment script for training the AlphaZero-style agent.

This mirrors the structure we use for other RL entrypoints (e.g., PPO) and
provides a convenient CLI for self-play + neural training loops.
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List, Optional

import torch
import wandb
from tqdm import tqdm

from catanatron.players.minimax import AlphaBetaPlayer
from catanatron.players.value import ValueFunctionPlayer

from catanrl.experiments.architecture_config import (
    ArchitecturePreset,
    add_config_argument,
    architecture_train_config_fields,
)
from catanrl.experiments.common_args import (
    DEFAULT_MAX_GRAD_NORM,
    add_device_argument,
    add_experiment_name_argument,
    add_save_every_updates_argument,
    add_wandb_arguments,
)
from catanrl.models.model_builders import (
    build_critic_model,
    build_policy_model,
    build_policy_value_model,
)
from catanrl.algorithms.alphazero.trainer import AlphaZeroConfig, AlphaZeroTrainer
from catanrl.experiment_store import (
    GameConfig,
    KIND_POLICY,
    KIND_POLICY_VALUE,
    KIND_VALUE,
    TrainingWarmStart,
    add_load_from_experiment_arguments,
    default_checkpoints_dir,
    make_experiment_name,
    network_spec_from_model,
    resolve_training_architecture_and_warm_start,
    save_experiment,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train AlphaZero-style self-play agent for Catan",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--iterations", type=int, default=50, help="Outer self-play / SGD iterations"
    )
    parser.add_argument(
        "--games-per-iteration", type=int, default=16, help="Self-play games per iteration"
    )
    parser.add_argument(
        "--optimizer-steps", type=int, default=64, help="SGD mini-batches per iteration"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=min(16, os.cpu_count() or 4),
        help="Number of across-games self-play worker processes feeding a central batched "
        "inference server (the default strategy). 1 keeps the single-process trainer. "
        "Throughput on a single shared GPU saturates near the core count; oversubscribing "
        "past it trades latency for a little more aggregate throughput.",
    )
    add_config_argument(parser)
    parser.add_argument(
        "--inference-batch-size",
        type=int,
        default=64,
        help="Max batch size for the central self-play inference server.",
    )
    parser.add_argument(
        "--inference-wait-ms",
        type=float,
        default=2.0,
        help="Max time the inference server waits to fill a batch.",
    )
    parser.add_argument(
        "--prunning",
        action="store_true",
        help="Prune clearly dominated actions before MCTS expansion.",
    )

    parser.add_argument("--simulations", type=int, default=64, help="MCTS simulations per move")
    parser.add_argument("--c-puct", type=float, default=1.5, help="PUCT exploration constant")
    parser.add_argument(
        "--temperature", type=float, default=1.0, help="Sampling temperature for early moves"
    )
    parser.add_argument(
        "--final-temperature", type=float, default=0.1, help="Temperature after the drop move"
    )
    parser.add_argument(
        "--temperature-drop-move",
        type=int,
        default=30,
        help="Move index to drop exploration temperature",
    )
    parser.add_argument(
        "--noise-turns", type=int, default=20, help="Turns to inject Dirichlet noise at the root"
    )
    parser.add_argument(
        "--dirichlet-alpha",
        type=float,
        default=0.3,
        help="Dirichlet alpha for root exploration noise",
    )
    parser.add_argument(
        "--dirichlet-frac", type=float, default=0.25, help="Mixing fraction for exploration noise"
    )
    parser.add_argument("--buffer-size", type=int, default=50_000, help="Replay buffer size")
    parser.add_argument("--batch-size", type=int, default=256, help="Mini-batch size for SGD")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="Optimizer weight decay")
    parser.add_argument(
        "--policy-loss-weight", type=float, default=1.0, help="Policy loss multiplier"
    )
    parser.add_argument(
        "--value-loss-weight", type=float, default=1.0, help="Value loss multiplier"
    )
    parser.add_argument("--max-grad-norm", type=float, default=DEFAULT_MAX_GRAD_NORM, help="Gradient clipping norm")
    add_device_argument(parser)
    parser.add_argument("--seed", type=int, default=42, help="Random seed (set <0 for random)")

    # I/O
    add_save_every_updates_argument(
        parser,
        default=0,
        help="Save checkpoint every N iterations (0 to disable periodic saves; best model always saved)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Directory for periodic checkpoints (defaults to save-path directory)",
    )
    add_load_from_experiment_arguments(parser)

    add_experiment_name_argument(parser)
    add_wandb_arguments(parser)
    parser.add_argument(
        "--wandb-entity", type=str, default=None, help="Weights & Biases entity / team"
    )

    return parser.parse_args()



def aggregate_metrics(metrics: List[Dict[str, float]]) -> Dict[str, float]:
    if not metrics:
        return {}
    totals: Dict[str, float] = {key: 0.0 for key in metrics[0].keys()}
    for metric in metrics:
        for key, value in metric.items():
            totals[key] += value
    count = float(len(metrics))
    return {key: totals[key] / count for key in totals.keys()}


def maybe_load_from_experiment(
    trainer: AlphaZeroTrainer, warm_start: Optional[TrainingWarmStart]
) -> None:
    if warm_start is None:
        return
    ckpts = warm_start.checkpoints
    if trainer.uses_shared_network:
        trainer.load(ckpts.policy)
    else:
        if ckpts.critic is None:
            raise FileNotFoundError(
                f"Experiment '{ckpts.experiment_name}' has no critic checkpoint "
                f"for selector '{ckpts.which}'."
            )
        trainer.load(ckpts.policy, ckpts.critic)
    print("  Weights loaded successfully")


def build_train_config(
    args: argparse.Namespace,
    config: AlphaZeroConfig,
    arch: ArchitecturePreset,
    device: str,
) -> Dict[str, object]:
    return {
        "algorithm": "AlphaZero",
        **architecture_train_config_fields(arch),
        "iterations": args.iterations,
        "games_per_iteration": args.games_per_iteration,
        "optimizer_steps": args.optimizer_steps,
        "num_workers": args.num_workers,
        "inference_batch_size": args.inference_batch_size,
        "inference_wait_ms": args.inference_wait_ms,
        "num_players": config.num_players,
        "critic_hidden_mode": config.critic_hidden_mode,
        "simulations": config.simulations,
        "c_puct": config.c_puct,
        "prunning": config.prunning,
        "temperature": config.temperature,
        "final_temperature": config.final_temperature,
        "temperature_drop_move": config.temperature_drop_move,
        "noise_turns": config.noise_turns,
        "dirichlet_alpha": config.dirichlet_alpha,
        "dirichlet_frac": config.dirichlet_frac,
        "buffer_size": config.buffer_size,
        "batch_size": config.batch_size,
        "lr": config.lr,
        "weight_decay": config.weight_decay,
        "policy_loss_weight": config.policy_loss_weight,
        "value_loss_weight": config.value_loss_weight,
        "max_grad_norm": config.max_grad_norm,
        "save_every_updates": args.save_every_updates,
        "load_from_experiment": args.load_from_experiment,
        "load_from_which": args.load_from_which,
        "seed": args.seed,
        "device": device,
        "save_path": args.save_path,
    }


def init_wandb(args: argparse.Namespace, train_config: Dict[str, object]) -> bool:
    if args.wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            entity=args.wandb_entity,
            config=train_config,
        )
        return True
    wandb.init(mode="disabled")
    return False


def log_self_play_stats(stats: Dict[str, float], trainer: AlphaZeroTrainer) -> None:
    games = int(stats.get("games", 0))
    win_parts = []
    for color in trainer.colors:
        key = f"wins_{color.value}"
        win_parts.append(f"{color.name}:{int(stats.get(key, 0))}")
    win_summary = ", ".join(win_parts) if win_parts else "none"
    print(f"  Self-play: {games} games | Wins -> {win_summary}")


def _eval_win_rate(stats: Dict[str, float]) -> float:
    games = float(stats.get("games", 0))
    if games <= 0:
        return 0.0
    return float(stats.get("agent_wins", 0)) / games


def log_eval_stats(name: str, stats: Dict[str, float]) -> None:
    games = int(stats.get("games", 0))
    agent_wins = int(stats.get("agent_wins", 0))
    draws = int(stats.get("draws", 0))
    win_rate = _eval_win_rate(stats)
    print(
        f"  Eval vs {name}: win_rate={win_rate:.2%} "
        f"(agent wins {agent_wins}/{games}, draws={draws})"
    )


def _close_trainer_if_needed(trainer: AlphaZeroTrainer) -> None:
    close_fn = getattr(trainer, "close", None)
    if callable(close_fn):
        close_fn()


def run_training(args: argparse.Namespace, trainer: AlphaZeroTrainer, wandb_enabled: bool) -> None:
    checkpoint_dir = args.checkpoint_dir or args.save_path
    best_loss = float("inf")
    global_step = 0

    try:
        for iteration in range(1, args.iterations + 1):
            print("\n" + "=" * 80)
            print(f"Iteration {iteration}/{args.iterations}")
            stats = trainer.self_play(args.games_per_iteration)
            log_self_play_stats(stats, trainer)
            wandb.log(
                {
                    "selfplay/games": stats.get("games", 0),
                    **{f"selfplay/{k}": v for k, v in stats.items() if k != "games"},
                    "iteration": iteration,
                },
                step=global_step,
            )

            iteration_metrics: List[Dict[str, float]] = []
            skipped_steps = 0
            with tqdm(
                range(1, args.optimizer_steps + 1),
                desc="Training",
                leave=False,
            ) as train_bar:
                for step in train_bar:
                    metrics = trainer.update_weights()
                    if metrics is None:
                        skipped_steps += 1
                        train_bar.set_postfix({"skipped": skipped_steps})
                        continue
                    iteration_metrics.append(metrics)
                    global_step += 1
                    train_bar.set_postfix(
                        {
                            "loss": f"{metrics['loss']:.3f}",
                            "policy": f"{metrics['policy_loss']:.3f}",
                            "value": f"{metrics['value_loss']:.3f}",
                        }
                    )
                    wandb.log(
                        {
                            "train/loss": metrics["loss"],
                            "train/policy_loss": metrics["policy_loss"],
                            "train/value_loss": metrics["value_loss"],
                            "iteration": iteration,
                            "optimizer_step": step,
                        },
                        step=global_step,
                    )

            if skipped_steps:
                print(f"  Skipped {skipped_steps} optimizer steps (buffer warming up).")

            summary = aggregate_metrics(iteration_metrics)
            if summary:
                print(
                    "  SGD metrics | "
                    f"loss: {summary['loss']:.4f} | "
                    f"policy: {summary['policy_loss']:.4f} | "
                    f"value: {summary['value_loss']:.4f}"
                )
                wandb.log(
                    {
                        "iteration/loss": summary["loss"],
                        "iteration/policy_loss": summary["policy_loss"],
                        "iteration/value_loss": summary["value_loss"],
                        "iteration": iteration,
                    },
                    step=global_step,
                )
                if summary["loss"] < best_loss:
                    best_loss = summary["loss"]
                    trainer.save(args.save_path, stem="best")
                    print(
                        f"  → Saved new best model to {args.save_path} "
                        f"({trainer.uses_shared_network and 'policy_value_best.pt' or 'policy_best.pt, critic_best.pt'}; "
                        f"loss={best_loss:.4f})"
                    )
                    if wandb_enabled:
                        wandb.run.summary["best_loss"] = best_loss
            else:
                print("  No optimizer metrics collected this iteration.")

            if args.save_every_updates and iteration % args.save_every_updates == 0:
                trainer.save(checkpoint_dir, stem=f"iter_{iteration}")
                print(
                    f"  → Saved checkpoint to {checkpoint_dir} "
                    f"({trainer.uses_shared_network and f'policy_value_iter_{iteration}.pt' or f'policy_iter_{iteration}.pt, critic_iter_{iteration}.pt'})"
                )

            value_eval_stats = trainer.evaluate_against(
                opponent_factory=lambda color: ValueFunctionPlayer(color),
                num_games=100,
                desc="Eval vs ValueFunctionPlayer",
            )
            # alphabeta_eval_stats = trainer.evaluate_against(
            #     opponent_factory=lambda color: AlphaBetaPlayer(color, depth=2, prunning=True),
            #     num_games=25,
            #     desc="Eval vs AlphaBetaPlayer",
            # )
            log_eval_stats("ValueFunctionPlayer", value_eval_stats)
            # log_eval_stats("AlphaBetaPlayer", alphabeta_eval_stats)
            wandb.log(
                {
                    "eval/value_function/win_rate": _eval_win_rate(value_eval_stats),
                    "eval/value_function/agent_wins": value_eval_stats.get("agent_wins", 0),
                    "eval/value_function/draws": value_eval_stats.get("draws", 0),
                    # "eval/alphabeta/win_rate": _eval_win_rate(alphabeta_eval_stats),
                    # "eval/alphabeta/agent_wins": alphabeta_eval_stats.get("agent_wins", 0),
                    # "eval/alphabeta/draws": alphabeta_eval_stats.get("draws", 0),
                    "iteration": iteration,
                },
                step=global_step,
            )

        if best_loss == float("inf"):
            trainer.save(args.save_path, stem="best")
            print(
                f"\nTraining finished without SGD metrics; saved model to {args.save_path} "
                f"({trainer.uses_shared_network and 'policy_value_best.pt' or 'policy_best.pt, critic_best.pt'})"
            )
        else:
            print(f"\nBest loss achieved: {best_loss:.4f}")
    finally:
        _close_trainer_if_needed(trainer)


def main() -> None:
    args = parse_args()

    try:
        setup = resolve_training_architecture_and_warm_start(args)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}")
        return

    arch = setup.arch
    warm_start = setup.warm_start
    if arch.num_players is None:
        print("Error: AlphaZero training requires game.num_players in the architecture preset.")
        return

    experiment_name = make_experiment_name("alphazero", args.wandb_run_name, args.experiment_name)
    if args.wandb and not args.wandb_run_name:
        args.wandb_run_name = experiment_name
    args.save_path = default_checkpoints_dir(experiment_name)
    os.makedirs(args.save_path, exist_ok=True)
    print(f"Architecture: {setup.architecture_source}")

    config = AlphaZeroConfig(
        num_players=arch.num_players,
        map_type=arch.map_type,
        actor_observation_level=arch.actor_observation_level,
        critic_observation_level=arch.critic_observation_level,
        critic_hidden_mode=arch.critic_hidden_mode or "full",
        network_mode=arch.network_mode,
        model_type=arch.model_type,
        vps_to_win=arch.vps_to_win,
        discard_limit=arch.discard_limit,
        simulations=args.simulations,
        c_puct=args.c_puct,
        prunning=args.prunning,
        temperature=args.temperature,
        final_temperature=args.final_temperature,
        temperature_drop_move=args.temperature_drop_move,
        noise_turns=args.noise_turns,
        dirichlet_alpha=args.dirichlet_alpha,
        dirichlet_frac=args.dirichlet_frac,
        num_game_workers=args.num_workers,
        inference_batch_size=args.inference_batch_size,
        inference_wait_ms=args.inference_wait_ms,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        policy_loss_weight=args.policy_loss_weight,
        value_loss_weight=args.value_loss_weight,
        max_grad_norm=args.max_grad_norm,
        device=args.device,
        seed=None if args.seed < 0 else args.seed,
    )

    device = config.device or ("cuda" if torch.cuda.is_available() else "cpu")
    critic_model = None
    if arch.network_mode == "shared":
        policy_model = build_policy_value_model(
            backbone_type=arch.backbone_type,
            model_type=config.model_type,
            hidden_dims=arch.policy_hidden_dims,
            num_players=config.num_players,
            map_type=config.map_type,
            actor_observation_level=config.actor_observation_level,
            critic_observation_level=config.critic_observation_level,
            device=device,
            network_mode=arch.network_mode,
            xdim_cnn_channels=arch.xdim_cnn_channels,
            xdim_cnn_kernel_size=arch.xdim_cnn_kernel_size,
            xdim_fusion_hidden_dim=arch.xdim_policy_fusion_hidden_dim,
        )
    else:
        policy_model = build_policy_model(
            backbone_type=arch.backbone_type,
            model_type=config.model_type,
            hidden_dims=arch.policy_hidden_dims,
            num_players=config.num_players,
            map_type=config.map_type,
            actor_observation_level=config.actor_observation_level,
            device=device,
            xdim_cnn_channels=arch.xdim_cnn_channels,
            xdim_cnn_kernel_size=arch.xdim_cnn_kernel_size,
            xdim_fusion_hidden_dim=arch.xdim_policy_fusion_hidden_dim,
        )
        critic_model = build_critic_model(
            backbone_type=arch.backbone_type,
            hidden_dims=arch.critic_hidden_dims,
            num_players=config.num_players,
            map_type=config.map_type,
            critic_observation_level=config.critic_observation_level,
            device=device,
            xdim_cnn_channels=arch.xdim_cnn_channels,
            xdim_cnn_kernel_size=arch.xdim_cnn_kernel_size,
            xdim_fusion_hidden_dim=arch.xdim_critic_fusion_hidden_dim,
        )

    trainer = AlphaZeroTrainer(config=config, policy_model=policy_model, critic_model=critic_model)
    maybe_load_from_experiment(trainer, warm_start)

    train_config = build_train_config(args, config, arch, device)
    wandb_enabled = init_wandb(args, train_config)
    print("\nStarting AlphaZero training...")
    print(f"  Device: {trainer.device}")
    print(f"  Backbone: {arch.backbone_type} | model: {config.model_type}")
    print(f"  Policy hidden dims: {list(arch.policy_hidden_dims)}")
    print(f"  Critic hidden dims: {list(arch.critic_hidden_dims)}")
    print(f"  Checkpoints: {args.save_path}")
    print(f"  Network mode: {arch.network_mode}")

    run_training(args, trainer, wandb_enabled)

    print(f"  Game workers: {args.num_workers}")

    if trainer.uses_shared_network:
        networks = {
            "policy": network_spec_from_model(
                policy_model,
                kind=KIND_POLICY_VALUE,
                model_type=config.model_type,
                observation_level=config.actor_observation_level,
            ),
        }
    else:
        networks = {
            "policy": network_spec_from_model(
                policy_model,
                kind=KIND_POLICY,
                model_type=config.model_type,
                observation_level=config.actor_observation_level,
            ),
            "critic": network_spec_from_model(
                critic_model,
                kind=KIND_VALUE,
                observation_level=config.critic_observation_level,
            ),
        }
    exp_path = save_experiment(
        experiment_name,
        args.save_path,
        algorithm="alphazero",
        game=GameConfig(
            num_players=config.num_players,
            map_type=config.map_type,
            vps_to_win=config.vps_to_win,
            discard_limit=config.discard_limit,
        ),
        networks=networks,
        train_config=train_config,
        wandb_info={"project": args.wandb_project, "name": args.wandb_run_name} if args.wandb else {},
    )

    print("\n" + "=" * 80)
    print("AlphaZero training complete!")
    print("=" * 80)
    print(f"Checkpoints saved to: {args.save_path}")
    if trainer.uses_shared_network:
        print(f"  - Policy-value: {args.save_path}/policy_value_best.pt")
    else:
        print(f"  - Policy: {args.save_path}/policy_best.pt")
        print(f"  - Critic: {args.save_path}/critic_best.pt")
    if exp_path:
        print(f"Experiment:  {exp_path}  (load via load_experiment('{experiment_name}'))")

    if args.wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
