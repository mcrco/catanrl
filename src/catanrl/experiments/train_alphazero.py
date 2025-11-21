#!/usr/bin/env python3
"""
Experiment script for training the AlphaZero-style agent.

This mirrors the structure we use for other RL entrypoints (e.g., PPO) and
provides a convenient CLI for self-play + neural training loops.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import asdict
from typing import Dict, List, Optional

import torch
import wandb
from tqdm import tqdm

from catanatron.gym.envs.catanatron_env import ACTION_SPACE_SIZE
from catanatron.players.minimax import AlphaBetaPlayer
from catanatron.players.value import ValueFunctionPlayer

from catanrl.algorithms.alphazero.trainer import AlphaZeroConfig, AlphaZeroTrainer
from catanrl.algorithms.alphazero.parallel_trainer import ParallelAlphaZeroTrainer
from catanrl.data.data_utils import compute_feature_vector_dim
from catanrl.models.models import HierarchicalPolicyValueNetwork, PolicyValueNetwork


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train AlphaZero-style self-play agent for Catan",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Training loop
    parser.add_argument("--iterations", type=int, default=50, help="Outer self-play / SGD iterations")
    parser.add_argument("--games-per-iteration", type=int, default=16, help="Self-play games per iteration")
    parser.add_argument("--optimizer-steps", type=int, default=64, help="SGD mini-batches per iteration")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of parallel self-play workers (1 keeps the single-process trainer)",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="hierarchical",
        choices=["standard", "hierarchical"],
        help="Which policy/value network architecture to use.",
    )

    # Model + config knobs (mirrors AlphaZeroConfig)
    parser.add_argument("--num-players", type=int, default=2, choices=[2, 3, 4], help="Number of players in self-play")
    parser.add_argument("--map-type", type=str, default="BASE", choices=["BASE", "MINI", "TOURNAMENT"], help="Map layout")
    parser.add_argument("--vps-to-win", type=int, default=10, help="Victory points to win the game")
    parser.add_argument("--simulations", type=int, default=64, help="MCTS simulations per move")
    parser.add_argument("--c-puct", type=float, default=1.5, help="PUCT exploration constant")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature for early moves")
    parser.add_argument("--final-temperature", type=float, default=0.1, help="Temperature after the drop move")
    parser.add_argument("--temperature-drop-move", type=int, default=30, help="Move index to drop exploration temperature")
    parser.add_argument("--noise-turns", type=int, default=20, help="Turns to inject Dirichlet noise at the root")
    parser.add_argument("--dirichlet-alpha", type=float, default=0.3, help="Dirichlet alpha for root exploration noise")
    parser.add_argument("--dirichlet-frac", type=float, default=0.25, help="Mixing fraction for exploration noise")
    parser.add_argument("--buffer-size", type=int, default=50_000, help="Replay buffer size")
    parser.add_argument("--batch-size", type=int, default=256, help="Mini-batch size for SGD")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="Optimizer weight decay")
    parser.add_argument("--policy-loss-weight", type=float, default=1.0, help="Policy loss multiplier")
    parser.add_argument("--value-loss-weight", type=float, default=1.0, help="Value loss multiplier")
    parser.add_argument("--max-grad-norm", type=float, default=1.0, help="Gradient clipping norm")
    parser.add_argument("--hidden-dims", type=str, default="512,512", help="Comma-separated hidden layer sizes")
    parser.add_argument("--device", type=str, default=None, help="Device override (cpu, cuda, cuda:0, ...)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (set <0 for random)")

    # I/O
    parser.add_argument("--save-path", type=str, default="weights/alphazero/best.pt", help="Path to save the best model")
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=0,
        help="Save checkpoint every N iterations (0 to disable)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Directory for periodic checkpoints (defaults to save-path directory)",
    )
    parser.add_argument("--load-weights", type=str, default=None, help="Optional path to initialize model weights")

    # Logging
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb-project", type=str, default="catan-rl", help="Weights & Biases project name")
    parser.add_argument("--wandb-run-name", type=str, default=None, help="Weights & Biases run name")
    parser.add_argument("--wandb-entity", type=str, default=None, help="Weights & Biases entity / team")

    return parser.parse_args()


def parse_hidden_dims(value: str) -> List[int]:
    dims = []
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        dim = int(token)
        if dim <= 0:
            raise ValueError("Hidden dimensions must be positive integers.")
        dims.append(dim)
    if not dims:
        raise ValueError("At least one hidden dimension is required.")
    return dims


def ensure_parent_dir(path: str) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)


def aggregate_metrics(metrics: List[Dict[str, float]]) -> Dict[str, float]:
    if not metrics:
        return {}
    totals: Dict[str, float] = {key: 0.0 for key in metrics[0].keys()}
    for metric in metrics:
        for key, value in metric.items():
            totals[key] += value
    count = float(len(metrics))
    return {key: totals[key] / count for key in totals.keys()}


def maybe_load_weights(trainer: AlphaZeroTrainer, path: Optional[str]) -> None:
    if not path:
        return
    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not find weights file: {path}")
    map_location = trainer.device
    if map_location is None:
        map_location = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading weights from {path} (map_location={map_location})")
    state_dict = torch.load(path, map_location=map_location)
    trainer.model.load_state_dict(state_dict)
    print("  ✓ Weights loaded successfully")


def init_wandb(args: argparse.Namespace, config: AlphaZeroConfig) -> bool:
    config_dict = {
        **asdict(config),
        "iterations": args.iterations,
        "games_per_iteration": args.games_per_iteration,
        "optimizer_steps": args.optimizer_steps,
        "hidden_dims": args.hidden_dims,
        "save_path": args.save_path,
        "checkpoint_every": args.checkpoint_every,
    }
    if args.wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            entity=args.wandb_entity,
            config=config_dict,
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
    ensure_parent_dir(args.save_path)
    checkpoint_dir = args.checkpoint_dir or os.path.dirname(args.save_path) or "."
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
                    trainer.save(args.save_path)
                    print(f"  → Saved new best model to {args.save_path} (loss={best_loss:.4f})")
                    if wandb_enabled:
                        wandb.run.summary["best_loss"] = best_loss
            else:
                print("  No optimizer metrics collected this iteration.")

            if args.checkpoint_every and iteration % args.checkpoint_every == 0:
                os.makedirs(checkpoint_dir, exist_ok=True)
                checkpoint_path = os.path.join(checkpoint_dir, f"alphazero_iter{iteration:04d}.pt")
                trainer.save(checkpoint_path)
                print(f"  → Saved checkpoint to {checkpoint_path}")

            value_eval_stats = trainer.evaluate_against(
                opponent_factory=lambda color: ValueFunctionPlayer(color),
                num_games=50,
                desc="Eval vs ValueFunctionPlayer",
            )
            alphabeta_eval_stats = trainer.evaluate_against(
                opponent_factory=lambda color: AlphaBetaPlayer(color, depth=2, prunning=True),
                num_games=25,
                desc="Eval vs AlphaBetaPlayer",
            )
            log_eval_stats("ValueFunctionPlayer", value_eval_stats)
            log_eval_stats("AlphaBetaPlayer", alphabeta_eval_stats)
            wandb.log(
                {
                    "eval/value_function/win_rate": _eval_win_rate(value_eval_stats),
                    "eval/value_function/agent_wins": value_eval_stats.get("agent_wins", 0),
                    "eval/value_function/draws": value_eval_stats.get("draws", 0),
                    "eval/alphabeta/win_rate": _eval_win_rate(alphabeta_eval_stats),
                    "eval/alphabeta/agent_wins": alphabeta_eval_stats.get("agent_wins", 0),
                    "eval/alphabeta/draws": alphabeta_eval_stats.get("draws", 0),
                    "iteration": iteration,
                },
                step=global_step,
            )

        if best_loss == float("inf"):
            trainer.save(args.save_path)
            print(f"\nTraining finished without SGD metrics; saved model to {args.save_path}")
        else:
            print(f"\nBest loss achieved: {best_loss:.4f}")
    finally:
        _close_trainer_if_needed(trainer)


def main() -> None:
    args = parse_args()
    hidden_dims = parse_hidden_dims(args.hidden_dims)

    config = AlphaZeroConfig(
        num_players=args.num_players,
        map_type=args.map_type,
        vps_to_win=args.vps_to_win,
        simulations=args.simulations,
        c_puct=args.c_puct,
        temperature=args.temperature,
        final_temperature=args.final_temperature,
        temperature_drop_move=args.temperature_drop_move,
        noise_turns=args.noise_turns,
        dirichlet_alpha=args.dirichlet_alpha,
        dirichlet_frac=args.dirichlet_frac,
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

    input_dim = compute_feature_vector_dim(config.num_players, config.map_type)
    if args.model_type == "hierarchical":
        model = HierarchicalPolicyValueNetwork(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
        )
    else:
        model = PolicyValueNetwork(
            input_dim=input_dim,
            num_actions=ACTION_SPACE_SIZE,
            hidden_dims=hidden_dims,
        )

    if args.num_workers <= 1:
        trainer = AlphaZeroTrainer(config=config, model=model)
    else:
        trainer = ParallelAlphaZeroTrainer(config=config, model=model, num_workers=args.num_workers)
    maybe_load_weights(trainer, args.load_weights)

    wandb_enabled = init_wandb(args, config)
    print("\nStarting AlphaZero training...")
    print(f"  Device: {trainer.device}")
    print(f"  Input dim: {input_dim}")
    print(f"  Hidden dims: {hidden_dims}")
    print(f"  Save path: {args.save_path}")
    print(f"  Num workers: {args.num_workers}")
    print(f"  Model type: {args.model_type}")

    run_training(args, trainer, wandb_enabled)

    print("\n" + "=" * 80)
    print("AlphaZero training complete!")
    print("=" * 80)

    if args.wandb:
        wandb.finish()


if __name__ == "__main__":
    main()

