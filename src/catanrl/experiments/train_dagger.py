import argparse
import os

import wandb
from catanatron.gym.envs.catanatron_env import ACTION_SPACE_SIZE, COLOR_ORDER
from catanatron.game import Game
from catanatron.models.map import build_map
from catanatron.models.player import Color, RandomPlayer

from ..features.catanatron_utils import game_to_features
from ..envs.single_env import create_opponents
from ..algorithms.imitation_learning.dagger import train as dagger_train


def _compute_input_dim(num_players: int, map_type: str) -> int:
    """Create a deterministic dummy game to infer feature dimensionality."""
    dummy_players = [RandomPlayer(color) for color in COLOR_ORDER[:num_players]]
    dummy_game = Game(dummy_players, catan_map=build_map(map_type))
    dummy_tensor = game_to_features(dummy_game, Color.RED, num_players, map_type)
    return dummy_tensor.shape[0]


def main():
    parser = argparse.ArgumentParser(
        description="Train a Catan policy-value network with DAgger imitation learning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["flat", "hierarchical"],
        default="flat",
        help="Model architecture to train (default: flat)",
    )
    parser.add_argument(
        "--hidden-dims",
        type=str,
        default="512,512",
        help="Comma-separated hidden layer sizes (default: 512,512)",
    )
    parser.add_argument(
        "--load-weights",
        type=str,
        default=None,
        help="Optional initialization checkpoint",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default=None,
        help="Where to store the best checkpoint (default: weights/policy_value_dagger.pt)",
    )
    parser.add_argument(
        "--iterations", type=int, default=10, help="Number of DAgger iterations (default: 10)"
    )
    parser.add_argument(
        "--episodes-per-iter",
        type=int,
        default=5,
        help="Episodes to collect per DAgger iteration (default: 5)",
    )
    parser.add_argument(
        "--train-epochs",
        type=int,
        default=1,
        help="Supervised epochs over the aggregated dataset each iteration (default: 1)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=256, help="Supervised batch size (default: 256)"
    )
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate (default: 1e-4)")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor (default: 0.99)")
    parser.add_argument(
        "--expert",
        type=str,
        default="AB:2",
        help="Expert policy spec (e.g. AB:3, MCTS:200, random). Default: AB:2",
    )
    parser.add_argument(
        "--opponents",
        type=str,
        nargs="+",
        default=["random"],
        help="Opponent bot configs for the environment (default: random)",
    )
    parser.add_argument(
        "--map-type",
        type=str,
        default="BASE",
        choices=["BASE", "MINI", "TOURNAMENT"],
        help="Board template to use (default: BASE)",
    )
    parser.add_argument(
        "--beta-init",
        type=float,
        default=1.0,
        help="Initial probability of executing expert actions (default: 1.0)",
    )
    parser.add_argument(
        "--beta-decay",
        type=float,
        default=0.9,
        help="Multiplicative decay applied to beta after each iteration (default: 0.9)",
    )
    parser.add_argument(
        "--beta-min",
        type=float,
        default=0.05,
        help="Lower bound for beta (default: 0.05)",
    )
    parser.add_argument(
        "--max-dataset-size",
        type=int,
        default=None,
        help="Optional cap on aggregated samples (keep last N). Default: unlimited",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=5,
        help="Episodes for evaluation rollout after each iteration (default: 5)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device override (cpu, cuda, cuda:1, ...). Default: auto",
    )
    parser.add_argument("--seed", type=int, default=42, help="RNG seed (default: 42)")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="catan-rl",
        help="Weights & Biases project (default: catan-rl)",
    )
    parser.add_argument(
        "--wandb-run-name",
        type=str,
        default=None,
        help="Weights & Biases run name (default: auto-generated)",
    )

    args = parser.parse_args()

    if args.load_weights and not os.path.exists(args.load_weights):
        print(f"Error: weights file '{args.load_weights}' not found")
        return

    if args.save_path is None:
        if args.wandb and args.wandb_run_name:
            args.save_path = f"weights/{args.wandb_run_name}/dagger.pt"
        else:
            args.save_path = "weights/policy_value_dagger.pt"

    save_dir = os.path.dirname(args.save_path)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    if not args.opponents:
        args.opponents = ["random"]
    opponents = create_opponents(args.opponents)
    num_players = len(opponents) + 1
    input_dim = _compute_input_dim(num_players, args.map_type)
    print(f"Number of players: {num_players}")
    print(f"Feature dimension: {input_dim}")

    hidden_dims = [int(dim.strip()) for dim in args.hidden_dims.split(",") if dim.strip()]
    wandb_config = None
    if args.wandb:
        wandb_config = {
            "project": args.wandb_project,
            "name": args.wandb_run_name,
            "config": {
                "algorithm": "DAgger",
                "model_type": args.model_type,
                "hidden_dims": hidden_dims,
                "iterations": args.iterations,
                "episodes_per_iteration": args.episodes_per_iter,
                "train_epochs": args.train_epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "gamma": args.gamma,
                "expert": args.expert,
                "opponents": args.opponents,
                "map_type": args.map_type,
                "beta_init": args.beta_init,
                "beta_decay": args.beta_decay,
                "beta_min": args.beta_min,
                "max_dataset_size": args.max_dataset_size,
                "eval_episodes": args.eval_episodes,
                "seed": args.seed,
            },
        }

    dagger_train(
        input_dim=input_dim,
        num_actions=ACTION_SPACE_SIZE,
        model_type=args.model_type,
        hidden_dims=hidden_dims,
        load_weights=args.load_weights,
        n_iterations=args.iterations,
        episodes_per_iteration=args.episodes_per_iter,
        train_epochs=args.train_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        gamma=args.gamma,
        expert_config=args.expert,
        opponent_configs=args.opponents,
        map_type=args.map_type,
        beta_init=args.beta_init,
        beta_decay=args.beta_decay,
        beta_min=args.beta_min,
        max_dataset_size=args.max_dataset_size,
        save_path=args.save_path,
        device=args.device,
        wandb_config=wandb_config,
        eval_episodes=args.eval_episodes,
        seed=args.seed,
    )

    print("\n" + "=" * 60)
    print("DAgger training finished")
    print("=" * 60)
    print(f"Best checkpoint saved to: {args.save_path}")

    if args.wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
