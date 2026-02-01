import argparse
import os

import wandb
from catanatron.gym.envs.catanatron_env import ACTION_SPACE_SIZE

from ..envs.gym.single_env import create_opponents
from ..algorithms.imitation_learning.dagger import train as dagger_train
from ..algorithms.imitation_learning.dataset import EvictionStrategy


def main():
    parser = argparse.ArgumentParser(
        description="Train Catan policy and critic networks with DAgger imitation learning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Model architecture
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["flat", "hierarchical"],
        default="flat",
        help="Policy model architecture (default: flat)",
    )
    parser.add_argument(
        "--backbone-type",
        type=str,
        choices=["mlp", "xdim"],
        default="mlp",
        help="Backbone architecture: 'mlp' or 'xdim' (cross-dimensional with CNN) (default: mlp)",
    )
    parser.add_argument(
        "--policy-hidden-dims",
        type=str,
        default="512,512",
        help="Comma-separated hidden layer sizes for policy network (default: 512,512)",
    )
    parser.add_argument(
        "--critic-hidden-dims",
        type=str,
        default="512,512",
        help="Comma-separated hidden layer sizes for critic network (default: 512,512)",
    )

    # Weight loading
    parser.add_argument(
        "--load-policy-weights",
        type=str,
        default=None,
        help="Optional policy initialization checkpoint",
    )
    parser.add_argument(
        "--load-critic-weights",
        type=str,
        default=None,
        help="Optional critic initialization checkpoint",
    )

    # Saving
    parser.add_argument(
        "--save-path",
        type=str,
        default=None,
        help="Directory to store checkpoints (default: weights/dagger)",
    )

    # Training loop
    parser.add_argument(
        "--iterations",
        type=int,
        default=10,
        help="Number of DAgger iterations (default: 10)",
    )
    parser.add_argument(
        "--steps-per-iter",
        type=int,
        default=2048,
        help="Environment steps to collect per iteration (default: 2048)",
    )
    parser.add_argument(
        "--train-epochs",
        type=int,
        default=1,
        help="Training epochs per iteration (default: 1)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size for training (default: 256)",
    )

    # Learning rates
    parser.add_argument(
        "--policy-lr",
        type=float,
        default=1e-4,
        help="Policy learning rate (default: 1e-4)",
    )
    parser.add_argument(
        "--critic-lr",
        type=float,
        default=1e-4,
        help="Critic learning rate (default: 1e-4)",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="Discount factor (default: 0.99)",
    )

    # DAgger-specific
    parser.add_argument(
        "--expert",
        type=str,
        default="F",
        help="Expert policy spec (e.g. AB:3, MCTS:200, random). Default: AB:2",
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
        help="Multiplicative decay for beta per iteration (default: 0.9)",
    )
    parser.add_argument(
        "--beta-min",
        type=float,
        default=0.05,
        help="Minimum beta value (default: 0.05)",
    )
    parser.add_argument(
        "--max-dataset-size",
        type=int,
        default=819200,
        help="Maximum samples in replay buffer (default: 819200)",
    )
    parser.add_argument(
        "--eviction-strategy",
        type=str,
        choices=["random", "fifo", "correct"],
        default="random",
        help="Eviction strategy when dataset is full: random, fifo, or correct (default: random)",
    )

    # Environment
    parser.add_argument(
        "--opponents",
        type=str,
        nargs="+",
        default=["F"],
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
        "--num-envs",
        type=int,
        default=4,
        help="Number of parallel environments (default: 4)",
    )
    parser.add_argument(
        "--reward-function",
        type=str,
        default="shaped",
        choices=["shaped", "win"],
        help="Reward function type (default: shaped)",
    )

    # Evaluation
    parser.add_argument(
        "--eval-games-per-opponent",
        type=int,
        default=1000,
        help="Games to play against each baseline opponent for evaluation (default: 1000)",
    )

    # Misc
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device (cpu, cuda, cuda:1, ...). Default: auto",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=5.0,
        help="Maximum gradient norm for clipping (default: 5.0)",
    )

    # Weights & Biases
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases logging",
    )
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

    # Validate weight paths if provided
    if args.load_policy_weights and not os.path.exists(args.load_policy_weights):
        print(f"Error: policy weights file '{args.load_policy_weights}' not found")
        return
    if args.load_critic_weights and not os.path.exists(args.load_critic_weights):
        print(f"Error: critic weights file '{args.load_critic_weights}' not found")
        return

    # Set save path
    if args.save_path is None:
        if args.wandb and args.wandb_run_name:
            args.save_path = f"weights/{args.wandb_run_name}"
        else:
            args.save_path = "weights/dagger"

    os.makedirs(args.save_path, exist_ok=True)

    # Parse hidden dims
    if not args.opponents:
        args.opponents = ["random"]
    opponents = create_opponents(args.opponents)
    num_players = len(opponents) + 1
    print(f"Number of players: {num_players}")

    policy_hidden_dims = [
        int(dim.strip()) for dim in args.policy_hidden_dims.split(",") if dim.strip()
    ]
    critic_hidden_dims = [
        int(dim.strip()) for dim in args.critic_hidden_dims.split(",") if dim.strip()
    ]

    # Setup W&B
    wandb_config = None
    if args.wandb:
        wandb_config = {
            "project": args.wandb_project,
            "name": args.wandb_run_name,
            "config": {
                "algorithm": "DAgger",
                "model_type": args.model_type,
                "backbone_type": args.backbone_type,
                "policy_hidden_dims": policy_hidden_dims,
                "critic_hidden_dims": critic_hidden_dims,
                "iterations": args.iterations,
                "steps_per_iteration": args.steps_per_iter,
                "train_epochs": args.train_epochs,
                "batch_size": args.batch_size,
                "policy_lr": args.policy_lr,
                "critic_lr": args.critic_lr,
                "gamma": args.gamma,
                "expert": args.expert,
                "opponents": args.opponents,
                "map_type": args.map_type,
                "num_envs": args.num_envs,
                "reward_function": args.reward_function,
                "beta_init": args.beta_init,
                "beta_decay": args.beta_decay,
                "beta_min": args.beta_min,
                "max_dataset_size": args.max_dataset_size,
                "eviction_strategy": args.eviction_strategy,
                "eval_games_per_opponent": args.eval_games_per_opponent,
                "seed": args.seed,
            },
        }

    # Run training
    policy_model, critic_model = dagger_train(
        num_actions=ACTION_SPACE_SIZE,
        model_type=args.model_type,
        backbone_type=args.backbone_type,
        policy_hidden_dims=policy_hidden_dims,
        critic_hidden_dims=critic_hidden_dims,
        load_policy_weights=args.load_policy_weights,
        load_critic_weights=args.load_critic_weights,
        n_iterations=args.iterations,
        steps_per_iteration=args.steps_per_iter,
        train_epochs=args.train_epochs,
        batch_size=args.batch_size,
        policy_lr=args.policy_lr,
        critic_lr=args.critic_lr,
        gamma=args.gamma,
        expert_config=args.expert,
        opponent_configs=args.opponents,
        map_type=args.map_type,
        beta_init=args.beta_init,
        beta_decay=args.beta_decay,
        beta_min=args.beta_min,
        max_dataset_size=args.max_dataset_size,
        eviction_strategy=EvictionStrategy(args.eviction_strategy),
        save_path=args.save_path,
        device=args.device,
        wandb_config=wandb_config,
        eval_games_per_opponent=args.eval_games_per_opponent,
        seed=args.seed,
        num_envs=args.num_envs,
        reward_function=args.reward_function,
        max_grad_norm=args.max_grad_norm,
    )

    print("\n" + "=" * 60)
    print("DAgger training finished")
    print("=" * 60)
    print(f"Checkpoints saved to: {args.save_path}")
    print(f"  - Policy: {args.save_path}/policy_best.pt")
    print(f"  - Critic: {args.save_path}/critic_best.pt")

    if args.wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
