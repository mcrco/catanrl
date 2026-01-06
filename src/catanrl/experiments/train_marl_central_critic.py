import argparse
import os
import wandb
from catanrl.algorithms.ppo.marl_ppo_central_critic import train


def main():
    parser = argparse.ArgumentParser(
        description="Train a multi-agent self-play PPO policy with centralized critic using the PettingZoo Catanatron env",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--num-players", type=int, default=2, choices=[2, 3, 4])
    parser.add_argument("--map-type", type=str, default="BASE", choices=["BASE", "MINI"])
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["flat", "hierarchical"],
        default="flat",
        help="Model architecture type: flat or hierarchical (default: flat)",
    )
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--rollout-steps", type=int, default=4096)
    parser.add_argument(
        "--policy-lr", type=float, default=3e-4, help="Learning rate for policy network"
    )
    parser.add_argument(
        "--critic-lr", type=float, default=3e-4, help="Learning rate for critic network"
    )
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-epsilon", type=float, default=0.2)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--policy-hidden-dims",
        type=str,
        default="512,512",
        help="Hidden dimensions for policy network (comma-separated)",
    )
    parser.add_argument(
        "--critic-hidden-dims",
        type=str,
        default="512,512",
        help="Hidden dimensions for critic network (comma-separated)",
    )
    parser.add_argument("--save-path", type=str, default="weights/policy_value_marl_central.pt")
    parser.add_argument("--save-freq", type=int, default=50)
    parser.add_argument(
        "--eval-freq",
        type=int,
        default=None,
        help="How often (in episodes) to run evaluation vs baselines. Defaults to --save-freq.",
    )
    parser.add_argument(
        "--eval-games",
        type=int,
        default=50,
        help="Number of evaluation games per opponent (RandomPlayer and ValueFunctionPlayer).",
    )
    parser.add_argument(
        "--load-policy-weights",
        type=str,
        default=None,
        help="Path to pre-trained policy weights to continue training from",
    )
    parser.add_argument(
        "--load-critic-weights",
        type=str,
        default=None,
        help="Path to pre-trained critic weights to continue training from",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument(
        "--deterministic-policy",
        action="store_true",
        help="Use greedy action selection during data collection (default: sample)",
    )
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb-project", type=str, default="catan-marl")
    parser.add_argument("--wandb-run-name", type=str, default=None)

    args = parser.parse_args()

    # Check if weight files exist
    if args.load_policy_weights and not os.path.exists(args.load_policy_weights):
        print(f"Error: Policy weights file '{args.load_policy_weights}' not found!")
        return
    if args.load_critic_weights and not os.path.exists(args.load_critic_weights):
        print(f"Error: Critic weights file '{args.load_critic_weights}' not found!")
        return

    # Create save directory if needed
    if args.save_path:
        save_dir = os.path.dirname(args.save_path)
        if save_dir and not os.path.exists(save_dir):
            print(f"Creating directory {save_dir}")
            os.makedirs(save_dir, exist_ok=True)

    # Parse hidden dimensions
    policy_hidden_dims = [int(dim) for dim in args.policy_hidden_dims.split(",") if dim.strip()]
    critic_hidden_dims = [int(dim) for dim in args.critic_hidden_dims.split(",") if dim.strip()]

    # Prepare wandb config
    wandb_config = None
    if args.wandb:
        config_dict = {
            "algorithm": "PPO_Central_Critic",
            "episodes": args.episodes,
            "rollout_steps": args.rollout_steps,
            "policy_lr": args.policy_lr,
            "critic_lr": args.critic_lr,
            "gamma": args.gamma,
            "gae_lambda": args.gae_lambda,
            "clip_epsilon": args.clip_epsilon,
            "value_coef": args.value_coef,
            "entropy_coef": args.entropy_coef,
            "ppo_epochs": args.ppo_epochs,
            "batch_size": args.batch_size,
            "policy_hidden_dims": policy_hidden_dims,
            "critic_hidden_dims": critic_hidden_dims,
            "num_players": args.num_players,
            "map_type": args.map_type,
            "model_type": args.model_type,
            "max_grad_norm": args.max_grad_norm,
            "deterministic_policy": args.deterministic_policy,
        }
        wandb_config = {
            "project": args.wandb_project,
            "name": args.wandb_run_name,
            "config": config_dict,
        }

    # Train model
    train(
        num_players=args.num_players,
        map_type=args.map_type,
        model_type=args.model_type,
        episodes=args.episodes,
        rollout_steps=args.rollout_steps,
        policy_lr=args.policy_lr,
        critic_lr=args.critic_lr,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_epsilon=args.clip_epsilon,
        value_coef=args.value_coef,
        entropy_coef=args.entropy_coef,
        ppo_epochs=args.ppo_epochs,
        batch_size=args.batch_size,
        policy_hidden_dims=policy_hidden_dims,
        critic_hidden_dims=critic_hidden_dims,
        save_path=args.save_path,
        save_freq=args.save_freq,
        load_policy_weights=args.load_policy_weights,
        load_critic_weights=args.load_critic_weights,
        wandb_config=wandb_config,
        seed=args.seed,
        max_grad_norm=args.max_grad_norm,
        deterministic_policy=args.deterministic_policy,
        eval_freq=args.eval_freq,
        eval_games_per_opponent=args.eval_games,
    )

    print("\n" + "=" * 60)
    print("Multi-agent central critic training complete!")
    print("=" * 60)
    if args.save_path:
        print(f"Policy model saved to: {args.save_path}")
        print(f"Critic model saved to: {os.path.splitext(args.save_path)[0]}_critic.pt")

    if args.wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
