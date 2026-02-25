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
    parser.add_argument(
        "--backbone-type",
        type=str,
        choices=["mlp", "xdim", "xdim_res"],
        default="mlp",
        help=(
            "Backbone architecture: mlp, xdim (cross-dimensional), "
            "or xdim_res (residual cross-dimensional)"
        ),
    )
    parser.add_argument(
        "--xdim-cnn-channels",
        type=str,
        default="64,128,128",
        help="Comma-separated CNN channels for xdim/xdim_res (default: 64,128,128)",
    )
    parser.add_argument(
        "--xdim-cnn-kernel-size",
        type=str,
        default="3,5",
        help="Kernel size for xdim/xdim_res CNN as 'height,width' (default: 3,5)",
    )
    parser.add_argument(
        "--xdim-policy-fusion-hidden-dim",
        type=int,
        default=None,
        help="Fusion hidden dim for policy xdim/xdim_res backbone (default: last policy hidden dim)",
    )
    parser.add_argument(
        "--xdim-critic-fusion-hidden-dim",
        type=int,
        default=None,
        help="Fusion hidden dim for critic xdim/xdim_res backbone (default: last critic hidden dim)",
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=1_000_000,
        help="Total number of environment steps to train for",
    )
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
    parser.add_argument("--activity-coef", type=float, default=0.0)
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
    parser.add_argument("--save-path", type=str, default="weights/marl_central_critic")
    parser.add_argument(
        "--eval-games",
        type=int,
        default=1000,
        help="Number of evaluation games per opponent (RandomPlayer and ValueFunctionPlayer).",
    )
    parser.add_argument(
        "--trend-eval-games",
        type=int,
        default=0,
        help="Games per opponent for fixed-seed trend eval (default: same as --eval-games).",
    )
    parser.add_argument(
        "--trend-eval-seed",
        type=int,
        default=67,
        help="Fixed seed used for trend-detection eval runs.",
    )
    parser.add_argument(
        "--eval-every-updates",
        type=int,
        default=1,
        help="Evaluate every N PPO updates (default: 1 = evaluate every update).",
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
        "--target-kl",
        type=float,
        default=None,
        help="Optional PPO KL target for early stopping (disabled when unset).",
    )
    parser.add_argument(
        "--deterministic-policy",
        action="store_true",
        help="Use greedy action selection during data collection (default: sample)",
    )
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb-project", type=str, default="catan-marl")
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--reward-function", type=str, default="shaped")
    parser.add_argument("--metric-window", type=int, default=200)

    args = parser.parse_args()

    # Check if weight files exist
    if args.load_policy_weights and not os.path.exists(args.load_policy_weights):
        print(f"Error: Policy weights file '{args.load_policy_weights}' not found!")
        return
    if args.load_critic_weights and not os.path.exists(args.load_critic_weights):
        print(f"Error: Critic weights file '{args.load_critic_weights}' not found!")
        return

    # Parse hidden dimensions
    policy_hidden_dims = [int(dim) for dim in args.policy_hidden_dims.split(",") if dim.strip()]
    critic_hidden_dims = [int(dim) for dim in args.critic_hidden_dims.split(",") if dim.strip()]
    xdim_cnn_channels = [int(ch) for ch in args.xdim_cnn_channels.split(",") if ch.strip()]
    xdim_kernel_parts = [int(k) for k in args.xdim_cnn_kernel_size.split(",") if k.strip()]
    if len(xdim_kernel_parts) != 2:
        raise ValueError(
            f"--xdim-cnn-kernel-size must have exactly 2 values (got: {args.xdim_cnn_kernel_size})"
        )
    xdim_cnn_kernel_size = (xdim_kernel_parts[0], xdim_kernel_parts[1])

    # Prepare wandb config (initialization happens in train)
    config_dict = {
        "algorithm": "PPO_Central_Critic",
        "total_timesteps": args.total_timesteps,
        "rollout_steps": args.rollout_steps,
        "policy_lr": args.policy_lr,
        "critic_lr": args.critic_lr,
        "gamma": args.gamma,
        "gae_lambda": args.gae_lambda,
        "clip_epsilon": args.clip_epsilon,
        "value_coef": args.value_coef,
        "entropy_coef": args.entropy_coef,
        "activity_coef": args.activity_coef,
        "ppo_epochs": args.ppo_epochs,
        "batch_size": args.batch_size,
        "policy_hidden_dims": policy_hidden_dims,
        "critic_hidden_dims": critic_hidden_dims,
        "num_players": args.num_players,
        "map_type": args.map_type,
        "model_type": args.model_type,
        "backbone_type": args.backbone_type,
        "xdim_cnn_channels": xdim_cnn_channels,
        "xdim_cnn_kernel_size": xdim_cnn_kernel_size,
        "xdim_policy_fusion_hidden_dim": args.xdim_policy_fusion_hidden_dim,
        "xdim_critic_fusion_hidden_dim": args.xdim_critic_fusion_hidden_dim,
        "max_grad_norm": args.max_grad_norm,
        "target_kl": args.target_kl,
        "deterministic_policy": args.deterministic_policy,
        "seed": args.seed,
        "trend_eval_games": args.trend_eval_games,
        "trend_eval_seed": args.trend_eval_seed,
        "eval_every_updates": args.eval_every_updates,
    }
    if args.wandb:
        wandb_config = {
            "project": args.wandb_project,
            "name": args.wandb_run_name,
            "config": config_dict,
        }
    else:
        wandb_config = {"mode": "disabled", "config": config_dict}

    # Train model
    train(
        num_players=args.num_players,
        map_type=args.map_type,
        model_type=args.model_type,
        backbone_type=args.backbone_type,
        xdim_cnn_channels=xdim_cnn_channels,
        xdim_cnn_kernel_size=xdim_cnn_kernel_size,
        xdim_policy_fusion_hidden_dim=args.xdim_policy_fusion_hidden_dim,
        xdim_critic_fusion_hidden_dim=args.xdim_critic_fusion_hidden_dim,
        total_timesteps=args.total_timesteps,
        rollout_steps=args.rollout_steps,
        policy_lr=args.policy_lr,
        critic_lr=args.critic_lr,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_epsilon=args.clip_epsilon,
        value_coef=args.value_coef,
        entropy_coef=args.entropy_coef,
        activity_coef=args.activity_coef,
        ppo_epochs=args.ppo_epochs,
        batch_size=args.batch_size,
        policy_hidden_dims=policy_hidden_dims,
        critic_hidden_dims=critic_hidden_dims,
        save_path=args.save_path,
        load_policy_weights=args.load_policy_weights,
        load_critic_weights=args.load_critic_weights,
        wandb_config=wandb_config,
        seed=args.seed,
        max_grad_norm=args.max_grad_norm,
        deterministic_policy=args.deterministic_policy,
        eval_games_per_opponent=args.eval_games,
        trend_eval_games_per_opponent=args.trend_eval_games,
        trend_eval_seed=args.trend_eval_seed,
        eval_every_updates=args.eval_every_updates,
        target_kl=args.target_kl,
        num_envs=args.num_envs,
        reward_function=args.reward_function,
        metric_window=args.metric_window,
    )

    print("\n" + "=" * 60)
    print("Multi-agent central critic training complete!")
    print("=" * 60)
    if args.save_path:
        print("Policy models saved under the configured directory.")
        print("Best policy: policy_best.pt")
        print("Best critic: critic_best.pt")

    if args.wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
