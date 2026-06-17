import argparse
import os
import wandb
from catanrl.algorithms.common.network_config import (
    add_observation_network_arguments,
    resolve_observation_network_args,
)
from catanrl.algorithms.ppo.marl_ppo_central_critic import train
from catanrl.experiment_store import (
    GameConfig,
    KIND_POLICY,
    KIND_VALUE,
    add_load_from_experiment_arguments,
    default_checkpoints_dir,
    make_experiment_name,
    network_spec_from_model,
    prepare_training_warm_start,
    save_experiment,
)


def main():
    parser = argparse.ArgumentParser(
        description="Train a multi-agent self-play PPO policy with centralized critic using the PettingZoo Catanatron env",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--num-players", type=int, default=2, choices=[2, 3, 4])
    parser.add_argument("--map-type", type=str, default="BASE", choices=["BASE", "MINI"])
    add_observation_network_arguments(
        parser,
        policy_mode_default="private",
        critic_mode_default="full",
        network_mode_default="separate",
    )
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
        "--h2h-eval-games",
        type=int,
        default=0,
        help="Optional current-vs-champion head-to-head games per eval pass (0 disables).",
    )
    parser.add_argument(
        "--h2h-eval-seed",
        type=int,
        default=123,
        help="Fixed seed used for current-vs-champion head-to-head eval runs.",
    )
    parser.add_argument(
        "--eval-every-updates",
        type=int,
        default=1,
        help="Evaluate every N PPO updates (default: 1 = evaluate every update).",
    )
    parser.add_argument(
        "--save-every-updates",
        type=int,
        default=1,
        help="Save snapshots every N PPO updates (default: 1 = save every update).",
    )
    add_load_from_experiment_arguments(parser)
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
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Experiment name (folder under experiments/ and W&B run name). "
        "Defaults to --wandb-run-name, else an auto-generated name.",
    )
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb-project", type=str, default="catan-marl")
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--reward-function", type=str, default="shaped")
    parser.add_argument(
        "--vps-to-win",
        type=int,
        default=15,
        help="Victory points required to win each game.",
    )
    parser.add_argument(
        "--discard-limit",
        type=int,
        default=9,
        help="Discard threshold used when a 7 is rolled.",
    )
    parser.add_argument("--metric-window", type=int, default=200)

    args = parser.parse_args()

    resolve_observation_network_args(args)
    if args.network_mode == "shared":
        raise ValueError(
            "MARL central critic training does not yet support network_mode='shared'. "
            "Use network_mode='separate'."
        )

    try:
        warm_start = prepare_training_warm_start(args)
    except FileNotFoundError as exc:
        print(f"Error: {exc}")
        return

    # Resolve experiment identity (shared by the folder and the W&B run).
    experiment_name = make_experiment_name("marl-cc", args.wandb_run_name, args.experiment_name)
    if args.wandb and not args.wandb_run_name:
        args.wandb_run_name = experiment_name
    args.save_path = default_checkpoints_dir(experiment_name)
    os.makedirs(args.save_path, exist_ok=True)

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
        "policy_mode": args.policy_mode,
        "critic_mode": args.critic_mode,
        "network_mode": args.network_mode,
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
        "h2h_eval_games": args.h2h_eval_games,
        "h2h_eval_seed": args.h2h_eval_seed,
        "eval_every_updates": args.eval_every_updates,
        "save_every_updates": args.save_every_updates,
        "vps_to_win": args.vps_to_win,
        "discard_limit": args.discard_limit,
        "load_from_experiment": args.load_from_experiment,
        "load_from_which": args.load_from_which,
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
    policy_model, critic_model = train(
        num_players=args.num_players,
        map_type=args.map_type,
        actor_observation_level=args.actor_observation_level,
        critic_observation_level=args.critic_observation_level,
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
        load_policy_weights=warm_start.checkpoints.policy if warm_start else None,
        load_critic_weights=warm_start.checkpoints.critic if warm_start else None,
        wandb_config=wandb_config,
        seed=args.seed,
        max_grad_norm=args.max_grad_norm,
        deterministic_policy=args.deterministic_policy,
        eval_games_per_opponent=args.eval_games,
        trend_eval_games_per_opponent=args.trend_eval_games,
        trend_eval_seed=args.trend_eval_seed,
        h2h_eval_games=args.h2h_eval_games,
        h2h_eval_seed=args.h2h_eval_seed,
        eval_every_updates=args.eval_every_updates,
        save_every_updates=args.save_every_updates,
        target_kl=args.target_kl,
        num_envs=args.num_envs,
        reward_function=args.reward_function,
        vps_to_win=args.vps_to_win,
        discard_limit=args.discard_limit,
        metric_window=args.metric_window,
    )

    networks = {
        "policy": network_spec_from_model(
            policy_model,
            kind=KIND_POLICY,
            model_type=args.model_type,
            observation_level=args.actor_observation_level,
        )
    }
    if critic_model is not None:
        networks["critic"] = network_spec_from_model(
            critic_model,
            kind=KIND_VALUE,
            observation_level=args.critic_observation_level,
        )
    exp_path = save_experiment(
        experiment_name,
        args.save_path,
        algorithm="marl_cc",
        game=GameConfig(
            num_players=args.num_players,
            map_type=args.map_type,
            vps_to_win=args.vps_to_win,
            discard_limit=args.discard_limit,
        ),
        networks=networks,
        train_config=config_dict,
        wandb_info={"project": args.wandb_project, "name": args.wandb_run_name} if args.wandb else {},
    )

    print("\n" + "=" * 60)
    print("Multi-agent central critic training complete!")
    print("=" * 60)
    if args.save_path:
        print(f"Policy models saved under: {args.save_path}")
        print("Best policy: policy_best.pt")
        print("Best critic: critic_best.pt")
    if exp_path:
        print(f"Experiment:  {exp_path}  (load via load_experiment('{experiment_name}'))")

    if args.wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
