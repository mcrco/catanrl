import argparse
import os
import wandb
from catanrl.experiments.architecture_config import (
    add_config_argument,
    architecture_train_config_fields,
)
from catanrl.experiments.common_args import (
    DEFAULT_MAX_GRAD_NORM,
    DEFAULT_METRIC_WINDOW,
    DEFAULT_TREND_EVAL_SEED,
    add_device_argument,
    add_experiment_name_argument,
    add_reward_function_argument,
    add_save_every_updates_argument,
    add_train_epochs_argument,
    add_wandb_arguments,
)
from catanrl.algorithms.ppo.marl_ppo_central_critic import train
from catanrl.experiment_store import (
    GameConfig,
    KIND_POLICY,
    KIND_VALUE,
    add_load_from_experiment_arguments,
    add_resume_argument,
    default_checkpoints_dir,
    make_experiment_name,
    network_spec_from_model,
    prepare_resume,
    resolve_training_architecture_and_warm_start,
    save_experiment,
    training_state_file,
)


def _wandb_info(args: argparse.Namespace) -> dict:
    """Experiment metadata W&B block, including the live run id when available."""
    if not args.wandb:
        return {}
    info = {"project": args.wandb_project, "name": args.wandb_run_name}
    if wandb.run is not None:
        info["id"] = wandb.run.id
    return info


def main():
    parser = argparse.ArgumentParser(
        description="Train a multi-agent self-play PPO policy with centralized critic using the Puffer Catanatron env",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_config_argument(parser)
    add_load_from_experiment_arguments(parser)
    add_resume_argument(parser)
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
    add_train_epochs_argument(parser, default=4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--fresh-eval-games-per-opponent",
        type=int,
        default=1000,
        help="Fresh evaluation games per opponent (RandomPlayer and ValueFunctionPlayer).",
    )
    parser.add_argument(
        "--trend-eval-games-per-opponent",
        type=int,
        default=0,
        help="Fixed-seed trend eval games per opponent (default: same as fresh-eval-games-per-opponent).",
    )
    parser.add_argument(
        "--trend-eval-seed",
        type=int,
        default=DEFAULT_TREND_EVAL_SEED,
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
    add_save_every_updates_argument(parser)
    add_device_argument(parser)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-grad-norm", type=float, default=DEFAULT_MAX_GRAD_NORM)
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
    add_experiment_name_argument(parser)
    add_wandb_arguments(parser)
    add_reward_function_argument(parser)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--metric-window", type=int, default=DEFAULT_METRIC_WINDOW)

    args = parser.parse_args()

    try:
        setup = resolve_training_architecture_and_warm_start(args)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}")
        return

    arch = setup.arch
    warm_start = setup.warm_start
    if arch.network_mode == "shared":
        print(
            "Error: MARL central critic training does not yet support network_mode='shared'. "
            "Use a preset with network_mode='separate'."
        )
        return
    if arch.num_players is None:
        print("Error: MARL training requires game.num_players in the architecture preset.")
        return

    try:
        resume = prepare_resume(args, warm_start)
    except ValueError as exc:
        print(f"Error: {exc}")
        return

    if resume.active:
        experiment_name = resume.experiment_name
        if args.wandb and not args.wandb_run_name:
            args.wandb_run_name = resume.wandb_run_name or experiment_name
    else:
        experiment_name = make_experiment_name(
            "marl-cc", args.wandb_run_name, args.experiment_name
        )
        if args.wandb and not args.wandb_run_name:
            args.wandb_run_name = experiment_name
    args.save_path = default_checkpoints_dir(experiment_name)
    os.makedirs(args.save_path, exist_ok=True)
    training_state_path = training_state_file(experiment_name)
    print(f"Architecture: {setup.architecture_source}")

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
        "train_epochs": args.train_epochs,
        "batch_size": args.batch_size,
        **architecture_train_config_fields(arch),
        "max_grad_norm": args.max_grad_norm,
        "target_kl": args.target_kl,
        "deterministic_policy": args.deterministic_policy,
        "seed": args.seed,
        "trend_eval_games_per_opponent": args.trend_eval_games_per_opponent,
        "trend_eval_seed": args.trend_eval_seed,
        "fresh_eval_games_per_opponent": args.fresh_eval_games_per_opponent,
        "h2h_eval_games": args.h2h_eval_games,
        "h2h_eval_seed": args.h2h_eval_seed,
        "eval_every_updates": args.eval_every_updates,
        "save_every_updates": args.save_every_updates,
        "device": args.device,
        "load_from_experiment": args.load_from_experiment,
        "load_from_which": args.load_from_which,
        "reward_function": args.reward_function,
        "num_envs": args.num_envs,
        "metric_window": args.metric_window,
    }
    if args.wandb:
        wandb_config = {
            "project": args.wandb_project,
            "name": args.wandb_run_name,
            "config": config_dict,
        }
        if resume.active and resume.wandb_run_id:
            wandb_config["id"] = resume.wandb_run_id
            wandb_config["resume"] = "must"
    else:
        wandb_config = {"mode": "disabled", "config": config_dict}

    policy_model, critic_model = train(
        num_players=arch.num_players,
        map_type=arch.map_type,
        actor_observation_level=arch.actor_observation_level,
        critic_observation_level=arch.critic_observation_level,
        model_type=arch.model_type,
        backbone_type=arch.backbone_type,
        xdim_cnn_channels=arch.xdim_cnn_channels,
        xdim_cnn_kernel_size=arch.xdim_cnn_kernel_size,
        xdim_policy_fusion_hidden_dim=arch.xdim_policy_fusion_hidden_dim,
        xdim_critic_fusion_hidden_dim=arch.xdim_critic_fusion_hidden_dim,
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
        train_epochs=args.train_epochs,
        batch_size=args.batch_size,
        policy_hidden_dims=arch.policy_hidden_dims,
        critic_hidden_dims=arch.critic_hidden_dims,
        save_path=args.save_path,
        load_policy_weights=warm_start.checkpoints.policy if warm_start else None,
        load_critic_weights=warm_start.checkpoints.critic if warm_start else None,
        wandb_config=wandb_config,
        seed=args.seed,
        device=args.device,
        max_grad_norm=args.max_grad_norm,
        deterministic_policy=args.deterministic_policy,
        fresh_eval_games_per_opponent=args.fresh_eval_games_per_opponent,
        trend_eval_games_per_opponent=args.trend_eval_games_per_opponent,
        trend_eval_seed=args.trend_eval_seed,
        h2h_eval_games=args.h2h_eval_games,
        h2h_eval_seed=args.h2h_eval_seed,
        eval_every_updates=args.eval_every_updates,
        save_every_updates=args.save_every_updates,
        target_kl=args.target_kl,
        num_envs=args.num_envs,
        reward_function=args.reward_function,
        vps_to_win=arch.vps_to_win,
        discard_limit=arch.discard_limit,
        metric_window=args.metric_window,
        resume_state=resume.state,
        training_state_path=training_state_path,
    )

    networks = {
        "policy": network_spec_from_model(
            policy_model,
            kind=KIND_POLICY,
            model_type=arch.model_type,
            observation_level=arch.actor_observation_level,
        )
    }
    if critic_model is not None:
        networks["critic"] = network_spec_from_model(
            critic_model,
            kind=KIND_VALUE,
            observation_level=arch.critic_observation_level,
        )
    exp_path = save_experiment(
        experiment_name,
        args.save_path,
        algorithm="marl_cc",
        game=GameConfig(
            num_players=arch.num_players,
            map_type=arch.map_type,
            vps_to_win=arch.vps_to_win,
            discard_limit=arch.discard_limit,
        ),
        networks=networks,
        train_config=config_dict,
        wandb_info=_wandb_info(args),
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
