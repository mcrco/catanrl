import argparse
import os
import wandb

from ..envs.puffer.common import compute_single_agent_dims, create_opponents
from .architecture_config import (
    add_config_argument,
    architecture_train_config_fields,
    validate_player_count,
)
from .common_args import (
    DEFAULT_MAX_GRAD_NORM,
    DEFAULT_METRIC_WINDOW,
    add_device_argument,
    add_experiment_name_argument,
    add_fresh_eval_arguments,
    add_reward_function_argument,
    add_save_every_updates_argument,
    add_train_epochs_argument,
    add_wandb_arguments,
)
from ..algorithms.ppo.sarl_ppo import train
from ..experiment_store import (
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
    wandb_grouping_kwargs,
)
from ..utils.catanatron_action_space import get_action_space_size


def _wandb_info(args: argparse.Namespace) -> dict:
    """Experiment metadata W&B block, including the live run id when available."""
    if not args.wandb:
        return {}
    info = {"project": args.wandb_project, "name": args.wandb_run_name}
    if wandb.run is not None:
        info["id"] = wandb.run.id
        if wandb.run.tags:
            info["tags"] = list(wandb.run.tags)
    return info


def main():
    parser = argparse.ArgumentParser(
        description="Train Catan Policy-Value Network with Self-Play RL (PPO)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_config_argument(parser)
    add_load_from_experiment_arguments(parser)
    add_resume_argument(parser)
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=1_000_000,
        help="Total number of environment steps to train for (default: 1000000)",
    )
    parser.add_argument(
        "--rollout-steps",
        type=int,
        default=4096,
        help="Update model every N rollout steps (default: 4096)",
    )
    parser.add_argument(
        "--policy-lr", type=float, default=1e-4, help="Policy learning rate (default: 1e-4)"
    )
    parser.add_argument(
        "--critic-lr",
        type=float,
        default=None,
        help="Critic learning rate with separate networks (default: use --policy-lr)",
    )
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor (default: 0.99)")
    parser.add_argument(
        "--gae-lambda", type=float, default=0.95, help="GAE lambda parameter (default: 0.95)"
    )
    parser.add_argument(
        "--clip-epsilon", type=float, default=0.2, help="PPO clipping parameter (default: 0.2)"
    )
    parser.add_argument(
        "--value-coef", type=float, default=0.5, help="Value loss coefficient (default: 0.5)"
    )
    parser.add_argument(
        "--entropy-coef", type=float, default=0.01, help="Entropy coefficient (default: 0.01)"
    )
    parser.add_argument(
        "--activity-coef",
        type=float,
        default=0.0,
        help="Activity regularization coefficient on raw policy logits (default: 0.0)",
    )
    add_train_epochs_argument(parser, default=10)
    parser.add_argument(
        "--batch-size", type=int, default=64, help="Mini-batch size for PPO update (default: 64)"
    )
    add_save_every_updates_argument(
        parser, help="Save policy snapshot every N PPO updates (default: 1)"
    )
    parser.add_argument(
        "--opponents",
        type=str,
        nargs="+",
        default=["random"],
        help="""Opponent bot types (1-3 opponents for 2-4 player game). Options:
                - random, R: RandomPlayer
                - weighted, W: WeightedRandomPlayer  
                - value, F: ValueFunctionPlayer
                - victorypoint, VP: VictoryPointPlayer
                - alphabeta, AB: AlphaBetaPlayer (use AB:depth:prunning for custom params)
                - sameturnalphabeta, SAB: SameTurnAlphaBetaPlayer
                - mcts, M: MCTSPlayer (use M:num_sims for custom simulations)
                - playouts, G: GreedyPlayoutsPlayer (use G:num_playouts for custom playouts)
                Examples: --opponents random random random
                            --opponents F W AB
                            --opponents AB:3:False M:500
                (default: random)""",
    )
    add_reward_function_argument(parser)
    add_experiment_name_argument(parser)
    add_wandb_arguments(parser)
    add_device_argument(parser)
    parser.add_argument(
        "--num-envs",
        type=int,
        default=4,
        help="Number of parallel environments for vectorized training (default: 4)",
    )
    parser.add_argument(
        "--use-lr-scheduler",
        action="store_true",
        help="Enable linear learning rate scheduling (default: False)",
    )
    parser.add_argument(
        "--lr-scheduler-start-factor",
        type=float,
        default=1.0,
        help="Start factor for LinearLR scheduler (default: 1.0)",
    )
    parser.add_argument(
        "--lr-scheduler-end-factor",
        type=float,
        default=0.0,
        help="End factor for LinearLR scheduler (default: 0.0)",
    )
    parser.add_argument(
        "--lr-scheduler-total-iters",
        type=int,
        default=None,
        help="Total iterations for LinearLR scheduler (default: total_timesteps/num_envs)",
    )
    parser.add_argument(
        "--metric-window",
        type=int,
        default=DEFAULT_METRIC_WINDOW,
        help="Window size for metrics (avg reward and length), also used for best model saving",
    )
    add_fresh_eval_arguments(parser)
    parser.add_argument(
        "--deterministic-policy",
        action="store_true",
        help="Use deterministic (argmax) policy during rollout collection",
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=DEFAULT_MAX_GRAD_NORM,
        help="Gradient clipping norm for PPO updates",
    )
    parser.add_argument(
        "--target-kl",
        type=float,
        default=None,
        help="Optional KL threshold for PPO early stopping (default: disabled)",
    )

    args = parser.parse_args()

    try:
        setup = resolve_training_architecture_and_warm_start(args)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}")
        return

    arch = setup.arch
    warm_start = setup.warm_start

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
            "sarl-ppo", args.wandb_run_name, args.experiment_name
        )
        if args.wandb and not args.wandb_run_name:
            args.wandb_run_name = experiment_name

    args.save_path = default_checkpoints_dir(experiment_name)
    os.makedirs(args.save_path, exist_ok=True)
    training_state_path = training_state_file(experiment_name)

    if not args.opponents:
        args.opponents = ["random"]
    temp_opponents = create_opponents(args.opponents)
    num_players = len(temp_opponents) + 1
    validate_player_count(arch, num_players)
    if warm_start is not None and warm_start.experiment.num_players != num_players:
        print(
            f"Warning: source experiment was trained for "
            f"{warm_start.experiment.num_players} players but --opponents implies "
            f"{num_players}."
        )
    action_space_size = get_action_space_size(num_players, arch.map_type)

    dims = compute_single_agent_dims(
        num_players,
        arch.map_type,
        actor_observation_level=arch.actor_observation_level,
    )
    input_dim = dims["actor_dim"]
    print(f"Architecture: {setup.architecture_source}")
    print(f"Number of players: {num_players}")
    print(f"Input dimension: {input_dim}")

    lr_scheduler_kwargs = None
    if args.use_lr_scheduler:
        lr_scheduler_kwargs = {
            "start_factor": args.lr_scheduler_start_factor,
            "end_factor": args.lr_scheduler_end_factor,
        }
        if args.lr_scheduler_total_iters is not None:
            lr_scheduler_kwargs["total_iters"] = args.lr_scheduler_total_iters

    train_config = {
        "algorithm": "PPO",
        "total_timesteps": args.total_timesteps,
        "rollout_steps": args.rollout_steps,
        "reward_function": args.reward_function,
        "policy_lr": args.policy_lr,
        "gamma": args.gamma,
        "gae_lambda": args.gae_lambda,
        "clip_epsilon": args.clip_epsilon,
        "value_coef": args.value_coef,
        "entropy_coef": args.entropy_coef,
        "activity_coef": args.activity_coef,
        "train_epochs": args.train_epochs,
        "batch_size": args.batch_size,
        **architecture_train_config_fields(arch),
        "load_from_experiment": args.load_from_experiment,
        "load_from_which": args.load_from_which,
        "opponents": args.opponents,
        "num_envs": args.num_envs,
        "critic_lr": args.critic_lr,
        "use_lr_scheduler": args.use_lr_scheduler,
        "lr_scheduler_kwargs": lr_scheduler_kwargs,
        "fresh_eval_games_per_opponent": args.fresh_eval_games_per_opponent,
        "trend_eval_games_per_opponent": args.trend_eval_games_per_opponent,
        "trend_eval_seed": args.trend_eval_seed,
        "eval_every_updates": args.eval_every_updates,
        "save_every_updates": args.save_every_updates,
        "deterministic_policy": args.deterministic_policy,
        "max_grad_norm": args.max_grad_norm,
        "target_kl": args.target_kl,
        "device": args.device,
    }

    wandb_config = None
    if args.wandb:
        wandb_config = {
            "project": args.wandb_project,
            "name": args.wandb_run_name,
            "config": train_config,
            **wandb_grouping_kwargs(
                args,
                group_default="sarl-ppo",
                warm_start=warm_start,
                resume=resume,
            ),
        }
        if resume.active and resume.wandb_run_id:
            wandb_config["id"] = resume.wandb_run_id
            wandb_config["resume"] = "must"

    policy_model, critic_model = train(
        input_dim=input_dim,
        num_actions=action_space_size,
        model_type=arch.model_type,
        backbone_type=arch.backbone_type,
        xdim_cnn_channels=arch.xdim_cnn_channels,
        xdim_cnn_kernel_size=arch.xdim_cnn_kernel_size,
        xdim_policy_fusion_hidden_dim=arch.xdim_policy_fusion_hidden_dim,
        xdim_critic_fusion_hidden_dim=arch.xdim_critic_fusion_hidden_dim,
        policy_hidden_dims=arch.policy_hidden_dims,
        critic_hidden_dims=arch.critic_hidden_dims,
        network_mode=arch.network_mode,
        load_weights=warm_start.checkpoints.policy if warm_start else None,
        load_critic_weights=warm_start.checkpoints.critic if warm_start else None,
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
        save_path=args.save_path,
        save_every_updates=args.save_every_updates,
        device=args.device,
        wandb_config=wandb_config,
        map_type=arch.map_type,
        actor_observation_level=arch.actor_observation_level,
        critic_observation_level=arch.critic_observation_level,
        vps_to_win=arch.vps_to_win,
        discard_limit=arch.discard_limit,
        opponent_configs=args.opponents,
        reward_function=args.reward_function,
        num_envs=args.num_envs,
        use_lr_scheduler=args.use_lr_scheduler,
        lr_scheduler_kwargs=lr_scheduler_kwargs,
        metric_window=args.metric_window,
        fresh_eval_games_per_opponent=args.fresh_eval_games_per_opponent,
        trend_eval_games_per_opponent=args.trend_eval_games_per_opponent,
        trend_eval_seed=args.trend_eval_seed,
        eval_every_updates=args.eval_every_updates,
        deterministic_policy=args.deterministic_policy,
        max_grad_norm=args.max_grad_norm,
        target_kl=args.target_kl,
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
        algorithm="sarl_ppo",
        game=GameConfig(
            num_players=num_players,
            map_type=arch.map_type,
            vps_to_win=arch.vps_to_win,
            discard_limit=arch.discard_limit,
        ),
        networks=networks,
        train_config=train_config,
        wandb_info=_wandb_info(args),
    )

    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Model saved: {args.save_path}")
    if exp_path:
        print(f"Experiment:  {exp_path}  (load via load_experiment('{experiment_name}'))")

    if args.wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
