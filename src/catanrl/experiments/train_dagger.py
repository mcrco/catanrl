import argparse
import os

import wandb

from ..envs.puffer.common import create_opponents
from .architecture_config import (
    add_config_argument,
    architecture_train_config_fields,
    validate_player_count,
)
from .common_args import (
    DEFAULT_MAX_GRAD_NORM,
    add_device_argument,
    add_experiment_name_argument,
    add_reward_function_argument,
    add_wandb_arguments,
)
from ..algorithms.imitation_learning.dagger import train as dagger_train
from ..algorithms.imitation_learning.dataset import EvictionStrategy
from ..experiment_store import (
    GameConfig,
    KIND_POLICY,
    KIND_POLICY_VALUE,
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
        description="Train Catan policy and critic networks with DAgger imitation learning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_config_argument(parser)
    add_load_from_experiment_arguments(parser)
    add_resume_argument(parser)
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
    parser.add_argument(
        "--expert",
        type=str,
        default="F",
        help="Expert policy spec (e.g. AB:3, MCTS:200, random). Default: F",
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
        default=0.95,
        help="Multiplicative decay for beta per iteration (default: 0.95)",
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
        default="fifo",
        help="Eviction strategy when dataset is full: random, fifo, or correct",
    )
    parser.add_argument(
        "--opponents",
        type=str,
        nargs="+",
        default=["F"],
        help="Opponent bot configs for the environment (default: F)",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=4,
        help="Number of parallel environments (default: 4)",
    )
    add_reward_function_argument(parser)
    parser.add_argument(
        "--fresh-eval-games-per-opponent",
        type=int,
        default=1000,
        help="Fresh evaluation games per baseline opponent (default: 1000)",
    )
    parser.add_argument(
        "--eval-every-iterations",
        type=int,
        default=1,
        help=(
            "Run evaluation every N DAgger iterations "
            "(always evaluates on final iteration) (default: 1)"
        ),
    )
    parser.add_argument(
        "--save-every-updates",
        type=int,
        default=1,
        help="Save checkpoints every N DAgger iterations (default: 1)",
    )
    add_device_argument(parser)
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=DEFAULT_MAX_GRAD_NORM,
        help="Maximum gradient norm for clipping",
    )
    add_experiment_name_argument(parser)
    add_wandb_arguments(parser)

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
            "dagger", args.wandb_run_name, args.experiment_name
        )
        if args.wandb and not args.wandb_run_name:
            args.wandb_run_name = experiment_name

    args.save_path = default_checkpoints_dir(experiment_name)
    os.makedirs(args.save_path, exist_ok=True)
    training_state_path = training_state_file(experiment_name)

    if not args.opponents:
        args.opponents = ["random"]
    opponents = create_opponents(args.opponents)
    num_players = len(opponents) + 1
    validate_player_count(arch, num_players)
    if warm_start is not None and warm_start.experiment.num_players != num_players:
        print(
            f"Warning: source experiment was trained for "
            f"{warm_start.experiment.num_players} players but --opponents implies "
            f"{num_players}."
        )
    action_space_size = get_action_space_size(num_players, arch.map_type)
    print(f"Architecture: {setup.architecture_source}")
    print(f"Number of players: {num_players}")

    train_config = {
        "algorithm": "DAgger",
        **architecture_train_config_fields(arch),
        "iterations": args.iterations,
        "steps_per_iteration": args.steps_per_iter,
        "train_epochs": args.train_epochs,
        "batch_size": args.batch_size,
        "policy_lr": args.policy_lr,
        "critic_lr": args.critic_lr,
        "gamma": args.gamma,
        "expert": args.expert,
        "opponents": args.opponents,
        "num_envs": args.num_envs,
        "reward_function": args.reward_function,
        "beta_init": args.beta_init,
        "beta_decay": args.beta_decay,
        "beta_min": args.beta_min,
        "max_dataset_size": args.max_dataset_size,
        "eviction_strategy": args.eviction_strategy,
        "fresh_eval_games_per_opponent": args.fresh_eval_games_per_opponent,
        "eval_every_iterations": args.eval_every_iterations,
        "save_every_updates": args.save_every_updates,
        "seed": args.seed,
        "load_from_experiment": args.load_from_experiment,
        "load_from_which": args.load_from_which,
    }

    wandb_config = None
    if args.wandb:
        wandb_config = {
            "project": args.wandb_project,
            "name": args.wandb_run_name,
            "config": train_config,
            **wandb_grouping_kwargs(
                args,
                group_default="dagger",
                warm_start=warm_start,
                resume=resume,
            ),
        }
        if resume.active and resume.wandb_run_id:
            wandb_config["id"] = resume.wandb_run_id
            wandb_config["resume"] = "must"

    policy_model, critic_model = dagger_train(
        num_actions=action_space_size,
        model_type=arch.model_type,
        backbone_type=arch.backbone_type,
        policy_hidden_dims=arch.policy_hidden_dims,
        critic_hidden_dims=arch.critic_hidden_dims,
        xdim_cnn_channels=arch.xdim_cnn_channels,
        xdim_cnn_kernel_size=arch.xdim_cnn_kernel_size,
        xdim_policy_fusion_hidden_dim=arch.xdim_policy_fusion_hidden_dim,
        xdim_critic_fusion_hidden_dim=arch.xdim_critic_fusion_hidden_dim,
        load_policy_weights=warm_start.checkpoints.policy if warm_start else None,
        load_critic_weights=warm_start.checkpoints.critic if warm_start else None,
        n_iterations=args.iterations,
        steps_per_iteration=args.steps_per_iter,
        train_epochs=args.train_epochs,
        batch_size=args.batch_size,
        policy_lr=args.policy_lr,
        critic_lr=args.critic_lr,
        gamma=args.gamma,
        expert_config=args.expert,
        opponent_configs=args.opponents,
        map_type=arch.map_type,
        actor_observation_level=arch.actor_observation_level,
        critic_observation_level=arch.critic_observation_level,
        network_mode=arch.network_mode,
        vps_to_win=arch.vps_to_win,
        discard_limit=arch.discard_limit,
        beta_init=args.beta_init,
        beta_decay=args.beta_decay,
        beta_min=args.beta_min,
        max_dataset_size=args.max_dataset_size,
        eviction_strategy=EvictionStrategy(args.eviction_strategy),
        save_path=args.save_path,
        device=args.device,
        wandb_config=wandb_config,
        fresh_eval_games_per_opponent=args.fresh_eval_games_per_opponent,
        eval_every_iterations=args.eval_every_iterations,
        save_every_updates=args.save_every_updates,
        seed=args.seed,
        num_envs=args.num_envs,
        reward_function=args.reward_function,
        max_grad_norm=args.max_grad_norm,
        resume_state=resume.state,
        training_state_path=training_state_path,
    )

    if arch.network_mode == "shared":
        networks = {
            "policy": network_spec_from_model(
                policy_model,
                kind=KIND_POLICY_VALUE,
                model_type=arch.model_type,
                observation_level=arch.actor_observation_level,
            ),
        }
    else:
        networks = {
            "policy": network_spec_from_model(
                policy_model,
                kind=KIND_POLICY,
                model_type=arch.model_type,
                observation_level=arch.actor_observation_level,
            ),
            "critic": network_spec_from_model(
                critic_model,
                kind=KIND_VALUE,
                observation_level=arch.critic_observation_level,
            ),
        }
    exp_path = save_experiment(
        experiment_name,
        args.save_path,
        algorithm="dagger",
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
    print("DAgger training finished")
    print("=" * 60)
    print(f"Checkpoints saved to: {args.save_path}")
    print(f"  - Policy: {args.save_path}/policy_best.pt")
    print(f"  - Critic: {args.save_path}/critic_best.pt")
    if exp_path:
        print(f"Experiment:  {exp_path}  (load via load_experiment('{experiment_name}'))")

    if args.wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
