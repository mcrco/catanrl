#!/usr/bin/env python3
"""Train a policy from neural MCTS targets.

``distill`` keeps the warm-started search teacher frozen and answers the narrow
question "can the raw policy absorb MCTS's improvement?". ``iterate`` treats an
accepted student as the next teacher, with deployable candidate-vs-champion and
fixed-opponent gates. Both modes use the same parallel MCTS collector.
"""

from __future__ import annotations

import argparse
import math
import os
from typing import Any, Iterable, Literal, cast

import torch
import wandb
from tqdm import tqdm

from catanrl.algorithms.alphazero.trainer import AlphaZeroConfig, AlphaZeroTrainer
from catanrl.eval.search_training import (
    PolicyEvalResult,
    decide_promotion,
    evaluate_candidate_vs_champion,
    evaluate_policy_vs_value,
)
from catanrl.experiment_store import (
    CHECKPOINTS_DIRNAME,
    KIND_POLICY,
    KIND_POLICY_VALUE,
    KIND_VALUE,
    GameConfig,
    ResumeContext,
    TrainingWarmStart,
    add_load_from_experiment_arguments,
    add_resume_argument,
    build_checkpoint_registry,
    default_checkpoints_dir,
    experiment_dir,
    make_experiment_name,
    network_spec_from_model,
    prepare_resume,
    resolve_training_architecture_and_warm_start,
    save_checkpoint_registry,
    save_experiment,
    training_state_file,
    wandb_grouping_kwargs,
)
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
from catanrl.experiments.network_config import validate_ismcts_observation_levels
from catanrl.features.catanatron_utils import ActorObservationLevel, CriticObservationLevel
from catanrl.models.model_builders import (
    build_critic_model,
    build_policy_model,
    build_policy_value_model,
)
from catanrl.models.wrappers import (
    PolicyNetworkWrapper,
    PolicyValueNetworkWrapper,
    ValueNetworkWrapper,
)

PolicyModel = PolicyNetworkWrapper | PolicyValueNetworkWrapper
CriticModel = ValueNetworkWrapper | None


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Distill a frozen MCTS teacher or run gated Expert Iteration.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=("distill", "iterate"),
        default="distill",
        help="Frozen teacher distillation or candidate/champion Expert Iteration.",
    )
    parser.add_argument(
        "--teacher-update",
        choices=("gated", "latest"),
        default="gated",
        help=(
            "For iterate mode, either gate teacher promotions at evaluation points "
            "or use the newly trained student for the next self-play iteration."
        ),
    )
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--games-per-iteration", type=int, default=64)
    parser.add_argument("--optimizer-steps", type=int, default=128)
    parser.add_argument(
        "--optimizer-epochs",
        type=int,
        default=0,
        help=(
            "Complete shuffled passes over the current replay buffer per iteration. "
            "A positive value overrides --optimizer-steps."
        ),
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=min(16, os.cpu_count() or 4),
        help="Across-game search workers sharing the central inference server.",
    )
    add_config_argument(parser)

    search = parser.add_argument_group("search teacher")
    search.add_argument(
        "--self-play-backend",
        choices=("python", "cppanatron"),
        default="python",
        help=(
            "Game/search implementation used to collect replay targets. "
            "cppanatron keeps tree traversal and game copies in native C++."
        ),
    )
    search.add_argument("--simulations", type=int, default=64)
    search.add_argument(
        "--ismcts-determinizations",
        "--is-mcts-determinizations",
        dest="ismcts_determinizations",
        type=int,
        default=8,
    )
    search.add_argument("--c-puct", type=float, default=1.5)
    search.add_argument(
        "--value-scale",
        type=float,
        default=1.0,
        help="Multiplier applied to non-terminal critic values backed up by native MCTS.",
    )
    search.add_argument(
        "--tree-reuse",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Reuse native subtrees after deterministic played actions.",
    )
    search.add_argument(
        "--canonical-pruning",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable exact successor and discard-order deduplication in native MCTS.",
    )
    search.add_argument(
        "--search-selection",
        choices=("puct", "completed-q"),
        default="puct",
        help=(
            "Tree allocation rule: ordinary PUCT at every node or Canopy-style "
            "root PUCT plus completed-Q improved-policy allocation below the root."
        ),
    )
    search.add_argument(
        "--full-search-probability",
        type=float,
        default=1.0,
        help=(
            "Probability that a non-forced decision uses the full search budget and "
            "contributes a policy target. Other decisions use --fast-simulations and "
            "contribute value targets only."
        ),
    )
    search.add_argument(
        "--fast-simulations",
        type=int,
        default=64,
        help="Search budget for playout-cap decisions whose policy loss is masked.",
    )
    search.add_argument(
        "--search-value-weight-max",
        type=float,
        default=0.0,
        help="Maximum root-search-Q contribution to terminal win/loss value targets.",
    )
    search.add_argument(
        "--search-value-weight-ramp-iterations",
        type=int,
        default=1,
        help="Self-play iterations over which root-Q target weight ramps from zero.",
    )
    search.add_argument(
        "--policy-target",
        choices=("visits", "completed-q"),
        default="visits",
        help=(
            "Policy target extracted from native search: ordinary visit counts or "
            "Canopy/Gumbel-AZ-style completed-Q policy improvement."
        ),
    )
    search.add_argument(
        "--c-visit",
        type=float,
        default=50.0,
        help="Completed-Q policy-improvement visit offset.",
    )
    search.add_argument(
        "--c-scale",
        type=float,
        default=1.0,
        help="Completed-Q policy-improvement Q scale.",
    )
    search.add_argument("--prunning", action="store_true")
    search.add_argument("--temperature", type=float, default=1.0)
    search.add_argument("--final-temperature", type=float, default=0.1)
    search.add_argument(
        "--target-temperature",
        type=float,
        default=None,
        help=(
            "Temperature for stored MCTS visit targets. By default it follows the "
            "trajectory action-temperature schedule."
        ),
    )
    search.add_argument("--temperature-drop-move", type=int, default=30)
    search.add_argument(
        "--noise-turns",
        type=int,
        default=20,
        help="Moves with root noise in visit-trajectory mode; Canopy mode uses noise throughout.",
    )
    search.add_argument(
        "--trajectory-action-selection",
        choices=("visits", "canopy"),
        default="visits",
        help=(
            "Choose actions with the temperature-adjusted visit policy, or match "
            "Canopy by sampling completed-Q targets for early exploration and then "
            "taking the visit-count argmax while keeping root noise on every move."
        ),
    )
    search.add_argument(
        "--explore-actions",
        type=int,
        default=24,
        help="Total game actions that sample the improved policy in Canopy trajectory mode.",
    )
    search.add_argument("--dirichlet-alpha", type=float, default=0.3)
    search.add_argument("--dirichlet-frac", type=float, default=0.25)
    search.add_argument("--inference-batch-size", type=int, default=64)
    search.add_argument("--inference-wait-ms", type=float, default=2.0)

    optimization = parser.add_argument_group("student optimization")
    optimization.add_argument("--buffer-size", type=int, default=50_000)
    optimization.add_argument("--batch-size", type=int, default=256)
    optimization.add_argument("--policy-lr", type=float, default=5e-5)
    optimization.add_argument("--critic-lr", type=float, default=1e-4)
    optimization.add_argument("--weight-decay", type=float, default=0.0)
    optimization.add_argument("--policy-loss-weight", type=float, default=1.0)
    optimization.add_argument(
        "--soft-policy-temperature",
        type=float,
        default=0.0,
        help="Temperature for a fresh auxiliary softened-search-policy head (0 disables).",
    )
    optimization.add_argument(
        "--soft-policy-weight",
        type=float,
        default=0.0,
        help="Loss weight for the auxiliary soft policy head.",
    )
    optimization.add_argument(
        "--value-loss-weight",
        type=float,
        default=None,
        help="Defaults to 0 for distill and 1 for iterate.",
    )
    optimization.add_argument("--max-grad-norm", type=float, default=DEFAULT_MAX_GRAD_NORM)
    optimization.add_argument(
        "--offload-inactive-models",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Move the inactive teacher/student pair and optimizer state to CPU.",
    )

    evaluation = parser.add_argument_group("strength evaluation")
    evaluation.add_argument("--eval-every-iterations", type=int, default=1)
    evaluation.add_argument(
        "--eval-games",
        type=int,
        default=500,
        help="Total fixed-seed games versus ValueFunctionPlayer, split across seats.",
    )
    evaluation.add_argument("--eval-seed", type=int, default=123)
    evaluation.add_argument(
        "--h2h-games",
        type=int,
        default=200,
        help="Total candidate-vs-teacher games per evaluation, split across seats.",
    )
    evaluation.add_argument("--h2h-seed", type=int, default=123)
    evaluation.add_argument("--promotion-threshold", type=float, default=0.52)
    evaluation.add_argument("--max-baseline-regression", type=float, default=0.02)

    add_save_every_updates_argument(
        parser,
        default=1,
        help="Save an iteration checkpoint every N iterations (0 disables periodic saves).",
    )
    add_load_from_experiment_arguments(parser)
    add_resume_argument(parser)
    parser.add_argument(
        "--reset-loaded-value-head",
        action="store_true",
        help=(
            "Warm-start the policy/backbone from --load-from-experiment but keep a "
            "freshly initialized value head. This prevents imitation value targets "
            "from seeding search-guided win/loss training."
        ),
    )
    add_experiment_name_argument(parser)
    add_device_argument(parser)
    parser.add_argument("--seed", type=int, default=42)
    add_wandb_arguments(parser)
    parser.add_argument("--wandb-entity", type=str, default=None)

    args = parser.parse_args(list(argv) if argv is not None else None)
    _validate_args(parser, args)
    return args


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    positive = {
        "iterations": args.iterations,
        "games-per-iteration": args.games_per_iteration,
        "optimizer-steps": args.optimizer_steps,
        "num-workers": args.num_workers,
        "simulations": args.simulations,
        "fast-simulations": args.fast_simulations,
        "ismcts-determinizations": args.ismcts_determinizations,
        "batch-size": args.batch_size,
        "buffer-size": args.buffer_size,
        "eval-every-iterations": args.eval_every_iterations,
    }
    for name, value in positive.items():
        if value < 1:
            parser.error(f"--{name} must be at least 1")
    if args.optimizer_epochs < 0:
        parser.error("--optimizer-epochs cannot be negative")
    for name in ("eval_games", "h2h_games"):
        value = int(getattr(args, name))
        if value < 2 or value % 2:
            parser.error(f"--{name.replace('_', '-')} must be a positive even number")
    if args.buffer_size < args.batch_size:
        parser.error("--buffer-size must be at least --batch-size")
    if args.mode == "distill" and not args.load_from_experiment:
        parser.error("--mode distill requires --load-from-experiment as the frozen teacher")
    if args.mode != "iterate" and args.teacher_update != "gated":
        parser.error("--teacher-update latest requires --mode iterate")
    if args.reset_loaded_value_head and not args.load_from_experiment:
        parser.error("--reset-loaded-value-head requires --load-from-experiment")
    if args.reset_loaded_value_head and args.resume:
        parser.error("--reset-loaded-value-head cannot be used with --resume")
    if not 0.0 <= args.promotion_threshold <= 1.0:
        parser.error("--promotion-threshold must be between 0 and 1")
    if args.max_baseline_regression < 0:
        parser.error("--max-baseline-regression cannot be negative")
    if not math.isfinite(args.soft_policy_weight) or args.soft_policy_weight < 0.0:
        parser.error("--soft-policy-weight must be finite and non-negative")
    if (
        not math.isfinite(args.soft_policy_temperature)
        or args.soft_policy_temperature < 0.0
        or (args.soft_policy_weight > 0.0 and args.soft_policy_temperature <= 0.0)
    ):
        parser.error("An enabled soft policy head requires a positive finite temperature")
    if not math.isfinite(args.value_scale) or args.value_scale < 0.0:
        parser.error("--value-scale must be finite and non-negative")
    if not 0.0 < args.full_search_probability <= 1.0:
        parser.error("--full-search-probability must be in (0, 1]")
    if args.full_search_probability < 1.0 and args.fast_simulations > args.simulations:
        parser.error("--fast-simulations cannot exceed --simulations")
    if not 0.0 <= args.search_value_weight_max <= 1.0:
        parser.error("--search-value-weight-max must be between 0 and 1")
    if args.search_value_weight_ramp_iterations < 0:
        parser.error("--search-value-weight-ramp-iterations cannot be negative")
    if not math.isfinite(args.c_visit) or args.c_visit < 0.0:
        parser.error("--c-visit must be finite and non-negative")
    if not math.isfinite(args.c_scale) or args.c_scale < 0.0:
        parser.error("--c-scale must be finite and non-negative")
    if args.policy_target == "completed-q" and args.self_play_backend != "cppanatron":
        parser.error("--policy-target completed-q currently requires cppanatron self-play")
    if args.search_selection == "completed-q" and args.self_play_backend != "cppanatron":
        parser.error("--search-selection completed-q currently requires cppanatron self-play")
    if args.explore_actions < 0:
        parser.error("--explore-actions cannot be negative")
    if args.trajectory_action_selection == "canopy":
        if args.self_play_backend != "cppanatron":
            parser.error("--trajectory-action-selection canopy requires cppanatron self-play")
        if args.policy_target != "completed-q" or args.search_selection != "completed-q":
            parser.error(
                "--trajectory-action-selection canopy requires completed-Q search and targets"
            )
        if args.target_temperature is None or not math.isclose(args.target_temperature, 1.0):
            parser.error("--trajectory-action-selection canopy requires --target-temperature 1")
    if args.value_loss_weight is None:
        args.value_loss_weight = 0.0 if args.mode == "distill" else 1.0
    if args.self_play_backend == "cppanatron":
        if args.ismcts_determinizations != 1:
            parser.error(
                "--self-play-backend cppanatron currently requires --ismcts-determinizations 1"
            )
        if args.prunning:
            parser.error("--self-play-backend cppanatron does not support --prunning")


def _build_model_pair(
    arch: ArchitecturePreset,
    *,
    device: str | torch.device,
) -> tuple[PolicyModel, CriticModel]:
    if arch.num_players is None:
        raise ValueError("Model construction requires a concrete player count.")
    map_type = cast(Literal["BASE", "MINI", "TOURNAMENT"], arch.map_type)
    actor_level = cast(ActorObservationLevel, arch.actor_observation_level)
    critic_level = cast(CriticObservationLevel, arch.critic_observation_level)
    if arch.network_mode == "shared":
        policy = build_policy_value_model(
            backbone_type=arch.backbone_type,
            model_type=arch.model_type,
            hidden_dims=arch.policy_hidden_dims,
            num_players=arch.num_players,
            map_type=map_type,
            actor_observation_level=actor_level,
            critic_observation_level=critic_level,
            device=device,
            xdim_cnn_channels=arch.xdim_cnn_channels,
            xdim_cnn_kernel_size=arch.xdim_cnn_kernel_size,
            xdim_fusion_hidden_dim=arch.xdim_policy_fusion_hidden_dim,
            value_head_type=arch.value_head_type,
        )
        return policy, None

    policy = build_policy_model(
        backbone_type=arch.backbone_type,
        model_type=arch.model_type,
        hidden_dims=arch.policy_hidden_dims,
        num_players=arch.num_players,
        map_type=map_type,
        actor_observation_level=actor_level,
        device=device,
        xdim_cnn_channels=arch.xdim_cnn_channels,
        xdim_cnn_kernel_size=arch.xdim_cnn_kernel_size,
        xdim_fusion_hidden_dim=arch.xdim_policy_fusion_hidden_dim,
    )
    critic = build_critic_model(
        backbone_type=arch.backbone_type,
        hidden_dims=arch.critic_hidden_dims,
        num_players=arch.num_players,
        map_type=map_type,
        critic_observation_level=critic_level,
        device=device,
        xdim_cnn_channels=arch.xdim_cnn_channels,
        xdim_cnn_kernel_size=arch.xdim_cnn_kernel_size,
        xdim_fusion_hidden_dim=arch.xdim_critic_fusion_hidden_dim,
    )
    return policy, critic


def _load_initial_weights(
    *,
    student_policy: PolicyModel,
    student_critic: CriticModel,
    teacher_policy: PolicyModel,
    teacher_critic: CriticModel,
    warm_start: TrainingWarmStart | None,
    reset_value_head: bool = False,
) -> None:
    fresh_value_state: dict[str, torch.Tensor] | None = None
    value_head: torch.nn.Module | None = None
    if reset_value_head:
        if warm_start is None:
            raise ValueError("reset_value_head requires a warm-start checkpoint")
        if isinstance(student_policy, PolicyValueNetworkWrapper):
            value_head = student_policy.value_head
        elif student_critic is not None:
            value_head = student_critic.value_head
        else:
            raise ValueError("Warm-started model does not expose a value head to reset")
        fresh_value_state = {
            key: value.detach().clone() for key, value in value_head.state_dict().items()
        }

    if warm_start is not None:
        checkpoints = warm_start.checkpoints
        policy_state = torch.load(checkpoints.policy, map_location="cpu")
        student_policy.load_state_dict(policy_state)
        if student_critic is not None:
            if checkpoints.critic is None:
                raise FileNotFoundError(
                    f"Experiment '{checkpoints.experiment_name}' has no paired critic "
                    f"for selector '{checkpoints.which}'."
                )
            critic_state = torch.load(checkpoints.critic, map_location="cpu")
            student_critic.load_state_dict(critic_state)
        if fresh_value_state is not None:
            assert value_head is not None
            value_head.load_state_dict(fresh_value_state)
        teacher_policy.load_state_dict(student_policy.state_dict())
        if student_critic is not None:
            assert teacher_critic is not None
            teacher_critic.load_state_dict(student_critic.state_dict())
        return

    teacher_policy.load_state_dict(student_policy.state_dict())
    if student_critic is not None:
        assert teacher_critic is not None
        teacher_critic.load_state_dict(student_critic.state_dict())


def _build_train_config(
    args: argparse.Namespace,
    config: AlphaZeroConfig,
    arch: ArchitecturePreset,
    device: str,
) -> dict[str, Any]:
    return {
        "algorithm": "mcts_distillation" if args.mode == "distill" else "expert_iteration",
        **architecture_train_config_fields(arch),
        **{key: value for key, value in vars(args).items() if key not in {"config", "wandb_tags"}},
        "device": device,
        "value_loss_weight": config.value_loss_weight,
    }


def _init_wandb(
    args: argparse.Namespace,
    train_config: dict[str, Any],
    grouping: dict[str, Any],
    resume: ResumeContext,
) -> bool:
    if not args.wandb:
        wandb.init(mode="disabled")
        return False
    kwargs: dict[str, Any] = {
        "project": args.wandb_project,
        "entity": args.wandb_entity,
        "name": args.wandb_run_name,
        "config": train_config,
        "job_type": args.mode,
        **grouping,
    }
    if resume.active and resume.wandb_run_id:
        kwargs.update(id=resume.wandb_run_id, resume="must")
    wandb.init(**kwargs)
    return True


def _wandb_info(args: argparse.Namespace) -> dict[str, Any]:
    if not args.wandb or wandb.run is None:
        return {}
    info: dict[str, Any] = {
        "project": args.wandb_project,
        "name": args.wandb_run_name,
        "id": wandb.run.id,
    }
    if wandb.run.tags:
        info["tags"] = list(wandb.run.tags)
    return info


def _network_specs(trainer: AlphaZeroTrainer, config: AlphaZeroConfig) -> dict[str, Any]:
    if trainer.uses_shared_network:
        return {
            "policy": network_spec_from_model(
                trainer.student_policy_model,
                kind=KIND_POLICY_VALUE,
                model_type=config.model_type,
                observation_level=config.actor_observation_level,
            )
        }
    assert trainer.student_critic_model is not None
    return {
        "policy": network_spec_from_model(
            trainer.student_policy_model,
            kind=KIND_POLICY,
            model_type=config.model_type,
            observation_level=config.actor_observation_level,
        ),
        "critic": network_spec_from_model(
            trainer.student_critic_model,
            kind=KIND_VALUE,
            observation_level=config.critic_observation_level,
        ),
    }


def _refresh_checkpoint_registry(
    experiment_name: str,
    checkpoint_dir: str,
    trainer: AlphaZeroTrainer,
) -> None:
    roles = ["policy"] if trainer.uses_shared_network else ["policy", "critic"]
    registry = build_checkpoint_registry(
        checkpoint_dir,
        roles_present=roles,
        relative_prefix=CHECKPOINTS_DIRNAME,
    )
    save_checkpoint_registry(experiment_dir(experiment_name), registry)


def _evaluate_student(trainer: AlphaZeroTrainer, config: AlphaZeroConfig, args) -> PolicyEvalResult:
    policy, _ = trainer.prepare_student_evaluation()
    return evaluate_policy_vs_value(
        policy_model=policy,
        model_type=config.model_type,
        map_type=config.map_type,
        actor_observation_level=config.actor_observation_level,
        num_players=config.num_players,
        num_games=args.eval_games,
        seed=args.eval_seed,
        vps_to_win=config.vps_to_win,
        discard_limit=config.discard_limit,
        show_tqdm=False,
    )


def _evaluate_h2h(trainer: AlphaZeroTrainer, config: AlphaZeroConfig, args) -> PolicyEvalResult:
    if config.num_players != 2:
        raise ValueError("Candidate/champion promotion evaluation currently requires 2 players.")
    candidate, _ = trainer.prepare_student_evaluation()
    champion, _ = trainer.teacher_evaluation_models()
    return evaluate_candidate_vs_champion(
        candidate_model=candidate,
        champion_model=champion,
        model_type=config.model_type,
        map_type=config.map_type,
        actor_observation_level=config.actor_observation_level,
        num_games=args.h2h_games,
        seed=args.h2h_seed,
        vps_to_win=config.vps_to_win,
        discard_limit=config.discard_limit,
        show_tqdm=False,
    )


def _mean_metrics(metrics: list[dict[str, float]]) -> dict[str, float]:
    if not metrics:
        return {}
    keys = set().union(*(metric.keys() for metric in metrics))
    return {key: sum(metric.get(key, 0.0) for metric in metrics) / len(metrics) for key in keys}


def _persist_training_state(
    *,
    path: str,
    trainer: AlphaZeroTrainer,
    mode: str,
    global_step: int,
    iteration: int,
    best_eval_score: float,
    best_iteration: int,
    champion_eval_score: float,
    promotions: int,
    wandb_enabled: bool,
) -> None:
    payload = {
        "algorithm": "search_guided_policy_iteration",
        "mode": mode,
        "global_step": global_step,
        "iteration": iteration,
        "best_eval_score": best_eval_score,
        "best_iteration": best_iteration,
        "champion_eval_score": champion_eval_score,
        "promotions": promotions,
        "trainer": trainer.state_dict(),
        "wandb_run_id": (wandb.run.id if wandb_enabled and wandb.run is not None else None),
        # Replay data is intentionally omitted: a 50k full-state buffer can be
        # multiple GB. Resumed runs refill it from the restored teacher.
        "replay_buffer_persisted": False,
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = f"{path}.tmp"
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _restore_training_state(
    trainer: AlphaZeroTrainer,
    resume: ResumeContext,
    mode: str,
) -> tuple[int, int, float, int, float, int] | None:
    state = resume.state
    if state is None:
        return None
    saved_mode = state.get("mode")
    if saved_mode is not None and saved_mode != mode:
        raise ValueError(f"Cannot resume mode '{saved_mode}' with --mode {mode}.")
    trainer_state = state.get("trainer")
    if not isinstance(trainer_state, dict):
        raise ValueError("Training state predates the rewritten search trainer and cannot resume.")
    trainer.load_state_dict(trainer_state)
    return (
        int(state.get("global_step", 0)),
        int(state.get("iteration", 0)),
        float(state.get("best_eval_score", float("-inf"))),
        int(state.get("best_iteration", 0)),
        float(state.get("champion_eval_score", float("-inf"))),
        int(state.get("promotions", 0)),
    )


def _print_eval(label: str, result: PolicyEvalResult) -> None:
    print(
        f"  {label}: {result.win_rate:.2%} ({result.wins}/{result.games}) | "
        f"first={result.first_win_rate:.2%}, second={result.second_win_rate:.2%}, "
        f"avg_vps={result.avg_vps:.3f}"
    )


def _update_search_teacher(
    *,
    trainer: AlphaZeroTrainer,
    strategy: str,
    candidate_win_rate: float | None,
    h2h_win_rate: float | None,
    champion_eval_score: float,
    promotion_threshold: float,
    max_baseline_regression: float,
) -> tuple[bool | None, float, str | None]:
    """Update the self-play teacher, returning acceptance, score, and reason."""
    if strategy == "latest":
        trainer.promote_student()
        score = (
            champion_eval_score
            if candidate_win_rate is None
            else candidate_win_rate
        )
        return True, score, "latest student becomes next iteration's search teacher"
    if strategy != "gated":
        raise ValueError(f"Unknown teacher update strategy: {strategy}")
    if candidate_win_rate is None or h2h_win_rate is None:
        return None, champion_eval_score, None

    decision = decide_promotion(
        h2h_win_rate=h2h_win_rate,
        candidate_baseline_win_rate=candidate_win_rate,
        champion_baseline_win_rate=champion_eval_score,
        h2h_threshold=promotion_threshold,
        max_baseline_regression=max_baseline_regression,
    )
    if decision.promote:
        trainer.promote_student()
        return True, candidate_win_rate, decision.reason
    trainer.restore_student_from_teacher()
    return False, champion_eval_score, decision.reason


def run_training(
    *,
    args: argparse.Namespace,
    config: AlphaZeroConfig,
    trainer: AlphaZeroTrainer,
    checkpoint_dir: str,
    experiment_name: str,
    training_state_path: str,
    resume: ResumeContext,
) -> None:
    restored = _restore_training_state(trainer, resume, args.mode) if resume.active else None
    if restored is None:
        global_step = 0
        start_iteration = 0
        promotions = 0
        initial_eval = _evaluate_student(trainer, config, args)
        _print_eval("Initial deployable policy vs F", initial_eval)
        best_eval_score = initial_eval.win_rate
        best_iteration = 0
        champion_eval_score = initial_eval.win_rate
        trainer.save(checkpoint_dir, "best")
        trainer.save(checkpoint_dir, "iter_0")
        _refresh_checkpoint_registry(experiment_name, checkpoint_dir, trainer)
        wandb.log(initial_eval.metrics("eval/candidate_vs_value") | {"iteration": 0}, step=0)
    else:
        (
            global_step,
            start_iteration,
            best_eval_score,
            best_iteration,
            champion_eval_score,
            promotions,
        ) = restored
        print(
            f"Restored iteration={start_iteration}, optimizer_step={global_step}, "
            f"best_eval={best_eval_score:.2%}, promotions={promotions}. "
            "Replay buffer will refill from the restored teacher."
        )

    final_iteration = start_iteration + args.iterations
    try:
        for iteration in range(start_iteration + 1, final_iteration + 1):
            print(f"\nIteration {iteration}/{final_iteration}: collecting teacher search targets")
            selfplay = trainer.collect_self_play(args.games_per_iteration)
            print(
                f"  collected {int(selfplay.get('experiences', 0)):,} decisions from "
                f"{int(selfplay.get('games', 0))} games; replay={int(selfplay.get('replay_size', 0)):,}"
            )
            wandb.log(
                {f"selfplay/{key}": value for key, value in selfplay.items()}
                | {"iteration": iteration},
                step=global_step,
            )

            updates: list[dict[str, float]] = []
            skipped = 0
            if args.optimizer_epochs > 0:
                optimizer_batches = trainer.iter_replay_epoch_batches(args.optimizer_epochs)
                optimizer_total = args.optimizer_epochs * math.ceil(
                    len(trainer.replay_buffer) / config.batch_size
                )
            else:
                optimizer_batches = (None for _ in range(args.optimizer_steps))
                optimizer_total = args.optimizer_steps
            for batch in tqdm(
                optimizer_batches,
                total=optimizer_total,
                desc="Student SGD",
                leave=False,
            ):
                metrics = trainer.update_weights(batch)
                if metrics is None:
                    skipped += 1
                    continue
                global_step += 1
                updates.append(metrics)
                wandb.log(
                    {f"train/{key}": value for key, value in metrics.items()}
                    | {"iteration": iteration},
                    step=global_step,
                )
            summary = _mean_metrics(updates)
            if summary:
                print(
                    f"  SGD: policy_loss={summary['policy_loss']:.4f}, "
                    f"value_loss={summary['value_loss']:.4f}, "
                    f"top1={summary['top1_agreement']:.2%}"
                )
                wandb.log(
                    {f"iteration/{key}": value for key, value in summary.items()}
                    | {"iteration": iteration},
                    step=global_step,
                )
            if skipped:
                print(f"  skipped {skipped} optimizer steps while the replay buffer warmed up")

            evaluation_due = (
                iteration % args.eval_every_iterations == 0 or iteration == final_iteration
            )
            candidate_eval: PolicyEvalResult | None = None
            h2h_eval: PolicyEvalResult | None = None
            eval_metrics: dict[str, float] | None = None
            if evaluation_due:
                candidate_eval = _evaluate_student(trainer, config, args)
                _print_eval("Candidate deployable policy vs F", candidate_eval)
                h2h_eval = _evaluate_h2h(trainer, config, args)
                _print_eval("Candidate vs teacher", h2h_eval)
                eval_metrics = (
                    candidate_eval.metrics("eval/candidate_vs_value")
                    | h2h_eval.metrics("eval/candidate_vs_teacher")
                    | {"iteration": float(iteration)}
                )

                if candidate_eval.win_rate > best_eval_score:
                    best_eval_score = candidate_eval.win_rate
                    best_iteration = iteration
                    trainer.save(checkpoint_dir, "best")
                    print(f"  saved new best deployable checkpoint ({best_eval_score:.2%})")

            if args.mode == "iterate":
                accepted, champion_eval_score, update_reason = _update_search_teacher(
                    trainer=trainer,
                    strategy=args.teacher_update,
                    candidate_win_rate=(
                        None if candidate_eval is None else candidate_eval.win_rate
                    ),
                    h2h_win_rate=None if h2h_eval is None else h2h_eval.win_rate,
                    champion_eval_score=champion_eval_score,
                    promotion_threshold=args.promotion_threshold,
                    max_baseline_regression=args.max_baseline_regression,
                )
                if accepted:
                    promotions += 1
                if update_reason is not None:
                    print(f"  teacher update: {update_reason}")
                if eval_metrics is not None and accepted is not None:
                    eval_metrics["promotion/accepted"] = float(accepted)
                    eval_metrics["promotion/count"] = float(promotions)
            if eval_metrics is not None:
                wandb.log(eval_metrics, step=global_step)

            if args.save_every_updates and iteration % args.save_every_updates == 0:
                trainer.save(checkpoint_dir, f"iter_{iteration}")
            _refresh_checkpoint_registry(experiment_name, checkpoint_dir, trainer)
            _persist_training_state(
                path=training_state_path,
                trainer=trainer,
                mode=args.mode,
                global_step=global_step,
                iteration=iteration,
                best_eval_score=best_eval_score,
                best_iteration=best_iteration,
                champion_eval_score=champion_eval_score,
                promotions=promotions,
                wandb_enabled=args.wandb,
            )
            if wandb.run is not None:
                wandb.run.summary["best_eval/win_rate_vs_value"] = best_eval_score
                wandb.run.summary["best_eval/iteration"] = best_iteration
                wandb.run.summary["promotion/count"] = promotions
    finally:
        trainer.close()


def main() -> None:
    args = parse_args()
    try:
        setup = resolve_training_architecture_and_warm_start(args)
        arch = setup.arch
        if arch.num_players is None:
            raise ValueError("Search-guided training requires game.num_players in metadata.")
        validate_ismcts_observation_levels(
            ismcts_determinizations=args.ismcts_determinizations,
            num_players=arch.num_players,
            actor_observation_level=cast(
                ActorObservationLevel, arch.actor_observation_level
            ),
            critic_observation_level=cast(
                CriticObservationLevel, arch.critic_observation_level
            ),
        )
        resume = prepare_resume(args, setup.warm_start)
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"Error: {exc}") from exc

    if resume.active:
        experiment_name = resume.experiment_name
        assert experiment_name is not None
        if args.wandb and not args.wandb_run_name:
            args.wandb_run_name = resume.wandb_run_name or experiment_name
    else:
        experiment_name = make_experiment_name(
            "mcts-distill" if args.mode == "distill" else "expert-iteration",
            args.wandb_run_name,
            args.experiment_name,
        )
        if args.wandb and not args.wandb_run_name:
            args.wandb_run_name = experiment_name

    checkpoint_dir = default_checkpoints_dir(experiment_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    state_path = training_state_file(experiment_name)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    config = AlphaZeroConfig(
        mode=args.mode,
        num_players=arch.num_players,
        map_type=cast(Literal["BASE", "MINI", "TOURNAMENT"], arch.map_type),
        actor_observation_level=cast(ActorObservationLevel, arch.actor_observation_level),
        critic_observation_level=cast(CriticObservationLevel, arch.critic_observation_level),
        network_mode=arch.network_mode,
        model_type=arch.model_type,
        vps_to_win=arch.vps_to_win,
        discard_limit=arch.discard_limit,
        self_play_backend=args.self_play_backend,
        simulations=args.simulations,
        c_puct=args.c_puct,
        value_scale=args.value_scale,
        tree_reuse=args.tree_reuse,
        canonical_pruning=args.canonical_pruning,
        search_selection=args.search_selection,
        policy_target=args.policy_target,
        c_visit=args.c_visit,
        c_scale=args.c_scale,
        full_search_probability=args.full_search_probability,
        fast_simulations=args.fast_simulations,
        search_value_weight_max=args.search_value_weight_max,
        search_value_weight_ramp_iterations=args.search_value_weight_ramp_iterations,
        prunning=args.prunning,
        ismcts_determinizations=args.ismcts_determinizations,
        temperature=args.temperature,
        final_temperature=args.final_temperature,
        target_temperature=args.target_temperature,
        temperature_drop_move=args.temperature_drop_move,
        noise_turns=args.noise_turns,
        dirichlet_alpha=args.dirichlet_alpha,
        dirichlet_frac=args.dirichlet_frac,
        trajectory_action_selection=args.trajectory_action_selection,
        explore_actions=args.explore_actions,
        num_game_workers=args.num_workers,
        inference_batch_size=args.inference_batch_size,
        inference_wait_ms=args.inference_wait_ms,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        policy_lr=args.policy_lr,
        critic_lr=args.critic_lr,
        weight_decay=args.weight_decay,
        policy_loss_weight=args.policy_loss_weight,
        value_loss_weight=args.value_loss_weight,
        soft_policy_temperature=args.soft_policy_temperature,
        soft_policy_weight=args.soft_policy_weight,
        max_grad_norm=args.max_grad_norm,
        offload_inactive_models=args.offload_inactive_models,
        device=device,
        seed=None if args.seed < 0 else args.seed,
    )

    # Construct on CPU so four large networks are never materialized on CUDA at once.
    student_policy, student_critic = _build_model_pair(arch, device="cpu")
    teacher_policy, teacher_critic = _build_model_pair(arch, device="cpu")
    _load_initial_weights(
        student_policy=student_policy,
        student_critic=student_critic,
        teacher_policy=teacher_policy,
        teacher_critic=teacher_critic,
        warm_start=setup.warm_start,
        reset_value_head=args.reset_loaded_value_head,
    )
    trainer = AlphaZeroTrainer(
        config,
        student_policy,
        student_critic,
        teacher_policy,
        teacher_critic,
    )

    train_config = _build_train_config(args, config, arch, device)
    grouping = wandb_grouping_kwargs(
        args,
        group_default="mcts-distillation" if args.mode == "distill" else "expert-iteration",
        warm_start=setup.warm_start,
        resume=resume,
    )
    _init_wandb(args, train_config, grouping, resume)

    if not resume.active:
        trainer.save(checkpoint_dir, "best")
        trainer.save(checkpoint_dir, "iter_0")
        save_experiment(
            experiment_name,
            checkpoint_dir,
            algorithm="mcts_distillation" if args.mode == "distill" else "expert_iteration",
            game=GameConfig(
                num_players=config.num_players,
                map_type=config.map_type,
                vps_to_win=config.vps_to_win,
                discard_limit=config.discard_limit,
            ),
            networks=_network_specs(trainer, config),
            train_config=train_config,
            wandb_info=_wandb_info(args),
            notes=(
                "Frozen neural-MCTS teacher distillation."
                if args.mode == "distill"
                else "Gated neural-MCTS Expert Iteration."
            ),
        )

    print("\nSearch-guided policy training")
    print(f"  mode={args.mode} | device={device} | architecture={setup.architecture_source}")
    print(
        f"  teacher={args.self_play_backend} "
        f"d{args.ismcts_determinizations} x s{args.simulations} | "
        f"games/iteration={args.games_per_iteration} | iterations={args.iterations}"
    )
    print(
        f"  student policy_lr={args.policy_lr:g} | value_weight={args.value_loss_weight:g} | "
        f"eval_seed={args.eval_seed}"
    )
    run_training(
        args=args,
        config=config,
        trainer=trainer,
        checkpoint_dir=checkpoint_dir,
        experiment_name=experiment_name,
        training_state_path=state_path,
        resume=resume,
    )
    _refresh_checkpoint_registry(experiment_name, checkpoint_dir, trainer)
    if args.wandb:
        wandb.finish()
    print(
        f"\nComplete. Experiment: {experiment_name}\nBest deployable checkpoint: {checkpoint_dir}"
    )


if __name__ == "__main__":
    main()
