"""Shared CLI arguments and parsers for training entrypoints."""

from __future__ import annotations

import argparse
from typing import cast

from ..utils.catanatron_map import (
    DEFAULT_NUMBER_PLACEMENT,
    NUMBER_PLACEMENT_CHOICES,
    NumberPlacement,
)

DEFAULT_MAX_GRAD_NORM = 1.0
DEFAULT_METRIC_WINDOW = 200
DEFAULT_EVAL_SEED = 67
DEFAULT_TREND_EVAL_SEED = DEFAULT_EVAL_SEED
DEFAULT_WANDB_PROJECT = "catan"


def add_number_placement_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--number-placement",
        choices=NUMBER_PLACEMENT_CHOICES,
        default=argparse.SUPPRESS,
        help=(
            "Number-token placement for generated boards. 'official_spiral' uses the "
            "official token sequence while still shuffling terrain and ports; 'random' "
            "also shuffles number tokens. New runs and evals default to official_spiral; "
            "resumed runs and --experiment evals inherit their saved setting unless "
            "this flag is passed"
        ),
    )


def resolve_number_placement(
    args: argparse.Namespace,
    *,
    resume_active: bool = False,
    saved_number_placement: str | None = None,
    experiment_number_placement: str | None = None,
) -> NumberPlacement:
    """Resolve board token placement for a new run, resume, or eval.

    ``--number-placement`` uses ``argparse.SUPPRESS``, so omitting the flag is
    distinguishable from passing it. New runs default to official spiral. Resumes
    keep the saved experiment value and reject a conflicting flag. Eval scripts
    inherit from ``--experiment`` when the flag is omitted. Any placement is
    accepted for any map type; TOURNAMENT boards are fully fixed and ignore
    the setting (see ``catanrl.utils.catanatron_map``).
    """
    requested = getattr(args, "number_placement", None)
    if resume_active:
        if saved_number_placement is None:
            raise ValueError(
                "--resume requires a saved number_placement on the source experiment."
            )
        if requested is not None and requested != saved_number_placement:
            raise ValueError(
                "--resume cannot change number placement from "
                f"{saved_number_placement!r} to {requested!r}. "
                "Start a new warm-start run to change the board distribution."
            )
        placement = saved_number_placement
    elif requested is not None:
        placement = requested
    elif experiment_number_placement is not None:
        placement = experiment_number_placement
    else:
        placement = DEFAULT_NUMBER_PLACEMENT
    if placement not in NUMBER_PLACEMENT_CHOICES:
        raise ValueError(f"Unknown number placement {placement!r}.")
    return cast(NumberPlacement, placement)


def apply_number_placement(
    args: argparse.Namespace,
    *,
    resume_active: bool = False,
    saved_number_placement: str | None = None,
    experiment_number_placement: str | None = None,
) -> NumberPlacement:
    """Resolve placement and store it on ``args.number_placement``."""
    placement = resolve_number_placement(
        args,
        resume_active=resume_active,
        saved_number_placement=saved_number_placement,
        experiment_number_placement=experiment_number_placement,
    )
    args.number_placement = placement
    return placement


def add_device_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device (cpu, cuda, cuda:0, ...). Default: auto",
    )


def add_experiment_name_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Experiment name (folder under experiments/ and W&B run name). "
        "Defaults to --wandb-run-name, else an auto-generated name.",
    )


def add_wandb_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument(
        "--wandb-project",
        type=str,
        default=DEFAULT_WANDB_PROJECT,
        help="Weights & Biases project name",
    )
    parser.add_argument(
        "--wandb-run-name",
        type=str,
        default=None,
        help="Weights & Biases run name",
    )
    parser.add_argument(
        "--wandb-group",
        type=str,
        default=None,
        help=(
            "W&B group for this run. Defaults to the algorithm family "
            "(e.g. dagger, sarl-ppo, marl-ppo, alphazero) so all runs of one "
            "method aggregate together within the shared project."
        ),
    )
    parser.add_argument(
        "--wandb-tags",
        type=str,
        nargs="*",
        default=None,
        help="Extra W&B tags. Warm-start/resume lineage tags are added automatically.",
    )


def add_reward_function_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--reward-function",
        type=str,
        default="shaped",
        choices=["shaped", "win"],
        help="Reward function type",
    )


def add_train_epochs_argument(parser: argparse.ArgumentParser, *, default: int) -> None:
    parser.add_argument(
        "--train-epochs",
        type=int,
        default=default,
        help="Training epochs per update/iteration",
    )


def add_save_every_updates_argument(
    parser: argparse.ArgumentParser,
    *,
    default: int = 1,
    help: str = "Save checkpoints every N updates/iterations",
) -> None:
    parser.add_argument(
        "--save-every-updates",
        type=int,
        default=default,
        help=help,
    )


def add_fresh_eval_arguments(
    parser: argparse.ArgumentParser,
    *,
    fresh_default: int = 0,
    trend_default: int | None = None,
    eval_every_default: int = 0,
) -> None:
    parser.add_argument(
        "--fresh-eval-games-per-opponent",
        type=int,
        default=fresh_default,
        help="Fresh evaluation games per opponent; 0 disables fresh eval",
    )
    parser.add_argument(
        "--trend-eval-games-per-opponent",
        type=int,
        default=trend_default,
        help="Fixed-seed trend evaluation games per opponent (default: use fresh-eval-games-per-opponent)",
    )
    parser.add_argument(
        "--trend-eval-seed",
        type=int,
        default=DEFAULT_TREND_EVAL_SEED,
        help="Seed for trend evaluation runs",
    )
    parser.add_argument(
        "--eval-every-updates",
        type=int,
        default=eval_every_default,
        help="Run evaluation every N updates/iterations; 0 disables eval",
    )
