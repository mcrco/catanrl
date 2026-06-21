"""Shared CLI arguments and parsers for training entrypoints."""

from __future__ import annotations

import argparse

DEFAULT_MAX_GRAD_NORM = 1.0
DEFAULT_METRIC_WINDOW = 200
DEFAULT_TREND_EVAL_SEED = 43
DEFAULT_WANDB_PROJECT = "catan"


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
