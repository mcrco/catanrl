#!/usr/bin/env python3
"""Fail fast when a Canopy-parity launch would change its Catan contract."""

from __future__ import annotations

import argparse

from catanrl.experiment_store import load_experiment
from catanrl.experiments.architecture_config import load_architecture_preset
from catanrl.experiments.canopy_contract import (
    validate_canopy_architecture,
    validate_canopy_experiment,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--config", help="Fresh model architecture YAML")
    source.add_argument("--experiment", help="Warm-start/evaluation experiment")
    parser.add_argument("--which", default="best", help="Checkpoint selector to verify")
    parser.add_argument(
        "--require-terminal-dagger",
        action="store_true",
        help="Require cppanatron DAgger with win-only reward and gamma=1",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.config:
        if args.require_terminal_dagger:
            raise SystemExit("--require-terminal-dagger requires --experiment")
        validate_canopy_architecture(load_architecture_preset(args.config))
        print(f"Canopy parity contract verified for config: {args.config}")
        return

    experiment = load_experiment(args.experiment)
    validate_canopy_experiment(
        experiment,
        require_terminal_dagger=args.require_terminal_dagger,
    )
    experiment.resolve_checkpoint(args.which, "policy")
    print(
        f"Canopy parity contract verified for experiment: {experiment.metadata.name} ({args.which})"
    )


if __name__ == "__main__":
    main()
