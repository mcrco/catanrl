#!/usr/bin/env python3
"""Print the global self-play iteration represented by a saved checkpoint."""

from __future__ import annotations

import argparse

from catanrl.experiment_store import load_experiment
from catanrl.experiments.canopy_contract import canopy_checkpoint_global_iteration


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--checkpoint-step", type=int, required=True)
    args = parser.parse_args()
    experiment = load_experiment(args.experiment)
    print(canopy_checkpoint_global_iteration(experiment, args.checkpoint_step))


if __name__ == "__main__":
    main()
