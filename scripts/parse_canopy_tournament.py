#!/usr/bin/env python3
"""Convert an official Canopy tournament log into a stable JSON result."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from catanrl.eval.canopy_reference import parse_canopy_tournament_summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path, help="Combined stdout/stderr from Canopy")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--opponent", default="random")
    parser.add_argument("--simulations", type=int, required=True)
    parser.add_argument("--max-actions", type=int, default=2000)
    args = parser.parse_args()
    if args.simulations < 1:
        parser.error("--simulations must be at least 1")
    if args.max_actions < 1:
        parser.error("--max-actions must be at least 1")
    return args


def main() -> None:
    args = _parse_args()
    summary = parse_canopy_tournament_summary(args.log.read_text(errors="replace"))
    payload = {
        "schema_version": 1,
        "implementation": "cullback/canopy",
        "config": {
            "checkpoint": args.checkpoint,
            "opponent": args.opponent,
            "simulations": args.simulations,
            "max_actions": args.max_actions,
            "random_dice": True,
            "vps_to_win": 15,
            "discard_limit": 9,
            "seat_balancing": "alternating",
            "tree_reuse": True,
            "dirichlet_alpha": 0.05,
            "dirichlet_epsilon": 0.25,
        },
        "summary": summary.payload(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    print(f"Canopy reference result: {args.output}")


if __name__ == "__main__":
    main()
