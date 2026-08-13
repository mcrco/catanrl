#!/usr/bin/env python3
"""Compare released Canopy and CatanRL search on matched Catanatron games."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from catanrl.eval.canopy_parity_comparison import compare_canopy_to_native


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canopy", type=Path, required=True)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--budget", type=int, default=None)
    parser.add_argument("--noninferiority-margin", type=float, default=0.05)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    result = compare_canopy_to_native(
        json.loads(args.canopy.read_text()),
        json.loads(args.native.read_text()),
        budget=args.budget,
        noninferiority_margin=args.noninferiority_margin,
    )
    print(json.dumps(result, indent=2))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
