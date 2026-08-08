#!/usr/bin/env python3
"""Test a native CatanRL result for non-inferiority to released Canopy."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping

from catanrl.eval.canopy_parity import (
    compare_independent_win_rates,
    validate_matching_action_cap,
)
from catanrl.eval.canopy_reference import validate_official_nexus_v3_reference


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument("--noninferiority-margin", type=float, default=0.05)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.budget < 1:
        parser.error("--budget must be at least 1")
    return args


def _require_equal(label: str, candidate: object, reference: object) -> None:
    if candidate != reference:
        raise ValueError(f"Incomparable {label}: candidate={candidate!r}, reference={reference!r}")


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _number(mapping: Mapping[str, Any], key: str) -> float:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{key} must be numeric, got {value!r}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{key} must be finite, got {value!r}")
    return result


def main() -> None:
    args = _parse_args()
    candidate_payload = _mapping(json.loads(args.candidate.read_text()), "candidate payload")
    reference_payload = _mapping(json.loads(args.reference.read_text()), "reference payload")
    candidate_config = _mapping(candidate_payload.get("config"), "candidate config")
    reference_config = _mapping(reference_payload.get("config"), "reference config")
    validate_official_nexus_v3_reference(reference_config)

    if candidate_config.get("game_opponent") != "random":
        raise ValueError("Candidate must be a native search-vs-random result")
    if reference_config.get("opponent") != "random":
        raise ValueError("Reference must be a Canopy search-vs-random result")
    _require_equal("simulation budget", args.budget, reference_config.get("simulations"))
    candidate_budgets = candidate_config.get("budgets")
    if not isinstance(candidate_budgets, list) or args.budget not in candidate_budgets:
        raise ValueError(f"Candidate does not contain requested budget {args.budget}")
    for key in ("vps_to_win", "discard_limit"):
        _require_equal(key, candidate_config.get(key), reference_config.get(key))
    validate_matching_action_cap(candidate_config, reference_config)
    for key in ("c_puct", "c_visit", "c_scale"):
        _require_equal(key, _number(candidate_config, key), _number(reference_config, key))
    _require_equal(
        "Dirichlet alpha",
        _number(candidate_config, "root_dirichlet_alpha"),
        _number(reference_config, "dirichlet_alpha"),
    )
    _require_equal(
        "Dirichlet fraction",
        _number(candidate_config, "root_dirichlet_fraction"),
        _number(reference_config, "dirichlet_epsilon"),
    )
    if candidate_config.get("search_selection") != "completed-q":
        raise ValueError("Candidate must use completed-Q search selection")
    if reference_config.get("random_dice") is not True:
        raise ValueError("Reference must use random dice")

    sweeps = _mapping(candidate_payload.get("game_sweeps"), "candidate game_sweeps")
    candidate_budget = _mapping(sweeps.get(str(args.budget)), "candidate budget result")
    candidate_summary = _mapping(candidate_budget.get("summary"), "candidate summary")
    reference_summary = _mapping(reference_payload.get("summary"), "reference summary")
    comparison = compare_independent_win_rates(
        candidate_summary,
        reference_summary,
        noninferiority_margin=args.noninferiority_margin,
    )
    output = {
        "schema_version": 1,
        "candidate": str(args.candidate),
        "reference": str(args.reference),
        "budget": args.budget,
        "comparison": comparison.payload(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output["comparison"], indent=2, sort_keys=True))
    if not comparison.noninferior:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
