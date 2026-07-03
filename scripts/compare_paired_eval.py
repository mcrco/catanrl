#!/usr/bin/env python3
"""Compare two matched evaluation result files with exact McNemar tests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import wandb

from catanrl.eval.paired_comparison import compare_paired_payloads, load_paired_results
from catanrl.experiments.common_args import DEFAULT_WANDB_PROJECT


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a", required=True, help="Paired-results JSON for variant A")
    parser.add_argument("--b", required=True, help="Paired-results JSON for variant B")
    parser.add_argument("--output-json", default=None, help="Optional path for comparison JSON")
    parser.add_argument("--wandb", action="store_true", help="Log McNemar results to W&B")
    parser.add_argument("--wandb-project", default=DEFAULT_WANDB_PROJECT)
    parser.add_argument("--wandb-group", default="paired-eval-comparison")
    parser.add_argument("--wandb-run-name", default=None)
    args = parser.parse_args()

    payload_a = load_paired_results(args.a)
    payload_b = load_paired_results(args.b)
    comparisons = compare_paired_payloads(payload_a, payload_b)

    print(f"A: {payload_a['agent']} ({payload_a['checkpoint']})")
    print(f"B: {payload_b['agent']} ({payload_b['checkpoint']})")
    for label, result in comparisons.items():
        print(
            f"{label}: A={result['win_rate_a']:.3%}, B={result['win_rate_b']:.3%}, "
            f"delta={result['win_rate_difference']:+.3%}, "
            f"discordant={result['discordant']} "
            f"(A-only={result['a_only_win']}, B-only={result['b_only_win']}), "
            f"exact p={result['p_value_exact_two_sided']:.6g}"
        )

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(comparisons, indent=2, sort_keys=True) + "\n")

    if args.wandb:
        run = wandb.init(
            project=args.wandb_project,
            group=args.wandb_group,
            name=args.wandb_run_name,
            job_type="eval-comparison",
            config={
                "results_a": args.a,
                "results_b": args.b,
                "agent_a": payload_a["agent"],
                "agent_b": payload_b["agent"],
                "checkpoint_a": payload_a["checkpoint"],
                "checkpoint_b": payload_b["checkpoint"],
                "scenario": payload_a["scenario"],
            },
        )
        run.summary.update(
            {
                f"mcnemar/{label}/{metric}": value
                for label, result in comparisons.items()
                for metric, value in result.items()
            }
        )
        run.finish()


if __name__ == "__main__":
    main()
