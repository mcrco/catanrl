#!/usr/bin/env python3
"""
Benchmark key AlphaZero steps (feature extraction, model forward, game step,
and MCTS rollouts) to identify bottlenecks.
"""

from __future__ import annotations

import argparse
import random
import time
from typing import Dict, List, Sequence

import numpy as np
import torch

from catanrl.features.catanatron_utils import COLOR_ORDER, compute_feature_vector_dim

from catanrl.models.models import build_hierarchical_policy_value_network


def _maybe_sync(device: str | torch.device) -> None:
    if torch.cuda.is_available() and device is not None and "cuda" in str(device):
        torch.cuda.synchronize()


def bench_model_forward(
    model: torch.nn.Module,
    device: str,
    input_dim: int,
    batch_sizes: Sequence[int],
    repeats: int,
    warmup: int,
) -> List[Dict[str, float]]:
    results: List[Dict[str, float]] = []
    model.to(device)
    model.eval()
    for bs in batch_sizes:
        x = torch.randn(bs, input_dim, device=device, dtype=torch.float32)
        with torch.no_grad():
            for _ in range(warmup):
                model(x)
        _maybe_sync(device)
        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(repeats):
                model(x)
        _maybe_sync(device)
        elapsed = time.perf_counter() - t0
        per_batch_ms = (elapsed / repeats) * 1000.0
        per_item_ms = per_batch_ms / float(bs)
        results.append(
            {
                "batch_size": bs,
                "per_batch_ms": per_batch_ms,
                "per_item_ms": per_item_ms,
            }
        )
    return results


def bench_model_forward_and_get_flat_action_logits(
    model: torch.nn.Module,
    device: str,
    input_dim: int,
    batch_sizes: Sequence[int],
    repeats: int,
    warmup: int,
) -> List[Dict[str, float]]:
    results: List[Dict[str, float]] = []
    model.to(device)
    model.eval()
    for bs in batch_sizes:
        x = torch.randn(bs, input_dim, device=device, dtype=torch.float32)
        with torch.no_grad():
            for _ in range(warmup):
                model(x)
        _maybe_sync(device)
        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(repeats):
                action_type_logits, param_logits = model(x)
                _ = model.get_flat_action_logits(action_type_logits, param_logits)
        _maybe_sync(device)
        elapsed = time.perf_counter() - t0
        per_batch_ms = (elapsed / repeats) * 1000.0
        per_item_ms = per_batch_ms / float(bs)
        results.append(
            {
                "batch_size": bs,
                "per_batch_ms": per_batch_ms,
                "per_item_ms": per_item_ms,
            }
        )
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile Neural Network and feature extraction",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--map-type", type=str, default="BASE", choices=["BASE", "MINI", "TOURNAMENT"]
    )
    parser.add_argument("--hidden-dims", type=str, default="512,512")
    parser.add_argument("--device", type=str, default="cuda", help="cpu, cuda, or cuda:0 style")
    parser.add_argument(
        "--batch-sizes",
        type=str,
        default="1,4,16,32,64,128,256,512,1024,4096,8192,16384,32768",
        help="Comma-separated batch sizes for model forward timing",
    )
    parser.add_argument(
        "--forward-repeats", type=int, default=50, help="Timed repeats per batch size"
    )
    parser.add_argument(
        "--forward-warmup", type=int, default=10, help="Warmup iterations per batch size"
    )
    parser.add_argument(
        "--feature-samples",
        type=int,
        default=10000,
        help="Number of feature extractions to average",
    )
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")
    return parser.parse_args()


def parse_int_list(value: str) -> List[int]:
    return [int(v.strip()) for v in value.split(",") if v.strip()]


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    hidden_dims = parse_int_list(args.hidden_dims)
    batch_sizes = parse_int_list(args.batch_sizes)

    input_dim = compute_feature_vector_dim(len(COLOR_ORDER), args.map_type)
    model = build_hierarchical_policy_value_network(input_dim=input_dim, hidden_dims=hidden_dims)

    print("== AlphaZero timing profile ==")
    print(f"Device: {device}")
    print(f"Map: {args.map_type} | Players: {len(COLOR_ORDER)}")
    print(f"Input dim: {input_dim} | Hidden dims: {hidden_dims}")

    print("\nModel forward (hierarchical wrapper):")
    forward_stats = bench_model_forward(
        model=model,
        device=device,
        input_dim=input_dim,
        batch_sizes=batch_sizes,
        repeats=args.forward_repeats,
        warmup=args.forward_warmup,
    )
    for stat in forward_stats:
        print(
            f"  batch={stat['batch_size']:>4} | "
            f"per_batch={stat['per_batch_ms']:.3f} ms | "
            f"per_item={stat['per_item_ms']:.4f} ms"
        )

    print("\nModel forward and get flat action logits (hierarchical wrapper):")
    forward_and_get_fal_stats = bench_model_forward_and_get_flat_action_logits(
        model=model,
        device=device,
        input_dim=input_dim,
        batch_sizes=batch_sizes,
        repeats=args.forward_repeats,
        warmup=args.forward_warmup,
    )
    for stat in forward_and_get_fal_stats:
        print(
            f"  batch={stat['batch_size']:>4} | "
            f"per_batch={stat['per_batch_ms']:.3f} ms | "
            f"per_item={stat['per_item_ms']:.4f} ms"
        )


if __name__ == "__main__":
    main()
