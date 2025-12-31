#!/usr/bin/env python3
"""
Benchmark how often the ParallelAlphaZeroTrainer inference server receives requests
for different worker counts, batch sizes, and wait times.
"""

from __future__ import annotations

import argparse
import contextlib
import queue
import random
import threading
import time
from dataclasses import replace
from typing import Dict, Iterable, List, Sequence

import numpy as np
import torch

from catanatron.gym.envs.catanatron_env import ACTION_SPACE_SIZE
from catanrl.algorithms.alphazero import parallel_trainer as parallel_mod
from catanrl.algorithms.alphazero.trainer import AlphaZeroConfig
from catanrl.features.catanatron_utils import compute_feature_vector_dim
from catanrl.models.models import build_hierarchical_policy_value_network

ParallelAlphaZeroTrainer = parallel_mod.ParallelAlphaZeroTrainer
_BaseInferenceServer = parallel_mod._InferenceServer


def _maybe_sync(device: str | torch.device) -> None:
    if torch.cuda.is_available() and device is not None and "cuda" in str(device):
        torch.cuda.synchronize()


class InferenceQueueStats:
    """Thread-safe collector for inference queue metrics."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.arrival_gaps: List[float] = []
        self.batch_sizes: List[int] = []
        self.batch_waits: List[float] = []
        self.backlog_samples: List[int] = []
        self.total_batches = 0
        self.total_requests = 0

    def record_arrival(self, delta_seconds: float) -> None:
        if delta_seconds <= 0:
            return
        with self._lock:
            self.arrival_gaps.append(delta_seconds)

    def record_batch(self, size: int, wait_seconds: float, backlog: int | None) -> None:
        with self._lock:
            self.batch_sizes.append(size)
            self.batch_waits.append(wait_seconds)
            if backlog is not None:
                self.backlog_samples.append(backlog)
            self.total_batches += 1
            self.total_requests += size

    @staticmethod
    def _describe(values: Sequence[float], scale: float = 1.0) -> Dict[str, float]:
        if not values:
            return {"avg": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}
        arr = np.array(values, dtype=np.float64)
        return {
            "avg": float(arr.mean() * scale),
            "p50": float(np.percentile(arr, 50) * scale),
            "p95": float(np.percentile(arr, 95) * scale),
            "max": float(arr.max() * scale),
        }

    def summary(self) -> Dict[str, Dict[str, float] | int]:
        with self._lock:
            arrivals = list(self.arrival_gaps)
            batch_sizes = list(self.batch_sizes)
            waits = list(self.batch_waits)
            backlog = list(self.backlog_samples)
            total_batches = self.total_batches
            total_requests = self.total_requests
        return {
            "total_batches": total_batches,
            "total_requests": total_requests,
            "arrival_gap_ms": self._describe(arrivals, scale=1000.0),
            "batch_size": self._describe(batch_sizes),
            "batch_wait_ms": self._describe(waits, scale=1000.0),
            "backlog": self._describe(backlog),
        }


def make_profiling_inference_server(
    base_cls: type[_BaseInferenceServer],
    stats: InferenceQueueStats,
    mock_inference: bool,
) -> type[_BaseInferenceServer]:
    class ProfilingInferenceServer(base_cls):  # type: ignore[misc]
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self._stats = stats
            self._mock = mock_inference

        def _safe_qsize(self) -> int | None:
            if not hasattr(self.inference_queue, "qsize"):
                return None
            try:
                return int(self.inference_queue.qsize())
            except (NotImplementedError, OSError):
                return None

        def run(self) -> None:
            last_arrival = None
            while not self._stop_event.is_set():
                try:
                    first = self.inference_queue.get(timeout=0.1)
                except queue.Empty:
                    continue

                if first is None:
                    break

                now = time.perf_counter()
                if last_arrival is not None:
                    self._stats.record_arrival(now - last_arrival)
                last_arrival = now

                batch = [first]
                wait_start = now
                deadline = time.time() + (self.max_wait_ms / 1000.0)
                while len(batch) < self.max_batch_size and time.time() < deadline:
                    remaining = max(0.0, deadline - time.time())
                    if remaining <= 0:
                        break
                    try:
                        item = self.inference_queue.get(timeout=remaining)
                    except queue.Empty:
                        break
                    if item is None:
                        self.inference_queue.put(None)
                        break
                    arrival_now = time.perf_counter()
                    if last_arrival is not None:
                        self._stats.record_arrival(arrival_now - last_arrival)
                    last_arrival = arrival_now
                    batch.append(item)

                if self._mock:
                    response_policy = np.full(
                        ACTION_SPACE_SIZE, 1.0 / ACTION_SPACE_SIZE, dtype=np.float32
                    )
                    for entry in batch:
                        worker_id = entry["worker_id"]
                        self.response_queues[worker_id].put((response_policy.copy(), 0.0))
                else:
                    features = np.stack([entry["features"] for entry in batch], axis=0)
                    states = torch.from_numpy(features).float().to(self.trainer.device)
                    self.trainer.model.eval()
                    _maybe_sync(self.trainer.device)
                    with torch.no_grad():
                        log_probs, values = self.trainer._model_forward(states)
                        probs = torch.exp(log_probs).cpu().numpy()
                        values_np = values.view(-1).cpu().numpy()
                    _maybe_sync(self.trainer.device)

                    for entry, policy, value in zip(batch, probs, values_np):
                        worker_id = entry["worker_id"]
                        response = (policy.astype(np.float32), float(value))
                        self.response_queues[worker_id].put(response)

                backlog = self._safe_qsize()
                self._stats.record_batch(len(batch), time.perf_counter() - wait_start, backlog)

    return ProfilingInferenceServer


@contextlib.contextmanager
def patched_inference_server(stats: InferenceQueueStats, mock_inference: bool):
    original_cls = parallel_mod._InferenceServer
    parallel_mod._InferenceServer = make_profiling_inference_server(
        original_cls, stats, mock_inference
    )
    try:
        yield
    finally:
        parallel_mod._InferenceServer = original_cls


def parse_int_list(value: str) -> List[int]:
    return [int(token.strip()) for token in value.split(",") if token.strip()]


def run_benchmark(
    base_config: AlphaZeroConfig,
    hidden_dims: Sequence[int],
    input_dim: int,
    worker_counts: Iterable[int],
    simulation_counts: Iterable[int],
    duration_seconds: float,
    games_per_worker: int,
    inference_batch_size: int,
    inference_wait_ms: float,
    mock_inference: bool,
) -> None:
    for sims in simulation_counts:
        print(f"\n##### Simulations per move: {sims} #####")
        config = replace(base_config, simulations=sims)
        for workers in worker_counts:
            if workers < 2:
                print(f"\nSkipping workers={workers}: requires >= 2 workers.")
                continue
            stats = InferenceQueueStats()
            model = build_hierarchical_policy_value_network(
                input_dim=input_dim, hidden_dims=list(hidden_dims)
            )
            trainer: ParallelAlphaZeroTrainer | None = None
            with patched_inference_server(stats, mock_inference):
                try:
                    trainer = ParallelAlphaZeroTrainer(
                        config=config,
                        model=model,
                        num_workers=workers,
                        inference_batch_size=inference_batch_size,
                        inference_max_wait_ms=inference_wait_ms,
                    )
                    num_games = max(1, workers * games_per_worker)
                    print(f"\n=== workers={workers} | run for ~{duration_seconds:.1f}s ===")
                    start = time.perf_counter()
                    deadline = start + duration_seconds
                    games_finished = 0
                    iterations = 0
                    while True:
                        play_stats = trainer.self_play(num_games)
                        iterations += 1
                        games_finished += int(play_stats.get("games", num_games))
                        if time.perf_counter() >= deadline:
                            break
                    elapsed = time.perf_counter() - start
                    throughput = games_finished / elapsed if elapsed > 0 else 0.0
                    summary = stats.summary()
                    print(
                        f"Duration: {elapsed:.2f}s | games completed: {games_finished} "
                        f"| throughput: {throughput:.2f} games/s | batches observed: {summary['total_batches']}"
                    )
                    print(f"Self-play iterations: {iterations} x {num_games} games/requested")
                    print(f"Requests processed: {summary['total_requests']}")
                    batch_stats = summary["batch_size"]
                    wait_stats = summary["batch_wait_ms"]
                    arrival_stats = summary["arrival_gap_ms"]
                    backlog_stats = summary["backlog"]
                    print(
                        "Batch size — "
                        f"avg {batch_stats['avg']:.2f}, p95 {batch_stats['p95']:.2f}, max {batch_stats['max']:.0f}"
                    )
                    print(
                        "Batch wait — "
                        f"avg {wait_stats['avg']:.3f} ms, p95 {wait_stats['p95']:.3f} ms, max {wait_stats['max']:.3f} ms"
                    )
                    print(
                        "Arrival gap — "
                        f"avg {arrival_stats['avg']:.3f} ms, p95 {arrival_stats['p95']:.3f} ms"
                    )
                    print(
                        "Queue backlog (approx qsize) — "
                        f"avg {backlog_stats['avg']:.2f}, max {backlog_stats['max']:.0f}"
                    )
                finally:
                    if trainer is not None:
                        trainer.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile inference queue arrivals for ParallelAlphaZeroTrainer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--map-type", type=str, default="BASE", choices=["BASE", "MINI", "TOURNAMENT"])
    parser.add_argument("--num-players", type=int, default=2, choices=[2, 3, 4])
    parser.add_argument("--hidden-dims", type=str, default="512,512")
    parser.add_argument("--device", type=str, default="cuda", help="cpu, cuda, or cuda:0 style")
    parser.add_argument(
        "--simulation-counts",
        type=str,
        default="64",
        help="Comma-separated MCTS simulation counts to sweep",
    )
    parser.add_argument("--c-puct", type=float, default=1.5, help="PUCT exploration constant")
    parser.add_argument("--workers", type=str, default="2,4", help="Comma-separated worker counts to profile")
    parser.add_argument(
        "--duration",
        type=float,
        default=10.0,
        help="Seconds to run each worker/simulation configuration",
    )
    parser.add_argument("--games-per-worker", type=int, default=2, help="Self-play games per worker per request")
    parser.add_argument("--inference-batch-size", type=int, default=64, help="Inference server max batch size")
    parser.add_argument("--inference-wait-ms", type=float, default=2.0, help="Inference server max wait in ms")
    parser.add_argument(
        "--mock-inference",
        action="store_true",
        help="Bypass neural inference to isolate simulation speed",
    )
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    hidden_dims = parse_int_list(args.hidden_dims)
    worker_counts = parse_int_list(args.workers)
    simulation_counts = parse_int_list(args.simulation_counts)

    config = AlphaZeroConfig(
        num_players=args.num_players,
        map_type=args.map_type,
        c_puct=args.c_puct,
        device=device,
        seed=args.seed,
    )
    input_dim = compute_feature_vector_dim(config.num_players, config.map_type)

    print("== Parallel inference queue profile ==")
    print(f"Device: {device}")
    print(f"Workers: {worker_counts}")
    print(f"Inference batch size: {args.inference_batch_size} | wait: {args.inference_wait_ms} ms")

    run_benchmark(
        base_config=config,
        hidden_dims=hidden_dims,
        input_dim=input_dim,
        worker_counts=worker_counts,
        simulation_counts=simulation_counts,
        duration_seconds=args.duration,
        games_per_worker=args.games_per_worker,
        inference_batch_size=args.inference_batch_size,
        inference_wait_ms=args.inference_wait_ms,
        mock_inference=args.mock_inference,
    )


if __name__ == "__main__":
    main()


