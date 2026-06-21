#!/usr/bin/env python3
"""
Benchmark NN MCTS parallelization strategies:

1. within-tree: one tree, ``num_search_workers`` fork processes (shared tree),
   central batched inference via a CUDA-safe coordinator subprocess
2. across-games: ``num_game_workers`` independent games, central batched inference

Example (from repo root, with PyTorch available):

    uv run --extra cu130 python scripts/profile_nn_mcts_parallel.py \\
        --num-simulations 128 --decisions 20 --parallel-workers 2,4,8 \\
        --inference-batch-size 32 --mode both

Use ``--policy-weights`` / ``--critic-weights`` to match your eval checkpoint.
MINI map and MLP keep startup light for quick A/B tests.
"""

from __future__ import annotations

import argparse
import cProfile
import gc
import io
import multiprocessing as mp
import pstats
import queue
import random
import threading
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass
from statistics import mean, pstdev
from typing import Iterator, Literal, Sequence

import numpy as np
import torch
from catanatron.game import Game
from catanatron.models.player import RandomPlayer

from catanrl.envs.puffer.common import compute_single_agent_dims
from catanrl.features.catanatron_utils import (
    ActorObservationLevel,
    COLOR_ORDER,
    CriticObservationLevel,
)
from catanrl.models.backbones import (
    BackboneConfig,
    CrossDimensionalBackboneConfig,
    MLPBackboneConfig,
)
from catanrl.models.models import (
    build_flat_policy_network,
    build_hierarchical_policy_network,
    build_value_network,
)
from catanrl.models.wrappers import PolicyNetworkWrapper, ValueNetworkWrapper
from catanrl.players import NNMCTSPlayer
from catanrl.players.nn_mcts_player import (
    _CentralNNMCTSInferenceServer,
    _RemoteNNMCTSInferenceBackend,
    _SyncRemoteNNMCTSInferenceBackend,
)
from catanrl.utils.catanatron_action_space import get_action_space_size
from catanrl.utils.catanatron_map import build_catan_map

BOARD_WIDTH = 21
BOARD_HEIGHT = 11


@dataclass(frozen=True)
class BenchResult:
    label: str
    workers: int
    decisions: int
    simulations: int
    total_seconds: float
    decision_seconds: list[float]
    trial_totals: tuple[float, ...] = ()

    @property
    def decisions_per_second(self) -> float:
        return self.decisions / self.total_seconds if self.total_seconds > 0 else 0.0

    @property
    def simulations_per_second(self) -> float:
        return self.simulations / self.total_seconds if self.total_seconds > 0 else 0.0

    @property
    def mean_ms(self) -> float:
        return mean(self.decision_seconds) * 1000.0 if self.decision_seconds else 0.0

    @property
    def std_ms(self) -> float:
        return pstdev(self.decision_seconds) * 1000.0 if len(self.decision_seconds) > 1 else 0.0

    @property
    def trial_time_std(self) -> float:
        return pstdev(self.trial_totals) if len(self.trial_totals) > 1 else 0.0


def parse_int_list(value: str) -> list[int]:
    return [int(token.strip()) for token in value.split(",") if token.strip()]


def unique_worker_counts(workers: Sequence[int]) -> list[int]:
    seen: set[int] = set()
    ordered: list[int] = []
    for w in workers:
        if w not in seen:
            seen.add(w)
            ordered.append(w)
    return ordered


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def maybe_sync(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def build_policy_model(
    backbone_type: str,
    model_type: str,
    hidden_dims: Sequence[int],
    num_players: int,
    map_type: Literal["BASE", "MINI", "TOURNAMENT"],
    actor_observation_level: ActorObservationLevel,
    device: torch.device,
) -> PolicyNetworkWrapper:
    dims = compute_single_agent_dims(
        num_players,
        map_type,
        actor_observation_level=actor_observation_level,
    )
    actor_dim = dims["actor_dim"]
    numeric_dim = dims["numeric_dim"]
    board_channels = dims["board_channels"]

    if backbone_type == "mlp":
        backbone_config = BackboneConfig(
            architecture="mlp",
            args=MLPBackboneConfig(input_dim=actor_dim, hidden_dims=list(hidden_dims)),
        )
    elif backbone_type == "xdim":
        backbone_config = BackboneConfig(
            architecture="cross_dimensional",
            args=CrossDimensionalBackboneConfig(
                board_height=BOARD_HEIGHT,
                board_width=BOARD_WIDTH,
                board_channels=board_channels,
                numeric_dim=numeric_dim,
                cnn_channels=[64, 128, 128],
                cnn_kernel_size=(3, 5),
                numeric_hidden_dims=list(hidden_dims),
                fusion_hidden_dim=hidden_dims[-1] if hidden_dims else 256,
                output_dim=hidden_dims[-1] if hidden_dims else 256,
            ),
        )
    else:
        raise ValueError(f"Unknown backbone_type '{backbone_type}'")

    if model_type == "flat":
        model = build_flat_policy_network(
            backbone_config=backbone_config,
            num_actions=get_action_space_size(num_players, map_type),
        )
    elif model_type == "hierarchical":
        model = build_hierarchical_policy_network(
            backbone_config=backbone_config,
            num_players=num_players,
            map_type=map_type,
        )
    else:
        raise ValueError(f"Unknown model_type '{model_type}'")

    return model.to(device)


def build_critic_model(
    backbone_type: str,
    hidden_dims: Sequence[int],
    num_players: int,
    map_type: Literal["BASE", "MINI", "TOURNAMENT"],
    critic_observation_level: CriticObservationLevel,
    device: torch.device,
) -> ValueNetworkWrapper:
    dims = compute_single_agent_dims(
        num_players,
        map_type,
        critic_observation_level=critic_observation_level,
    )
    critic_dim = dims["critic_dim"]
    critic_numeric_dim = dims["critic_numeric_dim"]
    board_channels = dims["board_channels"]

    if backbone_type == "mlp":
        backbone_config = BackboneConfig(
            architecture="mlp",
            args=MLPBackboneConfig(input_dim=critic_dim, hidden_dims=list(hidden_dims)),
        )
    elif backbone_type == "xdim":
        backbone_config = BackboneConfig(
            architecture="cross_dimensional",
            args=CrossDimensionalBackboneConfig(
                board_height=BOARD_HEIGHT,
                board_width=BOARD_WIDTH,
                board_channels=board_channels,
                numeric_dim=critic_numeric_dim,
                cnn_channels=[64, 128, 128],
                cnn_kernel_size=(3, 5),
                numeric_hidden_dims=list(hidden_dims),
                fusion_hidden_dim=hidden_dims[-1] if hidden_dims else 256,
                output_dim=hidden_dims[-1] if hidden_dims else 256,
            ),
        )
    else:
        raise ValueError(f"Unknown backbone_type '{backbone_type}'")

    return build_value_network(backbone_config=backbone_config).to(device)


def make_root_game(
    player: NNMCTSPlayer,
    num_players: int,
    map_type: Literal["BASE", "MINI", "TOURNAMENT"],
    seed: int,
    vps_to_win: int,
    discard_limit: int,
) -> Game:
    opponents = [RandomPlayer(color) for color in COLOR_ORDER[1:num_players]]
    players = [player, *opponents]
    game = Game(
        players=players,
        catan_map=build_catan_map(map_type, seed=seed, number_placement="random"),
        seed=seed,
        vps_to_win=vps_to_win,
        discard_limit=discard_limit,
    )
    return game


def make_player(
    args: argparse.Namespace,
    inference_backend,
    *,
    num_search_workers: int,
) -> NNMCTSPlayer:
    return NNMCTSPlayer(
        color=COLOR_ORDER[0],
        model_type=args.model_type,
        policy_model=None,
        critic_model=None,
        map_type=args.map_type,
        num_simulations=args.num_simulations,
        c_puct=args.c_puct,
        prunning=args.prunning,
        opponent_policy=args.adversarial_policy,
        actor_observation_level=args.actor_observation_level,
        critic_observation_level=args.critic_observation_level,
        num_search_workers=num_search_workers,
        inference_batch_size=args.inference_batch_size,
        inference_wait_ms=args.inference_wait_ms,
        virtual_loss=args.virtual_loss,
        device="cpu",
        inference_backend=inference_backend,
    )


def _args_dict(args: argparse.Namespace) -> dict:
    return {
        "model_type": args.model_type,
        "map_type": args.map_type,
        "num_players": args.num_players,
        "num_simulations": args.num_simulations,
        "c_puct": args.c_puct,
        "prunning": args.prunning,
        "adversarial_policy": args.adversarial_policy,
        "actor_observation_level": args.actor_observation_level,
        "critic_observation_level": args.critic_observation_level,
        "inference_batch_size": args.inference_batch_size,
        "inference_wait_ms": args.inference_wait_ms,
        "virtual_loss": args.virtual_loss,
        "vps_to_win": args.vps_to_win,
        "discard_limit": args.discard_limit,
        "warmup_decisions": args.warmup_decisions,
        "decisions": args.decisions,
        "trials": args.trials,
        "seed": args.seed,
    }


def _bridge_queue_forward(
    source: mp.Queue,
    dest: mp.Queue,
    stop_event: threading.Event,
) -> None:
    while not stop_event.is_set():
        try:
            item = source.get(timeout=0.05)
        except queue.Empty:
            continue
        if item is None:
            dest.put(None)
            return
        dest.put(item)


def _setup_fork_safe_remote_bridge(
    server_request_queue: mp.Queue,
    server_response_queues: list[mp.Queue],
    num_search_workers: int,
) -> tuple[mp.Queue, list[mp.Queue], list[threading.Thread], threading.Event]:
    """Fork children cannot safely use spawn-created queues; bridge via fork queues."""
    fork_ctx = mp.get_context("fork")
    local_request = fork_ctx.Queue(maxsize=max(1024, num_search_workers * 64))
    local_responses = [fork_ctx.Queue(maxsize=512) for _ in range(num_search_workers)]
    stop_event = threading.Event()
    threads = [
        threading.Thread(
            target=_bridge_queue_forward,
            args=(local_request, server_request_queue, stop_event),
            name="nn-mcts-fork-req-bridge",
            daemon=True,
        )
    ]
    for idx, server_response_queue in enumerate(server_response_queues):
        threads.append(
            threading.Thread(
                target=_bridge_queue_forward,
                args=(server_response_queue, local_responses[idx], stop_event),
                name=f"nn-mcts-fork-resp-bridge-{idx}",
                daemon=True,
            )
        )
    for thread in threads:
        thread.start()
    return local_request, local_responses, threads, stop_event


def _within_tree_coordinator_main(
    worker_id: int,
    args_dict: dict,
    request_queue: mp.Queue,
    response_queues: list[mp.Queue],
    result_queue: mp.Queue,
    *,
    num_search_workers: int,
    barrier,
) -> None:
    """One decision stream sharing a single tree; leaf evals go to the central server."""
    backend = None
    bridge_stop: threading.Event | None = None
    try:
        if num_search_workers > 1:
            local_request, local_responses, _, bridge_stop = _setup_fork_safe_remote_bridge(
                request_queue,
                response_queues,
                num_search_workers,
            )
            backend = _SyncRemoteNNMCTSInferenceBackend(
                worker_id=worker_id,
                request_queue=local_request,
                response_queue=local_responses[0],
                response_queues=local_responses,
            )
        else:
            backend = _RemoteNNMCTSInferenceBackend(
                worker_id=worker_id,
                request_queue=request_queue,
                response_queue=response_queues[0],
            )
        player = NNMCTSPlayer(
            color=COLOR_ORDER[0],
            model_type=str(args_dict["model_type"]),
            policy_model=None,
            critic_model=None,
            map_type=args_dict["map_type"],
            num_simulations=int(args_dict["num_simulations"]),
            c_puct=float(args_dict["c_puct"]),
            prunning=bool(args_dict["prunning"]),
            opponent_policy=str(args_dict["adversarial_policy"]),
            actor_observation_level=args_dict["actor_observation_level"],
            critic_observation_level=args_dict["critic_observation_level"],
            num_search_workers=num_search_workers,
            inference_batch_size=int(args_dict["inference_batch_size"]),
            inference_wait_ms=float(args_dict["inference_wait_ms"]),
            virtual_loss=float(args_dict["virtual_loss"]),
            device="cpu",
            inference_backend=backend,
        )
        num_players = int(args_dict["num_players"])
        map_type = args_dict["map_type"]
        vps = int(args_dict["vps_to_win"])
        discard = int(args_dict["discard_limit"])
        warmup = int(args_dict["warmup_decisions"])
        decisions = int(args_dict["decisions"])
        trials = int(args_dict["trials"])
        num_simulations = int(args_dict["num_simulations"])
        seed = int(args_dict["seed"]) + worker_id * 100_003

        for idx in range(warmup):
            game = make_root_game(player, num_players, map_type, seed + idx, vps, discard)
            player.decide(game, game.playable_actions)

        barrier.wait()
        start = time.perf_counter()
        for _trial in range(trials):
            for idx in range(decisions):
                game = make_root_game(
                    player, num_players, map_type, seed + warmup + idx, vps, discard
                )
                player.decide(game, game.playable_actions)
        elapsed = time.perf_counter() - start

        result_queue.put(
            {
                "worker_id": worker_id,
                "sims": trials * decisions * num_simulations,
                "elapsed": elapsed,
            }
        )
    except BaseException:
        result_queue.put({"worker_id": worker_id, "error": traceback.format_exc()})
    finally:
        if bridge_stop is not None:
            bridge_stop.set()
        if backend is not None:
            try:
                backend.close()
            except Exception:
                pass


def run_within_tree_remote(
    policy_model: PolicyNetworkWrapper,
    critic_model: ValueNetworkWrapper,
    args: argparse.Namespace,
    device: torch.device,
) -> list[BenchResult]:
    label = "within-tree"
    worker_counts = unique_worker_counts([1, *parse_int_list(args.parallel_workers)])
    results: list[BenchResult] = []
    args_dict = _args_dict(args)

    for idx, workers in enumerate(worker_counts):
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

        mp_ctx = mp.get_context("spawn")
        request_queue = mp_ctx.Queue(maxsize=max(1024, args.inference_batch_size * 4))
        if workers > 1:
            response_queues = [mp_ctx.Queue(maxsize=512) for _ in range(workers)]
        else:
            response_queues = [mp_ctx.Queue(maxsize=512)]
        result_queue = mp_ctx.Queue()
        barrier = mp_ctx.Barrier(1)
        server = _CentralNNMCTSInferenceServer(
            policy_model=policy_model,
            critic_model=critic_model,
            model_type=args.model_type,
            device=device,
            request_queue=request_queue,
            response_queues=response_queues,
            max_batch_size=args.inference_batch_size,
            max_wait_ms=args.inference_wait_ms,
        )
        server.start()
        proc = mp_ctx.Process(
            target=_within_tree_coordinator_main,
            args=(0, args_dict, request_queue, response_queues, result_queue),
            kwargs={"num_search_workers": workers, "barrier": barrier},
            daemon=False,
        )
        proc.start()
        message = result_queue.get()
        proc.join(timeout=120)
        server.stop()

        if "error" in message:
            raise RuntimeError(f"{label} workers={workers} failed:\n{message['error']}")

        elapsed = float(message["elapsed"])
        total_sims = int(message["sims"])
        total_decisions = args.decisions * args.trials
        result = BenchResult(
            label=label,
            workers=workers,
            decisions=total_decisions,
            simulations=total_sims,
            total_seconds=elapsed,
            decision_seconds=[elapsed / max(1, total_decisions)] * total_decisions,
            trial_totals=(elapsed,),
        )
        results.append(result)
        print(
            f"Finished {label} workers={workers}: "
            f"{result.simulations_per_second:.1f} sims/s"
        )
        if idx == 0:
            print("Baseline captured; speedups below are relative to this row.")

    return results


def print_results(results: list[BenchResult], trials: int) -> None:
    baseline = results[0]
    print()
    hdr_trial = f" {'Trial σ s':>10}" if trials > 1 else ""
    print(
        f"{'Mode':<22} {'Workers':>7} {'Dec/s':>11} {'Sims/s':>12} "
        f"{'Mean ms':>10} {'Std ms':>9} {'Speedup':>9}{hdr_trial}"
    )
    print("-" * (86 + (11 if trials > 1 else 0)))
    for result in results:
        speedup = (
            result.simulations_per_second / baseline.simulations_per_second
            if baseline.simulations_per_second > 0
            else 0.0
        )
        trial_col = f" {result.trial_time_std:>10.4f}" if trials > 1 else ""
        print(
            f"{result.label:<22} "
            f"{result.workers:>7} "
            f"{result.decisions_per_second:>11.3f} "
            f"{result.simulations_per_second:>12.1f} "
            f"{result.mean_ms:>10.2f} "
            f"{result.std_ms:>9.2f} "
            f"{speedup:>8.2f}x"
            f"{trial_col}"
        )

    print()
    print("Speedup is throughput (simulations / second) vs the first row (workers=1).")


class _PhaseTimer:
    def __init__(self) -> None:
        self.totals: dict[str, float] = {}
        self.counts: dict[str, int] = {}

    def add(self, name: str, dt: float) -> None:
        self.totals[name] = self.totals.get(name, 0.0) + dt
        self.counts[name] = self.counts.get(name, 0) + 1


@contextmanager
def instrument_sequential_player(player, timer: _PhaseTimer) -> Iterator[None]:
    import catanrl.players.nn_mcts_player as nnp

    orig_features = nnp.full_game_to_features
    orig_spectrum = nnp.execute_spectrum
    orig_select = nnp.NNMCTSPlayer._select
    backend = player._inference_backend
    orig_eval = backend.evaluate_leaf

    def timed_features(*a, **kw):
        start = time.perf_counter()
        try:
            return orig_features(*a, **kw)
        finally:
            timer.add("feature_extraction (full_game_to_features)", time.perf_counter() - start)

    def timed_spectrum(*a, **kw):
        start = time.perf_counter()
        try:
            return orig_spectrum(*a, **kw)
        finally:
            timer.add("child_expansion (execute_spectrum / game.copy)", time.perf_counter() - start)

    def timed_select(self, node):
        start = time.perf_counter()
        try:
            return orig_select(self, node)
        finally:
            timer.add("selection (_select / PUCT)", time.perf_counter() - start)

    def timed_eval(actor_features, critic_features):
        start = time.perf_counter()
        try:
            return orig_eval(actor_features, critic_features)
        finally:
            timer.add("nn_forward (evaluate_leaf + H2D/D2H)", time.perf_counter() - start)

    nnp.full_game_to_features = timed_features
    nnp.execute_spectrum = timed_spectrum
    nnp.NNMCTSPlayer._select = timed_select
    backend.evaluate_leaf = timed_eval
    try:
        yield
    finally:
        nnp.full_game_to_features = orig_features
        nnp.execute_spectrum = orig_spectrum
        nnp.NNMCTSPlayer._select = orig_select
        backend.evaluate_leaf = orig_eval


def run_breakdown(
    policy_model: PolicyNetworkWrapper,
    critic_model: ValueNetworkWrapper,
    args: argparse.Namespace,
    device: torch.device,
) -> None:
    mp_ctx = mp.get_context("spawn")
    request_queue = mp_ctx.Queue(maxsize=64)
    response_queue = mp_ctx.Queue(maxsize=64)
    server = _CentralNNMCTSInferenceServer(
        policy_model=policy_model,
        critic_model=critic_model,
        model_type=args.model_type,
        device=device,
        request_queue=request_queue,
        response_queues=[response_queue],
        max_batch_size=args.inference_batch_size,
        max_wait_ms=args.inference_wait_ms,
    )
    server.start()
    backend = _RemoteNNMCTSInferenceBackend(0, request_queue, response_queue)
    player = make_player(args, backend, num_search_workers=1)
    try:
        for idx in range(args.warmup_decisions):
            seed = args.seed + idx
            game = make_root_game(
                player, args.num_players, args.map_type, seed, args.vps_to_win, args.discard_limit
            )
            player.decide(game, game.playable_actions)

        timer = _PhaseTimer()
        total_decide = 0.0
        maybe_sync(device)
        with instrument_sequential_player(player, timer):
            for idx in range(args.decisions):
                seed = args.seed + args.warmup_decisions + idx
                game = make_root_game(
                    player,
                    args.num_players,
                    args.map_type,
                    seed,
                    args.vps_to_win,
                    args.discard_limit,
                )
                maybe_sync(device)
                start = time.perf_counter()
                player.decide(game, game.playable_actions)
                maybe_sync(device)
                total_decide += time.perf_counter() - start

        accounted = sum(timer.totals.values())
        other = max(0.0, total_decide - accounted)
        total_sims = args.decisions * args.num_simulations

        print("\n=== Per-decision phase breakdown (sequential, workers=1) ===")
        print(
            f"Decisions: {args.decisions} | sims/decision: {args.num_simulations} "
            f"| total decide wall: {total_decide:.3f}s"
        )
        rows = sorted(timer.totals.items(), key=lambda kv: kv[1], reverse=True)
        rows.append(("other", other))
        for name, total in rows:
            pct = 100.0 * total / total_decide if total_decide > 0 else 0.0
            print(f"{name:<48} {total:>9.3f}s {pct:>8.1f}%")
        if total_sims:
            print(f"Mean time per simulation: {total_decide / total_sims * 1000.0:.3f} ms")

        if args.cprofile:
            profiler = cProfile.Profile()
            profiler.enable()
            for idx in range(max(1, args.decisions // 2)):
                seed = args.seed + 10_000 + idx
                game = make_root_game(
                    player,
                    args.num_players,
                    args.map_type,
                    seed,
                    args.vps_to_win,
                    args.discard_limit,
                )
                player.decide(game, game.playable_actions)
            profiler.disable()
            stream = io.StringIO()
            pstats.Stats(profiler, stream=stream).sort_stats("tottime").print_stats(25)
            print(stream.getvalue())
    finally:
        backend.close()
        server.stop()


def _across_games_worker_main(
    worker_id: int,
    args_dict: dict,
    request_queue: mp.Queue,
    response_queue: mp.Queue,
    result_queue: mp.Queue,
    barrier,
) -> None:
    backend = None
    try:
        backend = _RemoteNNMCTSInferenceBackend(
            worker_id=worker_id,
            request_queue=request_queue,
            response_queue=response_queue,
        )
        player = NNMCTSPlayer(
            color=COLOR_ORDER[0],
            model_type=str(args_dict["model_type"]),
            policy_model=None,
            critic_model=None,
            map_type=args_dict["map_type"],
            num_simulations=int(args_dict["num_simulations"]),
            c_puct=float(args_dict["c_puct"]),
            prunning=bool(args_dict["prunning"]),
            opponent_policy=str(args_dict["adversarial_policy"]),
            actor_observation_level=args_dict["actor_observation_level"],
            critic_observation_level=args_dict["critic_observation_level"],
            num_search_workers=1,
            device="cpu",
            inference_backend=backend,
        )
        num_players = int(args_dict["num_players"])
        map_type = args_dict["map_type"]
        vps = int(args_dict["vps_to_win"])
        discard = int(args_dict["discard_limit"])
        warmup = int(args_dict["warmup_decisions"])
        decisions = int(args_dict["decisions"])
        num_simulations = int(args_dict["num_simulations"])
        seed = int(args_dict["seed"]) + worker_id * 100_003

        for idx in range(warmup):
            game = make_root_game(player, num_players, map_type, seed + idx, vps, discard)
            player.decide(game, game.playable_actions)

        barrier.wait()
        start = time.perf_counter()
        for idx in range(decisions):
            game = make_root_game(player, num_players, map_type, seed + warmup + idx, vps, discard)
            player.decide(game, game.playable_actions)
        elapsed = time.perf_counter() - start

        result_queue.put(
            {"worker_id": worker_id, "sims": decisions * num_simulations, "elapsed": elapsed}
        )
    except BaseException:
        result_queue.put({"worker_id": worker_id, "error": traceback.format_exc()})
    finally:
        if backend is not None:
            try:
                backend.close()
            except Exception:
                pass


def run_across_games(
    policy_model: PolicyNetworkWrapper,
    critic_model: ValueNetworkWrapper,
    args: argparse.Namespace,
    device: torch.device,
) -> None:
    worker_counts = unique_worker_counts(parse_int_list(args.across_game_workers))
    print("\n=== Across-games parallelism (independent games -> central batched server) ===")
    print(
        f"Each game worker runs sequential search; {args.decisions} decisions x "
        f"{args.num_simulations} sims after {args.warmup_decisions} warmup."
    )
    print(f"{'GameWorkers':>11} {'Agg sims/s':>12} {'Per-worker sims/s':>18} {'ms/sim':>9} {'Speedup':>9}")
    print("-" * 64)

    args_dict = _args_dict(args)
    baseline_sps = None
    for workers in worker_counts:
        mp_ctx = mp.get_context("spawn")
        request_queue = mp_ctx.Queue(maxsize=max(1024, args.inference_batch_size * workers * 4))
        response_queues = [mp_ctx.Queue(maxsize=512) for _ in range(workers)]
        result_queue = mp_ctx.Queue()
        barrier = mp_ctx.Barrier(workers)
        server = _CentralNNMCTSInferenceServer(
            policy_model=policy_model,
            critic_model=critic_model,
            model_type=args.model_type,
            device=device,
            request_queue=request_queue,
            response_queues=response_queues,
            max_batch_size=args.inference_batch_size,
            max_wait_ms=args.inference_wait_ms,
        )
        server.start()
        procs = []
        for wid in range(workers):
            proc = mp_ctx.Process(
                target=_across_games_worker_main,
                args=(wid, args_dict, request_queue, response_queues[wid], result_queue, barrier),
                daemon=True,
            )
            proc.start()
            procs.append(proc)

        results = [result_queue.get() for _ in range(workers)]
        for proc in procs:
            proc.join(timeout=30)
        server.stop()

        errors = [r for r in results if "error" in r]
        if errors:
            print(f"{workers:>11}  ERROR:\n{errors[0]['error']}")
            continue

        total_sims = sum(int(r["sims"]) for r in results)
        max_elapsed = max(float(r["elapsed"]) for r in results)
        agg_sps = total_sims / max_elapsed if max_elapsed > 0 else 0.0
        per_worker_sps = agg_sps / workers if workers else 0.0
        if baseline_sps is None:
            baseline_sps = per_worker_sps
        speedup = agg_sps / baseline_sps if baseline_sps else 0.0
        ms_per_sim = (max_elapsed / (total_sims / workers) * 1000.0) if total_sims else 0.0
        print(
            f"{workers:>11} {agg_sps:>12.1f} {per_worker_sps:>18.1f} {ms_per_sim:>9.3f} {speedup:>8.2f}x"
        )


def _run_within_tree(
    policy_model: PolicyNetworkWrapper,
    critic_model: ValueNetworkWrapper,
    args: argparse.Namespace,
    device: torch.device,
) -> None:
    print(
        "\n=== within-tree (shared tree + fork workers + virtual loss, central batched inference) ==="
    )
    results = run_within_tree_remote(policy_model, critic_model, args, device)
    print_results(results, trials=args.trials)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare within-tree (fork) vs across-games NN MCTS throughput.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model-type", choices=["flat", "hierarchical"], default="flat")
    parser.add_argument("--backbone-type", choices=["mlp", "xdim"], default="mlp")
    parser.add_argument("--policy-hidden-dims", type=int, nargs="+", default=[512, 512])
    parser.add_argument("--critic-hidden-dims", type=int, nargs="+", default=[512, 512])
    parser.add_argument("--map-type", choices=["BASE", "MINI", "TOURNAMENT"], default="MINI")
    parser.add_argument("--num-players", type=int, choices=[2, 3, 4], default=2)
    parser.add_argument(
        "--actor-observation-level",
        choices=["private", "public", "full"],
        default="private",
    )
    parser.add_argument(
        "--critic-observation-level",
        choices=["private", "public", "full"],
        default="full",
    )
    parser.add_argument("--policy-weights", type=str, default=None)
    parser.add_argument("--critic-weights", type=str, default=None)
    parser.add_argument("--num-simulations", type=int, default=64)
    parser.add_argument("--decisions", type=int, default=20)
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--warmup-decisions", type=int, default=2)
    parser.add_argument(
        "--parallel-workers",
        type=str,
        default="2,4,8",
        help="Comma-separated within-tree fork worker counts.",
    )
    parser.add_argument("--inference-batch-size", type=int, default=32)
    parser.add_argument("--inference-wait-ms", type=float, default=1.0)
    parser.add_argument("--virtual-loss", type=float, default=1.0)
    parser.add_argument(
        "--mode",
        type=str,
        default="both",
        choices=["within-tree", "across-games", "both"],
        help="Which strategies to benchmark. 'both' runs within-tree and across-games.",
    )
    parser.add_argument(
        "--across-game-workers",
        type=str,
        default="1,2,4,8",
        help="Comma-separated game-process counts for across-games mode.",
    )
    parser.add_argument("--breakdown", action="store_true")
    parser.add_argument("--cprofile", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument(
        "--compile-mode",
        type=str,
        default="reduce-overhead",
        choices=["default", "reduce-overhead", "max-autotune"],
    )
    parser.add_argument("--c-puct", type=float, default=1.5)
    parser.add_argument("--prunning", action="store_true")
    parser.add_argument(
        "--adversarial-policy",
        choices=[
            "self",
            "random",
            "weighted",
            "value",
            "victorypoint",
            "alphabeta",
            "sameturnalphabeta",
            "mcts",
            "playouts",
        ],
        default="self",
    )
    parser.add_argument("--vps-to-win", type=int, default=10)
    parser.add_argument("--discard-limit", type=int, default=9)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    policy_model = build_policy_model(
        backbone_type=args.backbone_type,
        model_type=args.model_type,
        hidden_dims=args.policy_hidden_dims,
        num_players=args.num_players,
        map_type=args.map_type,
        actor_observation_level=args.actor_observation_level,
        device=device,
    )
    critic_model = build_critic_model(
        backbone_type=args.backbone_type,
        hidden_dims=args.critic_hidden_dims,
        num_players=args.num_players,
        map_type=args.map_type,
        critic_observation_level=args.critic_observation_level,
        device=device,
    )

    if args.policy_weights is not None:
        policy_model.load_state_dict(torch.load(args.policy_weights, map_location=device))
    if args.critic_weights is not None:
        critic_model.load_state_dict(torch.load(args.critic_weights, map_location=device))
    policy_model.eval()
    critic_model.eval()

    if args.compile:
        policy_model = torch.compile(policy_model, mode=args.compile_mode)
        critic_model = torch.compile(critic_model, mode=args.compile_mode)
        print(f"torch.compile enabled (mode={args.compile_mode})")

    print("NN MCTS throughput profile")
    print(f"Device: {device}")
    print(f"Map: {args.map_type} | players: {args.num_players}")
    print(
        f"Simulations/decision: {args.num_simulations} | decisions/trial: {args.decisions} "
        f"| trials: {args.trials}"
    )
    print(
        f"Batch size: {args.inference_batch_size} | wait: {args.inference_wait_ms} ms "
        f"| virtual loss: {args.virtual_loss}"
    )

    if args.breakdown:
        run_breakdown(policy_model, critic_model, args, device)
        return

    run_all = args.mode == "both"
    if run_all or args.mode == "within-tree":
        _run_within_tree(policy_model, critic_model, args, device)
    if run_all or args.mode == "across-games":
        run_across_games(policy_model, critic_model, args, device)


if __name__ == "__main__":
    main()
