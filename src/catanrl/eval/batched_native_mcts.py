"""Lockstep native MCTS (or raw-policy) games batched across many boards.

The native ``MCTSSearch`` keeps one pending leaf per search, so a single game
needs one network evaluation per simulation. Driving each game from its own
worker process makes every simulation a separate inference round trip. Here
one process holds many games: each tick advances every board to its next
network request (a search root, a search leaf, or a raw-policy decision),
evaluates all requests in one GPU batch, and hands the results back to the C++
searches.

Searches, seeds and action choice match ``run_native_search_policy`` with
``action_temperature=0`` (argmax root visits over legal actions) and the
per-decision seeds used by ``catanrl.eval.search_matchups``, so records pair
with that harness's games.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from cppanatron.catanatron import observation_layout

import cppanatron as cpp
from catanrl.envs.cppanatron import FlatGame
from catanrl.eval.search_matchups import AgentStats
from catanrl.utils.seeding import derive_map_and_game_seeds, derive_seed

Evaluate = Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]


@dataclass
class _Board:
    game: FlatGame
    a_seat: int
    episode_seed: int
    buffer: np.ndarray
    decision: int = 0
    search: cpp.MCTSSearch | None = None
    simulations: int = 0
    pending: str | None = None
    stats: AgentStats = field(default_factory=AgentStats)
    search_started: float = 0.0


def run_batched_native_games(
    *,
    evaluate: Evaluate,
    scenarios: Sequence[tuple[int, int]],
    simulations: int,
    c_puct: float,
    settings: dict[str, Any],
    concurrency: int = 256,
    on_record: Callable[[dict[str, Any], AgentStats], None] | None = None,
) -> tuple[list[dict[str, Any]], AgentStats, dict[str, float]]:
    """Play agent A (MCTS with ``simulations``, or raw policy when 0) vs native F.

    ``scenarios`` holds ``(a_seat, episode_seed)`` pairs. Returns the game
    records, agent A's accumulated stats, and throughput counters.
    """
    layout = observation_layout()
    observation_size = cpp.full_observation_size(2, layout)
    turns_limit = int(settings["turns_limit"])
    queue = list(scenarios)
    boards: list[_Board] = []
    records: list[dict[str, Any]] = []
    total = AgentStats()
    rows = 0
    batches = 0
    started = time.perf_counter()

    def new_board(a_seat: int, episode_seed: int) -> _Board:
        map_seed, game_seed = derive_map_and_game_seeds(episode_seed)
        game = FlatGame(
            2,
            settings["map_type"],
            seed=game_seed,
            map_seed=map_seed,
            number_placement=settings["number_placement"],
            discard_limit=int(settings["discard_limit"]),
            vps_to_win=int(settings["vps_to_win"]),
        )
        return _Board(game, a_seat, episode_seed, np.empty(observation_size, dtype=np.float32))

    def finish_search(board: _Board) -> None:
        game = board.game
        legal = np.flatnonzero(game.action_mask())
        visits = board.search.root_visits()
        action = int(legal[int(np.argmax(visits[legal]))])
        board.stats.seconds += time.perf_counter() - board.search_started
        board.search = None
        game.step(action)
        board.decision += 1

    def advance(board: _Board) -> bool:
        """Advance until a network request is pending; False once the game ends."""
        game = board.game
        while True:
            if game.winner is not None or game.num_turns >= turns_limit:
                return False
            if board.search is not None:
                if board.simulations >= simulations:
                    finish_search(board)
                    continue
                board.simulations += 1
                if board.search.select_leaf(board.buffer) is None:
                    continue  # terminal leaf, backed up inside the search
                board.pending = "leaf"
                return True
            legal = np.flatnonzero(game.action_mask())
            if legal.size == 1:
                game.step(int(legal[0]))
                continue
            if game.current_player != board.a_seat:
                game.step(game.expert_action())
                board.decision += 1
                continue
            board.stats.decisions += 1
            board.stats.searched += 1
            board.search_started = time.perf_counter()
            if simulations <= 0:
                cpp.write_full_observation(game.game, game.current_player, layout, board.buffer)
                board.pending = "raw"
                return True
            board.search = game.search(
                c_puct=c_puct,
                seed=derive_seed(board.episode_seed, "matchup_mcts", board.decision),
            )
            board.simulations = 0
            board.search.root_observation(board.buffer)
            board.pending = "root"
            return True

    def record(board: _Board) -> None:
        game = board.game
        winner = game.winner
        result = {
            "episode_seed": board.episode_seed,
            "a_seat": board.a_seat,
            "seat": "first" if board.a_seat == 0 else "second",
            "winner": winner,
            "win": winner == board.a_seat,
            "draw": winner is None,
            "turns": game.num_turns,
            "a_vps": game.player(board.a_seat).actual_victory_points,
            "b_vps": game.player(1 - board.a_seat).actual_victory_points,
        }
        records.append(result)
        total.add(board.stats)
        if on_record is not None:
            on_record(result, board.stats)

    while queue or boards:
        while queue and len(boards) < concurrency:
            boards.append(new_board(*queue.pop(0)))
        active: list[_Board] = []
        for board in boards:
            if advance(board):
                active.append(board)
            else:
                record(board)
        boards = active
        if not boards:
            continue
        observations = np.stack([board.buffer for board in boards])
        logits, values = evaluate(observations)
        rows += len(boards)
        batches += 1
        for board, row_logits, value in zip(boards, logits, values):
            board.stats.evaluations += 1
            kind, board.pending = board.pending, None
            if kind == "root":
                board.search.initialize_root(np.ascontiguousarray(row_logits, dtype=np.float32))
            elif kind == "leaf":
                board.search.evaluate_leaf(
                    np.ascontiguousarray(row_logits, dtype=np.float32), float(value)
                )
            else:
                game = board.game
                legal = np.flatnonzero(game.action_mask())
                game.step(int(legal[int(np.argmax(row_logits[legal]))]))
                board.stats.seconds += time.perf_counter() - board.search_started
                board.decision += 1
    elapsed = time.perf_counter() - started
    return (
        records,
        total,
        {
            "wall_seconds": elapsed,
            "server_rows_per_second": rows / elapsed if elapsed else 0.0,
            "server_mean_batch_rows": rows / max(1, batches),
        },
    )


def _parallel_worker(
    worker_id: int,
    experiment_name: str,
    which: str,
    value_transform: str,
    scenarios: Sequence[tuple[int, int]],
    simulations: int,
    c_puct: float,
    settings: dict[str, Any],
    concurrency: int,
    result_queue,
) -> None:
    import traceback

    try:
        import torch

        from catanrl.eval.batched_inference import LocalEvaluator
        from catanrl.experiment_store import load_experiment

        torch.set_num_threads(1)
        experiment = load_experiment(experiment_name)
        policy = experiment.build_policy(which=which, device="cuda").eval()
        critic = experiment.build_critic(which=which, device="cuda").eval()
        evaluator = LocalEvaluator(
            policy,
            critic,
            experiment.model_type or "flat",
            "cuda",
            cpp.full_observation_size(2, observation_layout()),
            value_transform=value_transform,  # type: ignore[arg-type]
            cuda_graphs=True,
            max_graph_rows=max(1, concurrency),
        )

        def on_record(record: dict[str, Any], stats: AgentStats) -> None:
            result_queue.put({"worker_id": worker_id, "record": record, "stats": vars(stats)})

        _, _, throughput = run_batched_native_games(
            evaluate=evaluator.evaluate,
            scenarios=scenarios,
            simulations=simulations,
            c_puct=c_puct,
            settings=settings,
            concurrency=concurrency,
            on_record=on_record,
        )
        result_queue.put({"worker_id": worker_id, "done": True, "throughput": throughput})
    except BaseException:
        result_queue.put({"worker_id": worker_id, "error": traceback.format_exc()})


def run_batched_native_parallel(
    *,
    experiment_name: str,
    which: str,
    value_transform: str,
    scenarios: Sequence[tuple[int, int]],
    simulations: int,
    c_puct: float,
    settings: dict[str, Any],
    processes: int = 8,
    concurrency: int = 64,
    progress: bool = True,
    label: str = "",
) -> tuple[list[dict[str, Any]], AgentStats, dict[str, float]]:
    """``run_batched_native_games`` split over processes with their own model copy.

    Native expansion copies a game per child, so one process is CPU-bound at a
    few thousand simulations per second; several processes keep the GPU busy.
    """
    import multiprocessing as mp
    import queue as queue_module

    from tqdm import tqdm

    processes = max(1, min(processes, len(scenarios)))
    shares = [list(scenarios[index::processes]) for index in range(processes)]
    context = mp.get_context("spawn")
    result_queue = context.Queue()
    workers = [
        context.Process(
            target=_parallel_worker,
            args=(
                worker_id,
                experiment_name,
                which,
                value_transform,
                shares[worker_id],
                simulations,
                c_puct,
                settings,
                concurrency,
                result_queue,
            ),
            daemon=True,
        )
        for worker_id in range(processes)
    ]
    records: list[dict[str, Any]] = []
    total = AgentStats()
    evals_per_second = 0.0
    started = time.perf_counter()
    for worker in workers:
        worker.start()
    pending = set(range(processes))
    error = None
    bar = tqdm(total=len(scenarios), disable=not progress, desc=label or f"mcts{simulations}")
    try:
        while pending:
            try:
                message = result_queue.get(timeout=1.0)
            except queue_module.Empty:
                for worker_id in list(pending):
                    if workers[worker_id].exitcode is not None:
                        error = f"native worker {worker_id} exited with {workers[worker_id].exitcode}"
                        pending.discard(worker_id)
                if error:
                    break
                continue
            if "error" in message:
                error = message["error"]
                break
            if message.get("done"):
                pending.discard(int(message["worker_id"]))
                evals_per_second += message["throughput"]["server_rows_per_second"]
                continue
            records.append(message["record"])
            total.add(AgentStats(**message["stats"]))
            bar.update(1)
    finally:
        bar.close()
        for worker in workers:
            if error is not None and worker.is_alive():
                worker.terminate()
            worker.join(timeout=10.0)
    if error is not None:
        raise RuntimeError(f"batched native worker failed:\n{error}")
    elapsed = time.perf_counter() - started
    return (
        records,
        total,
        {
            "wall_seconds": elapsed,
            "server_rows_per_second": total.evaluations / elapsed if elapsed else 0.0,
            "worker_evals_per_second_sum": evals_per_second,
        },
    )


__all__ = ["run_batched_native_games", "run_batched_native_parallel"]
