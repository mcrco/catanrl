"""Direct released-Canopy versus CatanRL search games in authoritative Catanatron."""

from __future__ import annotations

import queue
import random
import threading
import time
from collections.abc import Sequence
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass
from statistics import NormalDist
from typing import Any, Literal, cast

import numpy as np
import torch
from catanatron.game import Game
from catanatron.models.enums import Action
from catanatron.models.player import Color, SimplePlayer
from catanatron.state_functions import get_actual_victory_points
from tqdm import tqdm

from catanrl.algorithms.alphazero.native_search import (
    reconcile_search_after_observed_step,
)
from catanrl.envs.cppanatron import NativeGame, NativeMCTSSearch, full_native_features
from catanrl.eval.canopy_catanatron_bridge import CanopyBridgeProcess
from catanrl.eval.native_mcts_budget import (
    _catanatron_native_mask,
    _replay_native_action,
    _search,
)
from catanrl.eval.reporting import wilson_interval
from catanrl.features.catanatron_utils import full_game_to_features
from catanrl.players.nn_mcts_player import (
    _CentralNNMCTSInferenceServer,
    _SyncRemoteNNMCTSInferenceBackend,
)
from catanrl.utils.catanatron_action_space import (
    canopy_action_count_increment,
    from_action_space,
    to_action_space,
)
from catanrl.utils.catanatron_game import force_player_order
from catanrl.utils.catanatron_map import build_catan_map_from_native_game
from catanrl.utils.seeding import derive_map_and_game_seeds, derive_seed

MapType = Literal["BASE", "MINI", "TOURNAMENT"]


def summarize_canopy_head_to_head(
    records: Sequence[dict[str, Any]],
    noninferiority_margin: float,
) -> dict[str, Any]:
    """Summarize direct candidate wins, including a conservative parity gate."""

    if not records:
        raise ValueError("At least one direct Canopy game is required")
    if not 0.0 <= noninferiority_margin < 0.5:
        raise ValueError("noninferiority_margin must be in [0, 0.5)")
    games = len(records)
    wins = sum(bool(record["win"]) for record in records)
    draws = sum(bool(record["draw"]) for record in records)
    ci_low, ci_high = wilson_interval(wins, games)
    one_sided_low, _ = wilson_interval(wins, games, z=NormalDist().inv_cdf(0.95))
    by_seat = {}
    for seat in ("first", "second"):
        rows = [record for record in records if record["seat"] == seat]
        if not rows:
            raise ValueError(f"Direct Canopy games contain no {seat}-seat records")
        seat_wins = sum(bool(record["win"]) for record in rows)
        seat_low, seat_high = wilson_interval(seat_wins, len(rows))
        by_seat[seat] = {
            "games": len(rows),
            "wins": seat_wins,
            "draws": sum(bool(record["draw"]) for record in rows),
            "win_rate": seat_wins / len(rows),
            "win_rate_ci95": [seat_low, seat_high],
        }
    return {
        "games": games,
        "wins": wins,
        "losses": games - wins - draws,
        "draws": draws,
        "win_rate": wins / games,
        "win_rate_ci95": [ci_low, ci_high],
        "mean_vps": sum(int(record["vps"]) for record in records) / games,
        "noninferiority": {
            "null_win_rate": 0.5 - noninferiority_margin,
            "margin": noninferiority_margin,
            "confidence": 0.95,
            "one_sided_wilson_low": one_sided_low,
            "passes": one_sided_low > 0.5 - noninferiority_margin,
        },
        "superiority": {
            "null_win_rate": 0.5,
            "confidence": 0.95,
            "one_sided_wilson_low": one_sided_low,
            "passes": one_sided_low > 0.5,
        },
        "by_seat": by_seat,
    }


@dataclass
class _CanopyRequest:
    game: Game
    actions: Sequence[Action]
    future: Future[Action]


class BatchingCanopyBackend(threading.Thread):
    """Coalesce synchronous game-thread requests into released-Canopy batches."""

    def __init__(
        self,
        bridge: CanopyBridgeProcess,
        catanatron_lock: threading.Lock,
        *,
        max_batch_size: int,
        max_wait_ms: float,
    ) -> None:
        super().__init__(name="canopy-catanatron-batcher", daemon=True)
        self.bridge = bridge
        self.catanatron_lock = catanatron_lock
        self.max_batch_size = max(1, int(max_batch_size))
        self.max_wait_ms = max(0.0, float(max_wait_ms))
        self.requests: queue.Queue[_CanopyRequest | None] = queue.Queue()
        self._stop_event = threading.Event()
        self._live_game_ids: set[str] = set()
        self._live_game_ids_lock = threading.Lock()

    def register(self, game: Game) -> None:
        with self._live_game_ids_lock:
            self._live_game_ids.add(game.id)

    def unregister(self, game: Game) -> None:
        with self._live_game_ids_lock:
            self._live_game_ids.discard(game.id)

    def decide(self, game: Game, actions: Sequence[Action]) -> Action:
        future: Future[Action] = Future()
        self.requests.put(_CanopyRequest(game=game, actions=tuple(actions), future=future))
        return future.result()

    def run(self) -> None:
        while not self._stop_event.is_set():
            first = self.requests.get()
            if first is None:
                break
            batch = [first]
            deadline = time.perf_counter() + self.max_wait_ms / 1000.0
            while len(batch) < self.max_batch_size:
                remaining = deadline - time.perf_counter()
                if remaining <= 0.0:
                    break
                try:
                    request = self.requests.get(timeout=remaining)
                except queue.Empty:
                    break
                if request is None:
                    self._stop_event.set()
                    break
                batch.append(request)
            try:
                # History reconstruction in the bridge touches Catanatron's
                # module-global RNG, so serialize it with authoritative steps.
                with self._live_game_ids_lock:
                    active_game_ids = tuple(self._live_game_ids)
                with self.catanatron_lock:
                    selected = self.bridge.decide_many(
                        [(request.game, request.actions) for request in batch],
                        active_game_ids=active_game_ids,
                    )
                for request, action in zip(batch, selected):
                    request.future.set_result(action)
            except BaseException as error:
                for request in batch:
                    request.future.set_exception(error)

    def close(self) -> None:
        self._stop_event.set()
        self.requests.put(None)
        self.join(timeout=10.0)


@dataclass
class _LiveHeadToHeadGame:
    game: Game
    native: NativeGame
    candidate: SimplePlayer
    candidate_seat: int
    episode_seed: int
    rng_state: tuple[Any, ...]
    actions: int = 0
    decisions: int = 0


@contextmanager
def _isolated_game_rng(live: _LiveHeadToHeadGame, lock: threading.Lock):
    with lock:
        process_state = random.getstate()
        random.setstate(live.rng_state)
        try:
            yield
        finally:
            live.rng_state = random.getstate()
            random.setstate(process_state)


def _new_game(
    episode_seed: int,
    candidate_seat: int,
    *,
    map_type: MapType,
    vps_to_win: int,
    discard_limit: int,
    catanatron_lock: threading.Lock,
) -> _LiveHeadToHeadGame:
    map_seed, game_seed = derive_map_and_game_seeds(episode_seed)
    native = NativeGame(
        2,
        map_type,
        seed=game_seed,
        map_seed=map_seed,
        number_placement="random",
        discard_limit=discard_limit,
        vps_to_win=vps_to_win,
    )
    candidate = SimplePlayer(Color.RED)
    canopy = SimplePlayer(Color.BLUE)
    players = [candidate, canopy] if candidate_seat == 0 else [canopy, candidate]
    with catanatron_lock:
        process_state = random.getstate()
        try:
            game = Game(
                players=players,
                catan_map=build_catan_map_from_native_game(native, map_type),
                seed=game_seed,
                discard_limit=discard_limit,
                vps_to_win=vps_to_win,
            )
            force_player_order(game, players)
            rng_state = random.getstate()
        finally:
            random.setstate(process_state)
    return _LiveHeadToHeadGame(
        game=game,
        native=native,
        candidate=candidate,
        candidate_seat=candidate_seat,
        episode_seed=episode_seed,
        rng_state=rng_state,
    )


def _assert_shadow_parity(live: _LiveHeadToHeadGame, map_type: MapType) -> np.ndarray:
    native_mask = live.native.valid_action_mask()
    catanatron_mask = _catanatron_native_mask(
        live.game,
        map_type,
        live.native.action_space_size,
    )
    if not np.array_equal(native_mask, catanatron_mask):
        differing = np.flatnonzero(native_mask != catanatron_mask).tolist()
        raise RuntimeError(f"Catanatron/native legal actions diverged: {differing}")
    python_features = full_game_to_features(
        live.game,
        2,
        map_type,
        base_color=live.game.state.current_color(),
    )
    native_features = full_native_features(live.native, map_type, live.native.current_player)
    if not np.allclose(native_features, python_features, rtol=0.0, atol=1e-7):
        differing = np.flatnonzero(
            ~np.isclose(native_features, python_features, rtol=0.0, atol=1e-7)
        )
        raise RuntimeError(
            f"Catanatron/native features diverged; first indices: {differing[:16].tolist()}"
        )
    return native_mask


def play_head_to_head_game(
    *,
    episode_seed: int,
    candidate_seat: int,
    budget: int,
    args_dict: dict[str, Any],
    actor_indices: np.ndarray,
    critic_indices: np.ndarray,
    inference_backend: Any,
    canopy_backend: BatchingCanopyBackend,
    catanatron_lock: threading.Lock,
) -> dict[str, Any]:
    """Play one direct match with Catanatron authoritative at every step."""

    map_type: MapType = args_dict["map_type"]
    live = _new_game(
        episode_seed,
        candidate_seat,
        map_type=map_type,
        vps_to_win=int(args_dict["vps_to_win"]),
        discard_limit=int(args_dict["discard_limit"]),
        catanatron_lock=catanatron_lock,
    )
    search: NativeMCTSSearch | None = None
    max_actions = int(args_dict["max_actions"])
    canopy_backend.register(live.game)
    try:
        while (
            live.game.winning_color() is None
            and live.game.state.num_turns < int(args_dict["turns_limit"])
            and (max_actions <= 0 or live.actions < max_actions)
        ):
            mask = _assert_shadow_parity(live, map_type)
            legal = np.flatnonzero(mask)
            if legal.size == 0:
                raise RuntimeError("Head-to-head game has no legal action")
            if legal.size == 1:
                action_index = int(legal[0])
            elif live.native.current_player == candidate_seat:
                if search is None and bool(args_dict["tree_reuse"]):
                    search = NativeMCTSSearch(
                        live.native,
                        map_type,
                        c_puct=float(args_dict["c_puct"]),
                        seed=derive_seed(
                            episode_seed,
                            "canopy_h2h_search",
                            live.decisions,
                        ),
                        canonical_pruning=bool(args_dict["canonical_pruning"]),
                        search_selection="completed-q",
                        c_visit=float(args_dict["c_visit"]),
                        c_scale=float(args_dict["c_scale"]),
                    )
                result = _search(
                    game=live.native,
                    budget=budget,
                    decision_index=live.decisions,
                    episode_seed=episode_seed,
                    args_dict=args_dict,
                    actor_indices=actor_indices,
                    critic_indices=critic_indices,
                    inference_backend=inference_backend,
                    search=search,
                )
                action_index = result.action
            else:
                action = canopy_backend.decide(live.game, live.game.playable_actions)
                action_index = to_action_space(
                    action,
                    2,
                    map_type,
                    tuple(live.game.state.colors),
                )

            action = from_action_space(
                action_index,
                live.game.state.current_color(),
                2,
                map_type,
                tuple(live.game.state.colors),
                live.game.playable_actions,
            )
            if action not in live.game.playable_actions:
                raise RuntimeError(f"Selected action {action_index} is not legal in Catanatron")
            with _isolated_game_rng(live, catanatron_lock):
                record = live.game.execute(action)
            _replay_native_action(live.native, action_index, action.action_type, record.result)
            search = reconcile_search_after_observed_step(
                game=live.native,
                map_type=map_type,
                action=action_index,
                search=search,
            )
            live.decisions += 1
            live.actions += canopy_action_count_increment(action_index, 2, map_type)

        winner_color = live.game.winning_color()
        winner = None if winner_color is None else live.game.state.color_to_index[winner_color]
        if live.native.winner != winner:
            raise RuntimeError(
                f"Catanatron/native winners diverged: {winner} != {live.native.winner}"
            )
        return {
            "seat": "first" if candidate_seat == 0 else "second",
            "episode_seed": episode_seed,
            "win": winner == candidate_seat,
            "draw": winner is None,
            "winner": winner,
            "vps": int(get_actual_victory_points(live.game.state, live.candidate.color)),
            "total_vps": int(
                sum(
                    get_actual_victory_points(live.game.state, color)
                    for color in live.game.state.colors
                )
            ),
            "turns": int(live.game.state.num_turns),
            "actions": live.actions,
            "decisions": live.decisions,
            "authoritative_engine": "Catanatron",
            "opponent": "cullback/canopy nexus-v3",
        }
    finally:
        canopy_backend.unregister(live.game)
        if search is not None:
            search.close()
        live.native.close()


def run_canopy_head_to_head(
    *,
    policy_model,
    critic_model,
    model_type: str,
    bridge: CanopyBridgeProcess,
    actor_indices: np.ndarray,
    critic_indices: np.ndarray,
    scenarios: Sequence[tuple[int, int]],
    budget: int,
    args_dict: dict[str, Any],
    device: str | torch.device,
    num_workers: int,
    inference_batch_size: int,
    inference_wait_ms: float,
    canopy_batch_size: int,
    canopy_wait_ms: float,
    show_tqdm: bool = True,
) -> list[dict[str, Any]]:
    """Run batched direct head-to-head games on one authoritative engine."""

    if num_workers < 1:
        raise ValueError("num_workers must be positive")
    # Everything here is threaded, so in-process queues avoid serialization
    # overhead and give every game an independent response channel.
    request_queue: queue.Queue = queue.Queue(maxsize=max(1024, num_workers * 256))
    response_queues: list[queue.Queue] = [queue.Queue(maxsize=512) for _ in scenarios]
    inference_server = _CentralNNMCTSInferenceServer(
        policy_model=policy_model,
        critic_model=critic_model,
        model_type=model_type,
        device=device,
        request_queue=cast(Any, request_queue),
        response_queues=cast(Any, response_queues),
        max_batch_size=inference_batch_size,
        max_wait_ms=inference_wait_ms,
    )
    catanatron_lock = threading.Lock()
    canopy_backend = BatchingCanopyBackend(
        bridge,
        catanatron_lock,
        max_batch_size=canopy_batch_size,
        max_wait_ms=canopy_wait_ms,
    )
    records: list[dict[str, Any]] = []
    inference_server.start()
    canopy_backend.start()
    progress = tqdm(total=len(scenarios), disable=not show_tqdm, desc="CatanRL vs Canopy")
    try:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            for index, (seat, episode_seed) in enumerate(scenarios):
                backend = _SyncRemoteNNMCTSInferenceBackend(
                    index,
                    cast(Any, request_queue),
                    cast(Any, response_queues[index]),
                )
                futures.append(
                    executor.submit(
                        play_head_to_head_game,
                        episode_seed=episode_seed,
                        candidate_seat=seat,
                        budget=budget,
                        args_dict=args_dict,
                        actor_indices=actor_indices,
                        critic_indices=critic_indices,
                        inference_backend=backend,
                        canopy_backend=canopy_backend,
                        catanatron_lock=catanatron_lock,
                    )
                )
            for future in as_completed(futures):
                records.append(future.result())
                progress.update(1)
    finally:
        progress.close()
        canopy_backend.close()
        inference_server.stop()
    return records


__all__ = [
    "BatchingCanopyBackend",
    "play_head_to_head_game",
    "run_canopy_head_to_head",
    "summarize_canopy_head_to_head",
]
