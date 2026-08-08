"""Across-games self-play orchestration shared by training and evaluation.

A central process owns the policy/critic models and a batched inference server
(`_CentralNNMCTSInferenceServer`); N worker processes each play full self-play
games with `NNMCTSPlayer` seats whose leaf evaluations are routed back to the
server via `_RemoteNNMCTSInferenceBackend`. This is the across-games parallelism
strategy that profiling showed scales to ~1500 sims/s (vs the within-tree
virtual-loss path, which is a net slowdown).

The same orchestration loop powers two callers:
- evaluation (`scripts/eval_mcts_self_play.py`), which only needs game stats;
- training (`generate_self_play_data` below), which also collects per-decision
  policy targets and final-outcome value targets.
"""

from __future__ import annotations

import multiprocessing as mp
import queue
import random
import time
import traceback
from collections import Counter
from dataclasses import dataclass
from typing import Callable, Literal, Sequence, cast

import numpy as np
import torch
from catanatron.game import Game
from tqdm import tqdm

from catanrl.features.catanatron_utils import (
    COLOR_ORDER,
    ActorObservationLevel,
    CriticObservationLevel,
)
from catanrl.players import NNMCTSPlayer
from catanrl.players.nn_mcts_player import (
    _CentralNNMCTSInferenceServer,
    _RemoteNNMCTSInferenceBackend,
)
from catanrl.utils.catanatron_action_space import MapType, get_action_space_size, to_action_space
from catanrl.utils.catanatron_game import force_player_order
from catanrl.utils.catanatron_map import build_catan_map
from catanrl.utils.seeding import derive_map_and_game_seeds, derive_seed


def run_inference_server_workers(
    *,
    policy_model,
    critic_model,
    model_type: str,
    device: str | torch.device,
    num_workers: int,
    inference_batch_size: int,
    inference_wait_ms: float,
    worker_target: Callable,
    worker_args: Sequence[tuple],
    handle_result: Callable[[dict], None],
    total: int,
    show_tqdm: bool = True,
    stall_timeout_s: float = 600.0,
    result_queue_size: int | None = None,
) -> str | None:
    """Spawn ``num_workers`` game-worker processes feeding one central batched
    inference server, collect their results, and return the first error string
    (or ``None`` on success).

    ``worker_target`` is invoked as
    ``worker_target(worker_id, request_queue, response_queue, result_queue, *worker_args[worker_id])``
    and must put a dict containing ``worker_id`` (and either ``error`` or its
    payload) on ``result_queue``. Workers may stream multiple payloads by setting
    ``done=False`` and finish with a final ``done=True`` payload. Messages
    without ``done`` retain the historical one-message-per-worker behavior.
    ``handle_result`` is called for each successful message.
    """
    mp_ctx = mp.get_context("spawn")

    # individual game workers' NN MCTS players send requests into queue.
    request_queue: mp.Queue = mp_ctx.Queue(
        maxsize=max(1024, inference_batch_size * num_workers * 4)
    )
    # central inference backend responds into this queue (ordered in terms of
    # requests received per worker).
    response_queues: list[mp.Queue] = [mp_ctx.Queue(maxsize=512) for _ in range(num_workers)]
    result_queue: mp.Queue = mp_ctx.Queue(
        maxsize=(
            max(16, num_workers * 2)
            if result_queue_size is None
            else max(1, int(result_queue_size))
        )
    )
    inference_server = _CentralNNMCTSInferenceServer(
        policy_model=policy_model,
        critic_model=critic_model,
        model_type=model_type,
        device=str(device),
        request_queue=request_queue,
        response_queues=response_queues,
        max_batch_size=inference_batch_size,
        max_wait_ms=inference_wait_ms,
    )
    processes = [
        mp_ctx.Process(
            target=worker_target,
            args=(
                worker_id,
                request_queue,
                response_queues[worker_id],
                result_queue,
                *worker_args[worker_id],
            ),
        )
        for worker_id in range(num_workers)
    ]

    pending_workers = set(range(num_workers))
    progress = tqdm(total=total, disable=not show_tqdm)
    first_error: str | None = None
    inference_started = time.perf_counter()
    last_status = inference_started
    last_activity = inference_started
    last_evaluations = 0

    def worker_state_summary() -> str:
        return ", ".join(
            f"{worker_id}:pid={process.pid},alive={process.is_alive()},exit={process.exitcode}"
            for worker_id, process in enumerate(processes)
            if worker_id in pending_workers
        )

    inference_server.start()
    try:
        for process in processes:
            process.start()

        while pending_workers:
            try:
                # collect responses from workers
                message = result_queue.get(timeout=0.5)
            except queue.Empty:
                # 500ms has passed without any response; check every component
                # before continuing so a dead inference thread cannot strand all
                # game workers on unresolved futures.
                for worker_id in list(pending_workers):
                    exitcode = processes[worker_id].exitcode
                    if exitcode is not None:
                        first_error = (
                            f"Worker {worker_id} exited with code {exitcode} "
                            "before reporting a result."
                        )
                        pending_workers.remove(worker_id)
                        break
                if first_error is not None:
                    break
                now = time.perf_counter()
                evaluations, batches = inference_server.stats()
                if evaluations > last_evaluations:
                    last_activity = now
                    last_evaluations = evaluations
                if not inference_server.is_alive():
                    detail = inference_server.fatal_error() or "no Python traceback available"
                    first_error = (
                        "Central NN MCTS inference thread exited while workers were pending. "
                        f"Workers: {worker_state_summary()}\n{detail}"
                    )
                    break
                if stall_timeout_s > 0.0 and now - last_activity >= stall_timeout_s:
                    first_error = (
                        "Self-play made no worker-result or neural-inference progress for "
                        f"{now - last_activity:.1f}s (limit={stall_timeout_s:.1f}s; "
                        f"evaluations={evaluations}, batches={batches}). "
                        f"Workers: {worker_state_summary()}"
                    )
                    break
                if show_tqdm and now - last_status >= 30.0:
                    elapsed = max(now - inference_started, 1e-9)
                    average_batch = evaluations / batches if batches else 0.0
                    progress.set_postfix_str(
                        f"{evaluations / elapsed:.0f} eval/s, batch={average_batch:.1f}",
                        refresh=True,
                    )
                    last_status = now
                continue

            # process message received from worker.
            last_activity = time.perf_counter()
            worker_id = int(message["worker_id"])
            if worker_id not in pending_workers:
                continue
            if "error" in message:
                pending_workers.remove(worker_id)
                first_error = str(message["error"])
                break
            handle_result(message)
            progress.update(int(message.get("games", 0)))
            if bool(message.get("done", True)):
                pending_workers.remove(worker_id)
    finally:
        progress.close()
        if first_error is not None:
            for process in processes:
                if process.is_alive():
                    process.terminate()
        for process in processes:
            process.join(timeout=5.0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=1.0)
        inference_server.stop()

    return first_error


# ----------------------------------------------------------------------
# Training data collection
# ----------------------------------------------------------------------


@dataclass
class SelfPlayExperience:
    """One AlphaZero training target collected from self-play."""

    actor_state: np.ndarray
    critic_state: np.ndarray
    policy: np.ndarray
    action_mask: np.ndarray
    value: float
    full_search: bool = True
    value_wdl: np.ndarray | None = None
    aux_value_targets: np.ndarray | None = None


def _put_training_result_chunks(
    *,
    result_queue: mp.Queue,
    worker_id: int,
    experiences: Sequence[tuple],
    stats: dict[str, int],
    chunk_size: int,
) -> None:
    """Bound the size of result-pipe messages while preserving game accounting."""
    if chunk_size < 1:
        raise ValueError("result chunk size must be at least 1")
    chunks = [
        experiences[start : start + chunk_size] for start in range(0, len(experiences), chunk_size)
    ] or [[]]
    for index, chunk in enumerate(chunks):
        result_queue.put(
            {
                "worker_id": worker_id,
                "done": False,
                "games": 1 if index == 0 else 0,
                "experiences": list(chunk),
                "stats": stats if index == 0 else {},
            }
        )


class _TrainingSelfPlayPlayer(NNMCTSPlayer):
    """NNMCTSPlayer that records (actor_state, critic_state, visit-policy) for
    each of its own decisions, using the training temperature/noise schedule."""

    def __init__(
        self,
        *args,
        temperature: float,
        final_temperature: float,
        target_temperature: float | None,
        temperature_drop_move: int,
        noise_turns: int,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._tr_temperature = float(temperature)
        self._tr_final_temperature = float(final_temperature)
        self._tr_target_temperature = target_temperature
        self._tr_temperature_drop_move = int(temperature_drop_move)
        self._tr_noise_turns = int(noise_turns)
        self.samples: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []

    def reset_samples(self) -> None:
        self.samples = []

    def decide(self, game, playable_actions):
        if len(playable_actions) == 1:
            return playable_actions[0]

        move_number = len(game.state.action_records)
        temperature = (
            self._tr_temperature
            if move_number < self._tr_temperature_drop_move
            else self._tr_final_temperature
        )
        add_noise = move_number < self._tr_noise_turns
        policy, action = self.run_search_policy(
            game,
            temperature=max(temperature, 1e-3),
            add_noise=add_noise,
            target_temperature=self._tr_target_temperature,
        )
        # Record the decision state from this seat's (the mover's) perspective.
        actor_state = self._observation_features(
            game,
            self.color,
            cast(ActorObservationLevel, self.actor_observation_level),
        )
        critic_state = self._observation_features(
            game,
            self.color,
            cast(CriticObservationLevel, self.critic_observation_level),
        )
        num_players = len(game.state.colors)
        game_colors = tuple(game.state.colors)
        map_type = cast(MapType, self.map_type)
        action_mask = np.zeros(get_action_space_size(num_players, map_type), dtype=np.bool_)
        for playable_action in playable_actions:
            action_mask[to_action_space(playable_action, num_players, map_type, game_colors)] = True
        self.samples.append((actor_state.copy(), policy.copy(), critic_state.copy(), action_mask))
        return action


def _build_training_seats(args_dict: dict, inference_backend) -> list[_TrainingSelfPlayPlayer]:
    num_players = int(args_dict["num_players"])
    return [
        _TrainingSelfPlayPlayer(
            color=color,
            model_type=str(args_dict["model_type"]),
            policy_model=None,
            critic_model=None,
            map_type=args_dict["map_type"],
            num_simulations=int(args_dict["num_simulations"]),
            c_puct=float(args_dict["c_puct"]),
            prunning=bool(args_dict["prunning"]),
            actor_observation_level=args_dict["actor_observation_level"],
            critic_observation_level=args_dict["critic_observation_level"],
            ismcts_determinizations=int(args_dict["ismcts_determinizations"]),
            dirichlet_alpha=float(args_dict["dirichlet_alpha"]),
            dirichlet_frac=float(args_dict["dirichlet_frac"]),
            inference_backend=inference_backend,
            temperature=float(args_dict["temperature"]),
            final_temperature=float(args_dict["final_temperature"]),
            target_temperature=(
                None
                if args_dict["target_temperature"] is None
                else float(args_dict["target_temperature"])
            ),
            temperature_drop_move=int(args_dict["temperature_drop_move"]),
            noise_turns=int(args_dict["noise_turns"]),
        )
        for color in COLOR_ORDER[:num_players]
    ]


def _training_worker_main(
    worker_id: int,
    request_queue: mp.Queue,
    response_queue: mp.Queue,
    result_queue: mp.Queue,
    episode_seeds: Sequence[int],
    args_dict: dict,
) -> None:
    inference_backend = _RemoteNNMCTSInferenceBackend(
        worker_id=worker_id,
        request_queue=request_queue,
        response_queue=response_queue,
        response_timeout_s=float(args_dict["inference_response_timeout_s"]),
    )
    try:
        players = _build_training_seats(args_dict, inference_backend)
        map_type = args_dict["map_type"]
        vps_to_win = int(args_dict["vps_to_win"])
        discard_limit = int(args_dict["discard_limit"])

        for episode_seed in episode_seeds:
            experiences: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]] = []
            stats: Counter[str] = Counter()
            random.seed(episode_seed)
            np.random.seed(episode_seed % (2**32))
            torch.manual_seed(episode_seed)
            map_seed, game_seed = derive_map_and_game_seeds(episode_seed)
            for player in players:
                player.reset_state()
                player.reset_samples()

            game = Game(
                players=players,
                catan_map=build_catan_map(map_type, seed=map_seed, number_placement="random"),
                seed=game_seed,
                discard_limit=discard_limit,
                vps_to_win=vps_to_win,
            )
            force_player_order(game, players)
            game.play()
            winner = game.winning_color()

            stats["games"] += 1
            if winner is not None:
                stats[f"wins_{winner.value}"] += 1

            for player in players:
                if winner is None:
                    value = 0.0
                else:
                    value = 1.0 if player.color == winner else -1.0
                for actor_state, policy, critic_state, action_mask in player.samples:
                    experiences.append((actor_state, critic_state, policy, action_mask, value))

            _put_training_result_chunks(
                result_queue=result_queue,
                worker_id=worker_id,
                experiences=experiences,
                stats=dict(stats),
                chunk_size=int(args_dict["result_chunk_size"]),
            )
        result_queue.put(
            {
                "worker_id": worker_id,
                "done": True,
                "games": 0,
                "experiences": [],
                "stats": {},
            }
        )
    except BaseException:
        result_queue.put({"worker_id": worker_id, "error": traceback.format_exc()})
    finally:
        inference_backend.close()


def _assign_episode_seeds(num_games: int, num_workers: int, seed: int) -> list[list[int]]:
    assignments: list[list[int]] = [[] for _ in range(max(1, num_workers))]
    for game_idx in range(num_games):
        episode_seed = derive_seed(seed, "self_play_episode", game_idx)
        assignments[game_idx % len(assignments)].append(episode_seed)
    return assignments


def generate_self_play_data(
    *,
    policy_model,
    critic_model,
    model_type: str,
    map_type: Literal["BASE", "MINI", "TOURNAMENT"],
    num_players: int,
    num_games: int,
    num_game_workers: int,
    num_simulations: int,
    c_puct: float,
    prunning: bool,
    actor_observation_level: ActorObservationLevel,
    critic_observation_level: CriticObservationLevel,
    ismcts_determinizations: int,
    inference_batch_size: int,
    inference_wait_ms: float,
    temperature: float,
    final_temperature: float,
    target_temperature: float | None,
    temperature_drop_move: int,
    noise_turns: int,
    dirichlet_alpha: float,
    dirichlet_frac: float,
    vps_to_win: int,
    discard_limit: int,
    seed: int,
    device: str | torch.device,
    show_tqdm: bool = True,
    worker_stall_timeout_s: float = 600.0,
    inference_response_timeout_s: float = 120.0,
    result_chunk_size: int = 64,
) -> tuple[list[SelfPlayExperience], dict[str, int]]:
    """Generate AlphaZero self-play training data across worker processes.

    Returns ``(experiences, stats)`` where experiences carry policy targets
    (visit distributions) and final-outcome value targets per decision.
    """
    if num_games <= 0:
        return [], {}
    if worker_stall_timeout_s <= 0.0:
        raise ValueError("worker_stall_timeout_s must be positive")
    if inference_response_timeout_s <= 0.0:
        raise ValueError("inference_response_timeout_s must be positive")
    if result_chunk_size < 1:
        raise ValueError("result_chunk_size must be at least 1")

    assignments = [a for a in _assign_episode_seeds(num_games, num_game_workers, seed) if a]
    num_workers = len(assignments)

    args_dict = {
        "model_type": model_type,
        "map_type": map_type,
        "num_players": num_players,
        "num_simulations": num_simulations,
        "c_puct": c_puct,
        "prunning": prunning,
        "actor_observation_level": actor_observation_level,
        "critic_observation_level": critic_observation_level,
        "ismcts_determinizations": ismcts_determinizations,
        "dirichlet_alpha": dirichlet_alpha,
        "dirichlet_frac": dirichlet_frac,
        "temperature": temperature,
        "final_temperature": final_temperature,
        "target_temperature": target_temperature,
        "temperature_drop_move": temperature_drop_move,
        "noise_turns": noise_turns,
        "vps_to_win": vps_to_win,
        "discard_limit": discard_limit,
        "inference_response_timeout_s": inference_response_timeout_s,
        "result_chunk_size": result_chunk_size,
    }
    worker_args = [(assignment, args_dict) for assignment in assignments]

    experiences: list[SelfPlayExperience] = []
    stats: Counter[str] = Counter()

    def handle_result(message: dict) -> None:
        for actor_state, critic_state, policy, action_mask, value in message["experiences"]:
            experiences.append(
                SelfPlayExperience(
                    actor_state=actor_state,
                    critic_state=critic_state,
                    policy=policy,
                    action_mask=action_mask,
                    value=float(value),
                )
            )
        for key, count in message["stats"].items():
            stats[key] += int(count)

    first_error = run_inference_server_workers(
        policy_model=policy_model,
        critic_model=critic_model,
        model_type=model_type,
        device=device,
        num_workers=num_workers,
        inference_batch_size=inference_batch_size,
        inference_wait_ms=inference_wait_ms,
        worker_target=_training_worker_main,
        worker_args=worker_args,
        handle_result=handle_result,
        total=num_games,
        show_tqdm=show_tqdm,
        stall_timeout_s=worker_stall_timeout_s,
    )
    if first_error is not None:
        raise RuntimeError(f"Self-play worker failed:\n{first_error}")

    return experiences, dict(stats)
