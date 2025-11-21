"""
Parallel AlphaZero trainer that distributes self-play across worker processes
while batching neural network inference on the main process / GPU.
"""

from __future__ import annotations

import queue
import random
import threading
import time
from collections import Counter
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.multiprocessing as mp
from tqdm import tqdm

from catanatron.game import Game
from catanatron.gym.envs.catanatron_env import ACTION_SPACE_SIZE, to_action_space
from catanatron.models.enums import Action
from catanatron.models.map import build_map
from catanatron.models.player import Color

from ...data.data_utils import game_to_features, get_numeric_feature_names
from .mcts import NeuralMCTS
from .trainer import (
    AlphaZeroConfig,
    AlphaZeroExperience,
    AlphaZeroSelfPlayPlayer,
    AlphaZeroTrainer,
)


WorkerResult = Dict[str, object]


class ParallelAlphaZeroTrainer(AlphaZeroTrainer):
    """Extends AlphaZeroTrainer with multiprocessing self-play workers."""

    def __init__(
        self,
        config: Optional[AlphaZeroConfig] = None,
        model: Optional[torch.nn.Module] = None,
        num_workers: int = 2,
        inference_batch_size: int = 64,
        inference_max_wait_ms: float = 2.0,
    ) -> None:
        if num_workers < 2:
            raise ValueError("ParallelAlphaZeroTrainer requires num_workers >= 2.")
        super().__init__(config=config, model=model)

        self.num_workers = num_workers
        self._mp_ctx = mp.get_context("spawn")
        self._inference_queue: mp.Queue = self._mp_ctx.Queue(maxsize=4096)
        self._result_queue: mp.Queue = self._mp_ctx.Queue(maxsize=self.num_workers * 2)
        self._control_queues: List[mp.Queue] = []
        self._response_queues: List[mp.Queue] = []
        self._workers: List[mp.Process] = []
        self._closed = False

        self._start_workers()
        self._inference_server = _InferenceServer(
            trainer=self,
            inference_queue=self._inference_queue,
            response_queues=self._response_queues,
            max_batch_size=inference_batch_size,
            max_wait_ms=inference_max_wait_ms,
        )
        self._inference_server.start()

    def self_play(self, num_games: int) -> Dict[str, float]:
        if num_games <= 0:
            return {}

        assignments = self._assign_games(num_games)
        active_workers = [
            worker_id for worker_id, seeds in enumerate(assignments) if seeds
        ]
        if not active_workers:
            return {}

        progress = tqdm(total=num_games, desc="Self-play", leave=False)
        for worker_id in active_workers:
            self._control_queues[worker_id].put({"type": "run", "seeds": assignments[worker_id]})

        stats: Counter[str] = Counter()
        pending = set(active_workers)
        try:
            while pending:
                result: WorkerResult = self._result_queue.get()
                worker_id = result.get("worker_id")
                if worker_id not in pending:
                    continue
                pending.remove(worker_id)
                worker_stats = Counter(result.get("stats", {}))
                stats.update(worker_stats)
                progress.update(int(worker_stats.get("games", 0)))
                experiences: List[AlphaZeroExperience] = result.get("experiences", [])
                for exp in experiences:
                    self.replay_buffer.append(exp)
        finally:
            progress.close()

        return dict(stats)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True

        for queue_ in self._control_queues:
            queue_.put({"type": "stop"})
        for proc in self._workers:
            proc.join(timeout=5.0)

        if self._inference_server is not None:
            self._inference_server.stop()
        self._inference_queue.put(None)

    def __del__(self) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _start_workers(self) -> None:
        for worker_id in range(self.num_workers):
            control_queue: mp.Queue = self._mp_ctx.Queue()
            response_queue: mp.Queue = self._mp_ctx.Queue(maxsize=512)
            process = self._mp_ctx.Process(
                target=_self_play_worker_main,
                args=(
                    worker_id,
                    self.config,
                    control_queue,
                    self._inference_queue,
                    response_queue,
                    self._result_queue,
                ),
            )
            process.daemon = True
            process.start()

            self._control_queues.append(control_queue)
            self._response_queues.append(response_queue)
            self._workers.append(process)

    def _assign_games(self, num_games: int) -> List[List[Optional[int]]]:
        assignments: List[List[Optional[int]]] = [[] for _ in range(self.num_workers)]
        if num_games <= 0:
            return assignments

        base_seed = self.config.seed
        counts = [num_games // self.num_workers] * self.num_workers
        for idx in range(num_games % self.num_workers):
            counts[idx] += 1

        next_seed_offset = 0
        for worker_id, count in enumerate(counts):
            seeds: List[Optional[int]] = []
            for _ in range(count):
                seed_value = None if base_seed is None else base_seed + next_seed_offset
                seeds.append(seed_value)
                next_seed_offset += 1
            assignments[worker_id] = seeds
        return assignments


# ----------------------------------------------------------------------
# Inference server
# ----------------------------------------------------------------------


class _InferenceServer(threading.Thread):
    def __init__(
        self,
        trainer: AlphaZeroTrainer,
        inference_queue: mp.Queue,
        response_queues: Sequence[mp.Queue],
        max_batch_size: int,
        max_wait_ms: float,
    ) -> None:
        super().__init__(daemon=True)
        self.trainer = trainer
        self.inference_queue = inference_queue
        self.response_queues = response_queues
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms
        self._stop_event = threading.Event()

    def run(self) -> None:
        while not self._stop_event.is_set():
            try:
                first = self.inference_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if first is None:
                break

            batch = [first]
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
                    # Propagate sentinel and exit.
                    self.inference_queue.put(None)
                    break
                batch.append(item)

            features = np.stack([entry["features"] for entry in batch], axis=0)
            states = torch.from_numpy(features).float().to(self.trainer.device)
            self.trainer.model.eval()
            with torch.no_grad():
                log_probs, values = self.trainer._model_forward(states)
                probs = torch.exp(log_probs).cpu().numpy()
                values_np = values.view(-1).cpu().numpy()

            for entry, policy, value in zip(batch, probs, values_np):
                worker_id: int = entry["worker_id"]
                response = (policy.astype(np.float32), float(value))
                self.response_queues[worker_id].put(response)

    def stop(self) -> None:
        self._stop_event.set()
        self.inference_queue.put(None)
        self.join(timeout=5.0)


# ----------------------------------------------------------------------
# Worker process
# ----------------------------------------------------------------------


class _RemoteInferenceClient:
    def __init__(self, worker_id: int, inference_queue: mp.Queue, response_queue: mp.Queue):
        self.worker_id = worker_id
        self.inference_queue = inference_queue
        self.response_queue = response_queue

    def evaluate(self, features: np.ndarray) -> tuple[np.ndarray, float]:
        payload = {"worker_id": self.worker_id, "features": features}
        self.inference_queue.put(payload)
        policy, value = self.response_queue.get()
        return policy, value


class _SelfPlayWorkerController:
    def __init__(
        self,
        worker_id: int,
        config: AlphaZeroConfig,
        inference_queue: mp.Queue,
        response_queue: mp.Queue,
    ) -> None:
        self.worker_id = worker_id
        self.config = config
        self.numeric_features = list(get_numeric_feature_names(config.num_players, config.map_type))
        self.colors = AlphaZeroTrainer.COLOR_ORDER[: config.num_players]
        self._inference_client = _RemoteInferenceClient(worker_id, inference_queue, response_queue)
        self._current_game_samples: List[tuple[Color, np.ndarray, np.ndarray]] = []
        self.mcts = NeuralMCTS(self)
        self._set_seed(self.config.seed)

    def self_play(self, seeds: Sequence[Optional[int]]) -> tuple[Counter, List[AlphaZeroExperience]]:
        stats: Counter[str] = Counter()
        experiences: List[AlphaZeroExperience] = []

        for seed in seeds:
            result = self._play_game(seed=seed)
            winner = result.get("winner")
            stats["games"] += 1
            if winner is not None:
                stats[f"wins_{winner.value}"] += 1
            experiences.extend(result.get("experiences", []))

        return stats, experiences

    # -- Methods used by NeuralMCTS --

    def select_action(self, game: Game, collect_data: bool = True) -> Action:
        color = game.state.current_color()
        state_vec = self._extract_features(game, color)
        move_number = len(game.state.actions)
        temperature = (
            self.config.temperature
            if move_number < self.config.temperature_drop_move
            else self.config.final_temperature
        )
        add_noise = move_number < self.config.noise_turns
        policy, action = self.mcts.search(
            game=game,
            root_features=state_vec,
            temperature=max(temperature, 1e-3),
            add_noise=add_noise,
        )
        if collect_data:
            self._record_sample(color, state_vec, policy)
        return action

    def _policy_value(
        self,
        game: Game,
        color: Color,
        cached_features: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, float]:
        features = cached_features if cached_features is not None else self._extract_features(game, color)
        policy, value_scalar = self._inference_client.evaluate(features)

        valid_actions = game.state.playable_actions
        if not valid_actions:
            return np.zeros(ACTION_SPACE_SIZE, dtype=np.float32), value_scalar

        mask = np.zeros(ACTION_SPACE_SIZE, dtype=np.float32)
        indices = [to_action_space(action) for action in valid_actions]
        mask[indices] = 1.0
        masked = policy * mask
        total = masked.sum()
        if total <= 0:
            masked[indices] = 1.0 / len(indices)
        else:
            masked /= total
        return masked.astype(np.float32), value_scalar

    # -- Local helpers mirroring AlphaZeroTrainer --

    def _play_game(self, seed: Optional[int]) -> Dict[str, Optional[Color]]:
        players = [AlphaZeroSelfPlayPlayer(color, self) for color in self.colors]
        catan_map = build_map(self.config.map_type)
        game = Game(players=players, seed=seed, catan_map=catan_map, vps_to_win=self.config.vps_to_win)
        self._current_game_samples.clear()
        winner = None
        try:
            winner = game.play()
            experiences = self._finalize_game_samples(winner)
            return {"winner": winner, "experiences": experiences}
        finally:
            self._current_game_samples.clear()

    def _extract_features(self, game: Game, color: Color) -> np.ndarray:
        return game_to_features(game, color, self.numeric_features)

    def _record_sample(self, color: Color, state: np.ndarray, policy: np.ndarray) -> None:
        self._current_game_samples.append((color, state.copy(), policy.copy()))

    def _finalize_game_samples(self, winner: Optional[Color]) -> List[AlphaZeroExperience]:
        if not self._current_game_samples:
            return []
        experiences: List[AlphaZeroExperience] = []
        for color, state, policy in self._current_game_samples:
            if winner is None:
                value = 0.0
            else:
                value = 1.0 if color == winner else -1.0
            experiences.append(AlphaZeroExperience(state=state, policy=policy, value=value))
        self._current_game_samples.clear()
        return experiences

    def _set_seed(self, seed: Optional[int]) -> None:
        if seed is None:
            return
        random.seed(seed)
        np.random.seed(seed)


def _self_play_worker_main(
    worker_id: int,
    config: AlphaZeroConfig,
    control_queue: mp.Queue,
    inference_queue: mp.Queue,
    response_queue: mp.Queue,
    result_queue: mp.Queue,
) -> None:
    controller = _SelfPlayWorkerController(
        worker_id=worker_id,
        config=config,
        inference_queue=inference_queue,
        response_queue=response_queue,
    )

    while True:
        command = control_queue.get()
        if command is None:
            break
        cmd_type = command.get("type")
        if cmd_type == "stop":
            break
        if cmd_type != "run":
            continue

        seeds: Sequence[Optional[int]] = command.get("seeds", [])
        stats, experiences = controller.self_play(seeds)
        message: WorkerResult = {
            "worker_id": worker_id,
            "stats": dict(stats),
            "experiences": experiences,
        }
        result_queue.put(message)

