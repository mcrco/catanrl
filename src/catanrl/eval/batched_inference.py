"""Batched frozen-network inference shared by search workers.

Requests carry one or more full observations. The server coalesces requests
from all workers into GPU batches, replaying CUDA graphs when it can.
``RemoteEvaluator`` and ``LocalEvaluator`` also implement ``evaluate_leaf`` so
native MCTS (``run_native_search_policy``) can use them as inference backends.
"""

from __future__ import annotations

import multiprocessing as mp
import queue
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch

from catanrl.models.inference_utils import forward_policy_value
from catanrl.players.nn_mcts_player import _LeafEvaluation, _NNMCTSInferenceBackend

ValueTransform = Literal["zero-sum", "win-probability"]


def value_transform_for_reward(reward_function: str | None) -> ValueTransform:
    """How a critic trained with ``reward_function`` maps to a zero-sum value.

    The ``win`` reward pays +1/-1, so its critic is already zero-sum for the
    player to move. The ``shaped`` reward pays +1 for a win and 0 for a loss
    (plus small VP/production shaping), so its critic estimates a win
    probability ``p`` whose zero-sum equivalent is ``2p - 1``. Searches that
    negate values across players need the zero-sum form.
    """
    return "zero-sum" if reward_function == "win" else "win-probability"


def to_zero_sum(values: np.ndarray, transform: ValueTransform) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if transform == "win-probability":
        return np.clip(2.0 * values - 1.0, -1.0, 1.0)
    return np.clip(values, -1.0, 1.0)


@dataclass
class _Request:
    worker_id: int
    request_id: int
    observations: np.ndarray


def _forward(
    policy_model,
    critic_model,
    model_type: str,
    device: str,
    actor_indices: np.ndarray | None,
    critic_indices: np.ndarray | None,
    observations: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    with torch.inference_mode():
        full = torch.from_numpy(observations).to(device, non_blocking=True)
        actor = full if actor_indices is None else full[:, actor_indices]
        critic = full if critic_indices is None else full[:, critic_indices]
        logits, values = forward_policy_value(
            policy_model, critic_model, actor, model_type, critic_states=critic
        )
        return (
            logits.float().cpu().numpy(),
            values.float().clamp(-1.0, 1.0).cpu().numpy(),
        )


class _GraphedForward:
    """Replays CUDA graphs captured at power-of-two batch sizes.

    Small-batch inference of the graph backbone is dominated by kernel-launch
    overhead (about 6 ms per forward on an RTX 2070 Super whatever the batch up
    to ~32 rows); replaying captured graphs removes most of it. Batches are
    padded up to the next captured size and larger ones are chunked.
    """

    def __init__(self, forward, observation_size: int, device: str, max_bucket: int) -> None:
        self.forward = forward
        self.buckets: list[int] = []
        self.graphs: dict[int, tuple[torch.cuda.CUDAGraph, torch.Tensor, tuple[torch.Tensor, torch.Tensor]]] = {}
        size = 1
        while size <= max_bucket:
            self.buckets.append(size)
            size *= 2
        with torch.inference_mode():
            for bucket in self.buckets:
                static = torch.zeros(bucket, observation_size, device=device)
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(2):
                        self.forward(static)
                torch.cuda.current_stream().wait_stream(stream)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    outputs = self.forward(static)
                self.graphs[bucket] = (graph, static, outputs)

    def __call__(self, observations: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        logits_parts, value_parts = [], []
        largest = self.buckets[-1]
        with torch.inference_mode():
            for start in range(0, len(observations), largest):
                chunk = observations[start : start + largest]
                bucket = next(size for size in self.buckets if size >= len(chunk))
                graph, static, (logits, values) = self.graphs[bucket]
                static[: len(chunk)].copy_(torch.from_numpy(chunk), non_blocking=False)
                graph.replay()
                logits_parts.append(logits[: len(chunk)].float().cpu().numpy())
                value_parts.append(values[: len(chunk)].float().clamp(-1.0, 1.0).cpu().numpy())
        return np.concatenate(logits_parts, axis=0), np.concatenate(value_parts, axis=0)


def _index_tensor(indices: np.ndarray | None, size: int, device: str) -> torch.Tensor | None:
    if indices is None:
        return None
    indices = np.asarray(indices, dtype=np.int64)
    if indices.size == size and bool((indices == np.arange(size)).all()):
        return None
    return torch.from_numpy(indices).to(device)


class LocalEvaluator:
    """In-process evaluator for tests and single-process probes.

    Values are returned in zero-sum form for the player to move.
    """

    def __init__(
        self,
        policy_model,
        critic_model,
        model_type: str,
        device: str,
        observation_size: int,
        actor_indices: np.ndarray | None = None,
        critic_indices: np.ndarray | None = None,
        value_transform: ValueTransform = "zero-sum",
        cuda_graphs: bool = False,
        max_graph_rows: int = 512,
    ) -> None:
        self.value_transform = value_transform
        self.policy_model = policy_model
        self.critic_model = critic_model
        self.model_type = model_type
        self.device = str(device)
        self.actor_indices = _index_tensor(actor_indices, observation_size, self.device)
        self.critic_indices = _index_tensor(critic_indices, observation_size, self.device)
        self._graphed: _GraphedForward | None = None
        if cuda_graphs and self.device.startswith("cuda"):

            def forward(full: torch.Tensor):
                actor = full if self.actor_indices is None else full[:, self.actor_indices]
                critic = full if self.critic_indices is None else full[:, self.critic_indices]
                return forward_policy_value(
                    policy_model, critic_model, actor, model_type, critic_states=critic
                )

            self._graphed = _GraphedForward(forward, observation_size, self.device, max_graph_rows)

    def evaluate(self, observations: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self._graphed is not None:
            logits, values = self._graphed(observations)
            return logits, to_zero_sum(values, self.value_transform)
        logits, values = _forward(
            self.policy_model,
            self.critic_model,
            self.model_type,
            self.device,
            self.actor_indices,
            self.critic_indices,
            observations,
        )
        return logits, to_zero_sum(values, self.value_transform)

    def evaluate_leaf(
        self, actor_features: np.ndarray, critic_features: np.ndarray
    ) -> _LeafEvaluation:
        """Native MCTS hook for full actor and critic observations."""
        if self.actor_indices is not None or self.critic_indices is not None:
            raise ValueError("evaluate_leaf needs full actor and critic observations")
        logits, values = self.evaluate(np.asarray(actor_features, dtype=np.float32)[None])
        return _LeafEvaluation(policy_logits=logits[0], value=float(values[0]))


class BatchInferenceServer(threading.Thread):
    def __init__(
        self,
        *,
        policy_model,
        critic_model,
        model_type: str,
        device: str,
        observation_size: int,
        actor_indices: np.ndarray | None,
        critic_indices: np.ndarray | None,
        request_queue: mp.Queue,
        response_queues: Sequence[mp.Queue],
        max_rows: int = 2048,
        max_wait_ms: float = 1.0,
        cuda_graphs: bool = True,
        max_graph_rows: int = 512,
    ) -> None:
        super().__init__(name="batch-inference", daemon=True)
        self.observation_size = observation_size
        self.cuda_graphs = bool(cuda_graphs) and str(device).startswith("cuda")
        self.max_graph_rows = int(max_graph_rows)
        self._graphed: _GraphedForward | None = None
        self.ready = threading.Event()
        self.startup_error: str | None = None
        self.policy_model = policy_model
        self.critic_model = critic_model
        self.model_type = model_type
        self.device = str(device)
        self.actor_indices = _index_tensor(actor_indices, observation_size, self.device)
        self.critic_indices = _index_tensor(critic_indices, observation_size, self.device)
        self.request_queue = request_queue
        self.response_queues = response_queues
        self.max_rows = max(1, int(max_rows))
        self.max_wait = max(0.0, float(max_wait_ms)) / 1000.0
        self._stop_event = threading.Event()
        self.rows_evaluated = 0
        self.batches = 0

    def _eager(self, observations: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return _forward(
            self.policy_model,
            self.critic_model,
            self.model_type,
            self.device,
            self.actor_indices,
            self.critic_indices,
            observations,
        )

    def _capture(self) -> None:
        def forward(full: torch.Tensor):
            actor = full if self.actor_indices is None else full[:, self.actor_indices]
            critic = full if self.critic_indices is None else full[:, self.critic_indices]
            return forward_policy_value(
                self.policy_model, self.critic_model, actor, self.model_type, critic_states=critic
            )

        try:
            self._graphed = _GraphedForward(
                forward, self.observation_size, self.device, self.max_graph_rows
            )
        except Exception as exc:  # pragma: no cover - depends on the model
            self._graphed = None
            self.startup_error = f"CUDA graph capture failed, using eager inference: {exc!r}"

    def run(self) -> None:
        if self.cuda_graphs:
            self._capture()
        self.ready.set()
        while not self._stop_event.is_set():
            try:
                first = self.request_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if first is None:
                break
            batch = [first]
            rows = len(first.observations)
            deadline = time.perf_counter() + self.max_wait
            while rows < self.max_rows:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    break
                try:
                    item = self.request_queue.get(timeout=remaining)
                except queue.Empty:
                    break
                if item is None:
                    self.request_queue.put(None)
                    break
                batch.append(item)
                rows += len(item.observations)
            self._evaluate(batch)

    def _evaluate(self, batch: list[_Request]) -> None:
        try:
            observations = np.concatenate([request.observations for request in batch], axis=0)
            if self._graphed is not None:
                logits, values = self._graphed(observations)
            else:
                logits_parts, value_parts = [], []
                for start in range(0, len(observations), self.max_rows):
                    logits, values = self._eager(observations[start : start + self.max_rows])
                    logits_parts.append(logits)
                    value_parts.append(values)
                logits = np.concatenate(logits_parts, axis=0)
                values = np.concatenate(value_parts, axis=0)
            self.rows_evaluated += len(observations)
            self.batches += 1
            offset = 0
            for request in batch:
                count = len(request.observations)
                self.response_queues[request.worker_id].put(
                    (request.request_id, logits[offset : offset + count], values[offset : offset + count])
                )
                offset += count
        except Exception as exc:  # pragma: no cover - surfaced to the worker
            for request in batch:
                self.response_queues[request.worker_id].put((request.request_id, repr(exc), None))

    def stop(self) -> None:
        self._stop_event.set()
        self.request_queue.put(None)
        self.join(timeout=5.0)


class RemoteEvaluator(_NNMCTSInferenceBackend):
    """Blocking client for ``BatchInferenceServer`` (one request in flight).

    Values are returned in zero-sum form for the player to move, including
    through ``evaluate_leaf``, so native MCTS backs up consistent values.
    """

    def __init__(
        self,
        worker_id: int,
        request_queue: mp.Queue,
        response_queue: mp.Queue,
        observation_size: int,
        actor_indices: np.ndarray | None = None,
        critic_indices: np.ndarray | None = None,
        value_transform: ValueTransform = "zero-sum",
    ) -> None:
        self.value_transform = value_transform
        self.worker_id = worker_id
        self.request_queue = request_queue
        self.response_queue = response_queue
        self.observation_size = observation_size
        self._next_request_id = 0
        self.rows = 0
        self.requests = 0
        self._leaf_buffer = np.empty((1, observation_size), dtype=np.float32)
        actor = None if actor_indices is None else np.asarray(actor_indices)
        critic = None if critic_indices is None else np.asarray(critic_indices)
        self._actor_identity = actor is None or (
            actor.size == observation_size and bool((actor == np.arange(observation_size)).all())
        )
        self._critic_identity = critic is None or (
            critic.size == observation_size and bool((critic == np.arange(observation_size)).all())
        )

    def evaluate(self, observations: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        request_id = self._next_request_id
        self._next_request_id += 1
        self.request_queue.put(
            _Request(self.worker_id, request_id, np.ascontiguousarray(observations, dtype=np.float32))
        )
        self.rows += len(observations)
        self.requests += 1
        while True:
            response_id, logits, values = self.response_queue.get()
            if response_id != request_id:
                continue
            if values is None:
                raise RuntimeError(f"batched inference failed: {logits}")
            return logits, to_zero_sum(values, self.value_transform)

    def evaluate_leaf(
        self, actor_features: np.ndarray, critic_features: np.ndarray
    ) -> _LeafEvaluation:
        """Native MCTS hook; the server slices full observations itself."""
        if not (self._actor_identity and self._critic_identity):
            raise ValueError("evaluate_leaf needs full actor and critic observations")
        self._leaf_buffer[0] = actor_features
        logits, values = self.evaluate(self._leaf_buffer)
        return _LeafEvaluation(policy_logits=logits[0], value=float(values[0]))


__all__ = [
    "BatchInferenceServer",
    "LocalEvaluator",
    "RemoteEvaluator",
    "ValueTransform",
    "to_zero_sum",
    "value_transform_for_reward",
]
