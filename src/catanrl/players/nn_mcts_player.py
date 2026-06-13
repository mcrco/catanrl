from __future__ import annotations

import math
import multiprocessing as mp
import queue
import random
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import Callable, Literal, Sequence, Tuple

import numpy as np
import torch
from catanatron.features import create_sample
from catanatron.game import Game
from catanatron.models.decks import starting_devcard_bank
from catanatron.models.enums import DEVELOPMENT_CARDS
from catanatron.models.player import Color, Player, RandomPlayer
from catanatron.players.mcts import MCTSPlayer
from catanatron.players.minimax import AlphaBetaPlayer, SameTurnAlphaBetaPlayer
from catanatron.players.playouts import GreedyPlayoutsPlayer
from catanatron.players.search import VictoryPointPlayer
from catanatron.players.tree_search_utils import execute_spectrum, list_prunned_actions
from catanatron.players.value import ValueFunctionPlayer
from catanatron.players.weighted_random import WeightedRandomPlayer

from catanrl.features.catanatron_utils import (
    ActorObservationLevel,
    CriticObservationLevel,
    full_game_to_features,
    get_full_numeric_feature_names,
    get_observation_indices_from_full,
)
from catanrl.models import PolicyNetworkWrapper, ValueNetworkWrapper
from catanrl.utils.catanatron_action_space import to_action_space

EPSILON = 1e-8
CRITIC_MODE_ALIASES = {
    "full": "full",
    "guess": "guessed_dev_cards",
    "guessed": "guessed_dev_cards",
    "guessed_dev_cards": "guessed_dev_cards",
}


@dataclass
class _Node:
    game: Game
    parent: "_Node | None"
    prior_action: object | None = None
    visits: int = 0
    value_sum: float = 0.0
    virtual_visits: int = 0
    virtual_value_sum: float = 0.0
    expanding: bool = False
    lock: threading.RLock = field(default_factory=threading.RLock, init=False, repr=False)
    condition: threading.Condition = field(init=False, repr=False)

    def __post_init__(self):
        self.children: dict[object, list[tuple["_Node", float]]] = defaultdict(list)
        self.action_priors: dict[object, float] = {}
        self.condition = threading.Condition(self.lock)

    @property
    def expanded(self) -> bool:
        return bool(self.children)

    @property
    def value(self) -> float:
        return self.value_sum / self.visits if self.visits > 0 else 0.0

    @property
    def search_visits(self) -> int:
        return self.visits + self.virtual_visits

    @property
    def search_value(self) -> float:
        visits = self.search_visits
        if visits <= 0:
            return 0.0
        return (self.value_sum + self.virtual_value_sum) / visits


@dataclass
class _LeafEvaluation:
    policy_logits: np.ndarray
    value: float


@dataclass
class _LeafEvaluationRequest:
    actor_features: np.ndarray
    critic_features: np.ndarray
    future: Future[_LeafEvaluation]


@dataclass
class _RemoteLeafEvaluationRequest:
    request_id: int
    worker_id: int
    actor_features: np.ndarray
    critic_features: np.ndarray


class _NNMCTSInferenceBackend:
    def evaluate_leaf(
        self,
        actor_features: np.ndarray,
        critic_features: np.ndarray,
    ) -> _LeafEvaluation:
        raise NotImplementedError

    def close(self) -> None:
        return


class _LocalNNMCTSInferenceBackend(_NNMCTSInferenceBackend):
    def __init__(
        self,
        policy_model: PolicyNetworkWrapper,
        critic_model: ValueNetworkWrapper,
        model_type: str,
        device: str,
    ) -> None:
        self.policy_model = policy_model
        self.critic_model = critic_model
        self.model_type = model_type
        self.device = device

    def evaluate_leaf(
        self,
        actor_features: np.ndarray,
        critic_features: np.ndarray,
    ) -> _LeafEvaluation:
        # Single forward for policy + critic, then one fused GPU->CPU transfer.
        # Avoids the two separate syncs (.cpu() and .item()) that dominate the
        # per-leaf latency at batch size 1.
        with torch.inference_mode():
            actor_tensor = torch.from_numpy(actor_features.reshape(1, -1)).to(self.device)
            critic_tensor = torch.from_numpy(critic_features.reshape(1, -1)).to(self.device)
            if self.model_type == "flat":
                logits = self.policy_model(actor_tensor)
            elif self.model_type == "hierarchical":
                action_type_logits, param_logits = self.policy_model(actor_tensor)
                logits = self.policy_model.get_flat_action_logits(action_type_logits, param_logits)
            else:
                raise ValueError(f"Unknown model_type '{self.model_type}'")
            # Clone before the critic runs: under torch.compile(mode="reduce-overhead")
            # the policy output lives in a CUDA-graph static buffer that the critic's
            # graph replay would overwrite. Clone is a no-op-cheap copy when uncompiled.
            policy_vector = logits.reshape(-1).clone()
            value_tensor = self.critic_model(critic_tensor).reshape(-1)
            combined = torch.cat([policy_vector, value_tensor]).to("cpu")

        combined_np = combined.numpy()
        policy_logits = np.array(combined_np[:-1], dtype=np.float32)
        value = float(np.clip(float(combined_np[-1]), -1.0, 1.0))
        return _LeafEvaluation(policy_logits=policy_logits, value=value)


class _BatchedNNMCTSInferenceBackend(_NNMCTSInferenceBackend):
    def __init__(
        self,
        policy_model: PolicyNetworkWrapper,
        critic_model: ValueNetworkWrapper,
        model_type: str,
        device: str,
        max_batch_size: int,
        max_wait_ms: float,
    ) -> None:
        self.policy_model = policy_model
        self.critic_model = critic_model
        self.model_type = model_type
        self.device = device
        self.max_batch_size = max(1, int(max_batch_size))
        self.max_wait_ms = max(0.0, float(max_wait_ms))
        self._queue: queue.Queue[_LeafEvaluationRequest | None] = queue.Queue()
        self._thread = threading.Thread(target=self._run, name="nn-mcts-inference", daemon=True)
        self._closed = False
        self._thread.start()

    def evaluate_leaf(
        self,
        actor_features: np.ndarray,
        critic_features: np.ndarray,
    ) -> _LeafEvaluation:
        if self._closed:
            raise RuntimeError("NN MCTS inference backend is closed.")
        future: Future[_LeafEvaluation] = Future()
        self._queue.put(
            _LeafEvaluationRequest(
                actor_features=actor_features.astype(np.float32, copy=False),
                critic_features=critic_features.astype(np.float32, copy=False),
                future=future,
            )
        )
        return future.result()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._queue.put(None)
        self._thread.join(timeout=5.0)

    def _run(self) -> None:
        while True:
            first = self._queue.get()
            if first is None:
                break

            batch = [first]
            deadline = time.perf_counter() + (self.max_wait_ms / 1000.0)
            while len(batch) < self.max_batch_size:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    break
                try:
                    item = self._queue.get(timeout=remaining)
                except queue.Empty:
                    break
                if item is None:
                    self._queue.put(None)
                    break
                batch.append(item)

            self._evaluate_batch(batch)

    def _evaluate_batch(self, batch: list[_LeafEvaluationRequest]) -> None:
        try:
            actor_features = np.stack([request.actor_features for request in batch], axis=0)
            critic_features = np.stack([request.critic_features for request in batch], axis=0)
            with torch.no_grad():
                actor_tensor = torch.from_numpy(actor_features).to(self.device)
                if self.model_type == "flat":
                    logits = self.policy_model(actor_tensor)
                elif self.model_type == "hierarchical":
                    action_type_logits, param_logits = self.policy_model(actor_tensor)
                    logits = self.policy_model.get_flat_action_logits(
                        action_type_logits, param_logits
                    )
                else:
                    raise ValueError(f"Unknown model_type '{self.model_type}'")
                policy_logits = logits.detach().cpu().numpy()

                critic_tensor = torch.from_numpy(critic_features).to(self.device)
                values = self.critic_model(critic_tensor).view(-1).detach().cpu().numpy()

            for request, logits_np, value in zip(batch, policy_logits, values):
                request.future.set_result(
                    _LeafEvaluation(
                        policy_logits=logits_np.astype(np.float32, copy=False),
                        value=float(np.clip(float(value), -1.0, 1.0)),
                    )
                )
        except Exception as exc:
            for request in batch:
                request.future.set_exception(exc)


class _RemoteNNMCTSInferenceBackend(_NNMCTSInferenceBackend):
    def __init__(
        self,
        worker_id: int,
        request_queue: mp.Queue,
        response_queue: mp.Queue,
    ) -> None:
        self.worker_id = worker_id
        self.request_queue = request_queue
        self.response_queue = response_queue
        self._closed = False
        self._next_request_id = 0
        self._request_lock = threading.Lock()
        self._pending: dict[int, Future[_LeafEvaluation]] = {}
        self._listener = threading.Thread(
            target=self._listen,
            name=f"nn-mcts-remote-inference-{worker_id}",
            daemon=True,
        )
        self._listener.start()

    def evaluate_leaf(
        self,
        actor_features: np.ndarray,
        critic_features: np.ndarray,
    ) -> _LeafEvaluation:
        if self._closed:
            raise RuntimeError("Remote NN MCTS inference backend is closed.")

        with self._request_lock:
            request_id = self._next_request_id
            self._next_request_id += 1
            future: Future[_LeafEvaluation] = Future()
            self._pending[request_id] = future

        self.request_queue.put(
            _RemoteLeafEvaluationRequest(
                request_id=request_id,
                worker_id=self.worker_id,
                actor_features=actor_features.astype(np.float32, copy=False),
                critic_features=critic_features.astype(np.float32, copy=False),
            )
        )
        return future.result()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.response_queue.put(None)
        self._listener.join(timeout=5.0)

    def _listen(self) -> None:
        while True:
            message = self.response_queue.get()
            if message is None:
                break

            request_id = int(message["request_id"])
            with self._request_lock:
                future = self._pending.pop(request_id, None)
            if future is None:
                continue

            error = message.get("error")
            if error is not None:
                future.set_exception(RuntimeError(str(error)))
            else:
                future.set_result(
                    _LeafEvaluation(
                        policy_logits=message["policy_logits"],
                        value=float(message["value"]),
                    )
                )


class _CentralNNMCTSInferenceServer(threading.Thread):
    def __init__(
        self,
        policy_model: PolicyNetworkWrapper,
        critic_model: ValueNetworkWrapper,
        model_type: str,
        device: str | torch.device,
        request_queue: mp.Queue,
        response_queues: Sequence[mp.Queue],
        max_batch_size: int,
        max_wait_ms: float,
    ) -> None:
        super().__init__(name="nn-mcts-central-inference", daemon=True)
        self.policy_model = policy_model
        self.critic_model = critic_model
        self.model_type = model_type
        self.device = str(device)
        self.request_queue = request_queue
        self.response_queues = response_queues
        self.max_batch_size = max(1, int(max_batch_size))
        self.max_wait_ms = max(0.0, float(max_wait_ms))
        self._stop_event = threading.Event()

    def run(self) -> None:
        while not self._stop_event.is_set():
            try:
                first = self.request_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if first is None:
                break

            batch = [first]
            deadline = time.perf_counter() + (self.max_wait_ms / 1000.0)
            while len(batch) < self.max_batch_size:
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

            self._evaluate_batch(batch)

    def stop(self) -> None:
        self._stop_event.set()
        self.request_queue.put(None)
        self.join(timeout=5.0)

    def _evaluate_batch(self, batch: list[_RemoteLeafEvaluationRequest]) -> None:
        try:
            actor_features = np.stack([request.actor_features for request in batch], axis=0)
            critic_features = np.stack([request.critic_features for request in batch], axis=0)
            with torch.no_grad():
                actor_tensor = torch.from_numpy(actor_features).to(self.device)
                if self.model_type == "flat":
                    logits = self.policy_model(actor_tensor)
                elif self.model_type == "hierarchical":
                    action_type_logits, param_logits = self.policy_model(actor_tensor)
                    logits = self.policy_model.get_flat_action_logits(
                        action_type_logits, param_logits
                    )
                else:
                    raise ValueError(f"Unknown model_type '{self.model_type}'")
                policy_logits = logits.detach().cpu().numpy()

                critic_tensor = torch.from_numpy(critic_features).to(self.device)
                values = self.critic_model(critic_tensor).view(-1).detach().cpu().numpy()

            for request, logits_np, value in zip(batch, policy_logits, values):
                self.response_queues[request.worker_id].put(
                    {
                        "request_id": request.request_id,
                        "policy_logits": logits_np.astype(np.float32, copy=False),
                        "value": float(np.clip(float(value), -1.0, 1.0)),
                    }
                )
        except Exception as exc:
            for request in batch:
                self.response_queues[request.worker_id].put(
                    {"request_id": request.request_id, "error": repr(exc)}
                )


class NNMCTSPlayer(Player):
    """
    MCTS player guided by policy priors and critic value estimates.
    """

    def __init__(
        self,
        color: Color,
        model_type: str,
        policy_model: PolicyNetworkWrapper | None,
        critic_model: ValueNetworkWrapper | None,
        map_type: Literal["BASE", "MINI", "TOURNAMENT"] = "BASE",
        num_simulations: int = 64,
        c_puct: float = 1.5,
        prunning: bool = False,
        opponent_policy: str = "self",
        opponent_factory: Callable[[Color], Player] | None = None,
        critic_mode: Literal["full", "guessed_dev_cards"] = "full",
        actor_observation_level: ActorObservationLevel = "private",
        critic_observation_level: CriticObservationLevel = "full",
        num_search_workers: int = 1,
        inference_batch_size: int = 32,
        inference_wait_ms: float = 1.0,
        virtual_loss: float = 1.0,
        device: str | torch.device | None = None,
        inference_backend: _NNMCTSInferenceBackend | None = None,
        **kwargs,
    ):
        super().__init__(color, is_bot=True, **kwargs)

        self.device = str(device) if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_type = model_type
        self.map_type = map_type
        self.num_simulations = int(num_simulations)
        self.c_puct = float(c_puct)
        self.prunning = bool(prunning)
        self.opponent_policy = opponent_policy.lower()
        self.opponent_factory = opponent_factory
        self.critic_mode = self._normalize_critic_mode(critic_mode)
        self.actor_observation_level = actor_observation_level
        self.critic_observation_level = critic_observation_level
        self.num_search_workers = max(1, int(num_search_workers))
        self.inference_batch_size = max(1, int(inference_batch_size))
        self.inference_wait_ms = max(0.0, float(inference_wait_ms))
        self.virtual_loss = float(virtual_loss)
        self._opponents_by_color: dict[Color, Player] = {}
        self._critic_index_cache: dict[int, dict[str, int]] = {}
        self._observation_index_cache: dict[tuple[int, str], np.ndarray] = {}
        self._starting_dev_counter = Counter(starting_devcard_bank())

        if self.opponent_factory is None and self.opponent_policy != "self":
            self.opponent_factory = self._build_opponent_factory(self.opponent_policy)
        self.policy_model = None
        self.critic_model = None
        self._owns_inference_backend = inference_backend is None
        if inference_backend is None:
            if policy_model is None or critic_model is None:
                raise ValueError(
                    "policy_model and critic_model are required when inference_backend is not supplied."
                )
            self.policy_model = policy_model.to(self.device)
            self.critic_model = critic_model.to(self.device)
            self.policy_model.eval()
            self.critic_model.eval()
            self._inference_backend = _LocalNNMCTSInferenceBackend(
                self.policy_model,
                self.critic_model,
                self.model_type,
                self.device,
            )
        else:
            self._inference_backend = inference_backend

    def decide(self, game, playable_actions):
        if len(playable_actions) == 1:
            return playable_actions[0]

        root = _Node(game=game.copy(), parent=None)
        self._expand(root, self._inference_backend)

        if not root.children:
            return playable_actions[0]

        if self.num_search_workers <= 1:
            for _ in range(self.num_simulations):
                self._run_simulation(root, self._inference_backend)
        else:
            self._run_parallel_search(root)

        best_action = max(
            root.children.keys(),
            key=lambda action: sum(child.visits for child, _ in root.children[action]),
        )
        return self._resolve_root_action(
            best_action, playable_actions, len(game.state.colors), tuple(game.state.colors)
        )

    def _run_simulation(
        self,
        root: _Node,
        inference_backend: _NNMCTSInferenceBackend,
    ) -> None:
        node = root
        path = [node]

        while node.expanded and node.game.winning_color() is None:
            node = self._select(node)
            path.append(node)

        if node.game.winning_color() is not None:
            winner = node.game.winning_color()
            leaf_value = 1.0 if winner == self.color else -1.0
        else:
            leaf_value = self._expand(node, inference_backend).value

        for path_node in reversed(path):
            path_node.visits += 1
            path_node.value_sum += leaf_value

    def _run_parallel_search(self, root: _Node) -> None:
        if self._owns_inference_backend:
            if self.policy_model is None or self.critic_model is None:
                raise RuntimeError("Local models are not configured for batched inference.")
            inference_backend: _NNMCTSInferenceBackend = _BatchedNNMCTSInferenceBackend(
                self.policy_model,
                self.critic_model,
                self.model_type,
                self.device,
                self.inference_batch_size,
                self.inference_wait_ms,
            )
            close_inference_backend = True
        else:
            inference_backend = self._inference_backend
            close_inference_backend = False
        completed = 0
        completed_lock = threading.Lock()
        first_error: list[BaseException] = []

        def claim_simulation() -> bool:
            nonlocal completed
            with completed_lock:
                if completed >= self.num_simulations or first_error:
                    return False
                completed += 1
                return True

        def worker() -> None:
            try:
                while claim_simulation():
                    self._run_parallel_simulation(root, inference_backend)
            except BaseException as exc:
                with completed_lock:
                    first_error.append(exc)

        threads = [
            threading.Thread(target=worker, name=f"nn-mcts-search-{idx}", daemon=True)
            for idx in range(self.num_search_workers)
        ]
        try:
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
        finally:
            if close_inference_backend:
                inference_backend.close()

        if first_error:
            raise first_error[0]

    def _run_parallel_simulation(
        self,
        root: _Node,
        inference_backend: _NNMCTSInferenceBackend,
    ) -> None:
        node = root
        path = [node]

        while True:
            if node.game.winning_color() is not None:
                winner = node.game.winning_color()
                leaf_value = 1.0 if winner == self.color else -1.0
                self._backpropagate_parallel(path, leaf_value)
                return

            with node.condition:
                while node.expanding:
                    node.condition.wait()
                if node.expanded:
                    child = self._select_locked(node)
                    with child.lock:
                        child.virtual_visits += 1
                        child.virtual_value_sum -= self.virtual_loss
                    node = child
                    path.append(child)
                    continue
                node.expanding = True

            try:
                leaf_value = self._expand(node, inference_backend).value
            finally:
                with node.condition:
                    node.expanding = False
                    node.condition.notify_all()

            self._backpropagate_parallel(path, leaf_value)
            return

    def _backpropagate_parallel(self, path: list[_Node], leaf_value: float) -> None:
        for path_node in reversed(path):
            with path_node.lock:
                if path_node is not path[0] and path_node.virtual_visits > 0:
                    path_node.virtual_visits -= 1
                    path_node.virtual_value_sum += self.virtual_loss
                path_node.visits += 1
                path_node.value_sum += leaf_value

    def _expand(
        self,
        node: _Node,
        inference_backend: _NNMCTSInferenceBackend,
    ) -> _LeafEvaluation:
        if node.game.winning_color() is not None or node.expanded:
            return _LeafEvaluation(policy_logits=np.array([], dtype=np.float32), value=0.0)

        actions = (
            list_prunned_actions(node.game)
            if self.prunning
            else list(node.game.playable_actions)
        )
        if not actions:
            return _LeafEvaluation(policy_logits=np.array([], dtype=np.float32), value=0.0)

        current_color = node.game.state.current_color()
        evaluation = self._evaluate_leaf(node.game, inference_backend)
        if current_color != self.color and self.opponent_factory is not None:
            opponent_action = self._opponent_action(node.game, actions)
            node.action_priors = {opponent_action: 1.0}
            actions = [opponent_action]
        else:
            node.action_priors = self._policy_priors_from_logits(
                node.game,
                actions,
                evaluation.policy_logits,
            )

        for action in actions:
            outcomes = execute_spectrum(node.game, action)
            for next_game, probability in outcomes:
                node.children[action].append((_Node(game=next_game, parent=node), probability))
        return evaluation

    def _opponent_action(self, game, actions):
        if self.opponent_factory is None:
            raise RuntimeError("Opponent factory is not configured.")
        color = game.state.current_color()
        opponent = self._opponents_by_color.get(color)
        if opponent is None:
            opponent = self.opponent_factory(color)
            self._opponents_by_color[color] = opponent
        return opponent.decide(game, actions)

    @staticmethod
    def _build_opponent_factory(policy: str) -> Callable[[Color], Player]:
        name = policy.lower()
        if name in {"random", "r"}:
            return lambda color: RandomPlayer(color)
        if name in {"weighted", "w"}:
            return lambda color: WeightedRandomPlayer(color)
        if name in {"value", "valuefunction", "f"}:
            return lambda color: ValueFunctionPlayer(color)
        if name in {"victorypoint", "vp"}:
            return lambda color: VictoryPointPlayer(color)
        if name in {"alphabeta", "ab"}:
            return lambda color: AlphaBetaPlayer(color, depth=2, prunning=True)
        if name in {"sameturnalphabeta", "sab"}:
            return lambda color: SameTurnAlphaBetaPlayer(color)
        if name in {"mcts", "m"}:
            return lambda color: MCTSPlayer(color, 100)
        if name in {"playouts", "greedyplayouts", "g"}:
            return lambda color: GreedyPlayoutsPlayer(color, 25)
        raise ValueError(
            f"Unknown opponent_policy '{policy}'. "
            "Use 'self' or one of: random, weighted, value, victorypoint, alphabeta, "
            "sameturnalphabeta, mcts, playouts."
        )

    @staticmethod
    def _normalize_critic_mode(critic_mode: str) -> str:
        normalized = CRITIC_MODE_ALIASES.get(critic_mode.lower())
        if normalized is None:
            valid_modes = ", ".join(sorted(CRITIC_MODE_ALIASES))
            raise ValueError(f"Unknown critic_mode '{critic_mode}'. Use one of: {valid_modes}.")
        return normalized

    def _select(self, node: _Node) -> _Node:
        with node.lock:
            return self._select_locked(node)

    def _select_locked(self, node: _Node) -> _Node:
        total_visits = sum(
            child.search_visits for kids in node.children.values() for child, _ in kids
        )
        maximizing = node.game.state.current_color() == self.color
        best_score = -float("inf")
        best_action = None

        for action, children in node.children.items():
            action_visits = sum(child.search_visits for child, _ in children)
            if action_visits > 0:
                q_value = sum(
                    probability * child.search_value for child, probability in children
                ) / sum(probability for _, probability in children)
            else:
                q_value = 0.0

            if not maximizing:
                q_value = -q_value

            prior = node.action_priors.get(action, 0.0)
            u_value = self.c_puct * prior * math.sqrt(total_visits + 1.0) / (1.0 + action_visits)
            score = q_value + u_value
            if score > best_score:
                best_score = score
                best_action = action

        assert best_action is not None
        children = node.children[best_action]
        outcomes = [child for child, _ in children]
        probs = [probability for _, probability in children]
        return random.choices(outcomes, weights=probs, k=1)[0]

    def _policy_priors_from_logits(
        self,
        game,
        actions,
        logits: np.ndarray,
    ) -> dict[object, float]:
        num_players = len(game.state.colors)
        game_colors = tuple(game.state.colors)
        indices = [to_action_space(action, num_players, self.map_type, game_colors) for action in actions]
        action_logits = np.array([logits[idx] for idx in indices], dtype=np.float32)
        action_logits = action_logits - float(np.max(action_logits))
        exp_logits = np.exp(action_logits)
        probs = exp_logits / max(float(exp_logits.sum()), EPSILON)
        return {action: float(prob) for action, prob in zip(actions, probs)}

    def _evaluate_leaf(
        self,
        game,
        inference_backend: _NNMCTSInferenceBackend,
    ) -> _LeafEvaluation:
        current_color = game.state.current_color()
        actor_features = self._observation_features(
            game,
            current_color,
            self.actor_observation_level,
        )
        if self.critic_mode == "full":
            critic_features = self._observation_features(
                game,
                self.color,
                self.critic_observation_level,
            )
        else:
            full_guess = self._guessed_dev_cards_features(game)
            critic_features = self._select_observation_features(
                full_guess,
                len(game.state.colors),
                self.critic_observation_level,
            )
        return inference_backend.evaluate_leaf(actor_features, critic_features)

    def _observation_features(
        self,
        game,
        color: Color,
        level: ActorObservationLevel | CriticObservationLevel,
    ) -> np.ndarray:
        num_players = len(game.state.colors)
        full_features = full_game_to_features(
            game,
            num_players,
            self.map_type,
            base_color=color,
        )
        return self._select_observation_features(full_features, num_players, level)

    def _select_observation_features(
        self,
        full_features: np.ndarray,
        num_players: int,
        level: ActorObservationLevel | CriticObservationLevel,
    ) -> np.ndarray:
        indices = self._get_observation_indices(num_players, level)
        return full_features[indices].astype(np.float32, copy=False)

    def _guessed_dev_cards_features(self, game) -> np.ndarray:
        num_players = len(game.state.colors)
        features = full_game_to_features(
            game,
            num_players,
            self.map_type,
            base_color=self.color,
        ).astype(np.float32, copy=True)
        sample = create_sample(game, self.color)
        dev_guess = self._sample_opponent_dev_cards(sample, num_players)
        numeric_name_to_index = self._get_full_numeric_index(num_players)

        for player_idx, card_counter in dev_guess.items():
            vp_feature = f"P{player_idx}_VICTORY_POINT_IN_HAND"
            old_vp = float(features[numeric_name_to_index[vp_feature]])
            new_vp = float(card_counter.get("VICTORY_POINT", 0))
            features[numeric_name_to_index[vp_feature]] = new_vp
            for dev_card in DEVELOPMENT_CARDS:
                feature_name = f"P{player_idx}_{dev_card}_IN_HAND"
                features[numeric_name_to_index[feature_name]] = float(card_counter.get(dev_card, 0))
            actual_vps_feature = f"P{player_idx}_ACTUAL_VPS"
            if actual_vps_feature in numeric_name_to_index:
                features[numeric_name_to_index[actual_vps_feature]] += new_vp - old_vp
        return features

    def _sample_opponent_dev_cards(self, sample, num_players: int) -> dict[int, Counter]:
        known_cards = Counter()
        for dev_card in DEVELOPMENT_CARDS:
            known_cards[dev_card] += int(sample.get(f"P0_{dev_card}_IN_HAND", 0))
            for player_idx in range(num_players):
                known_cards[dev_card] += int(sample.get(f"P{player_idx}_{dev_card}_PLAYED", 0))

        unseen = self._starting_dev_counter - known_cards
        unseen_cards = [card for card, count in unseen.items() for _ in range(max(0, int(count)))]
        random.shuffle(unseen_cards)

        opponent_card_counts = {
            player_idx: int(sample.get(f"P{player_idx}_NUM_DEVS_IN_HAND", 0))
            for player_idx in range(1, num_players)
        }
        guessed: dict[int, Counter] = {}
        cursor = 0
        for player_idx in range(1, num_players):
            num_cards = max(0, opponent_card_counts[player_idx])
            draw = unseen_cards[cursor : cursor + num_cards]
            cursor += num_cards
            hand_counter = Counter(draw)
            if len(draw) < num_cards:
                self._fill_missing_dev_cards(hand_counter, num_cards - len(draw), unseen)
            guessed[player_idx] = hand_counter
        return guessed

    @staticmethod
    def _fill_missing_dev_cards(hand_counter: Counter, missing: int, unseen: Counter) -> None:
        if missing <= 0:
            return
        fallback_pool = [card for card, count in unseen.items() for _ in range(max(0, int(count)))]
        if not fallback_pool:
            fallback_pool = ["KNIGHT"]
        for _ in range(missing):
            hand_counter[random.choice(fallback_pool)] += 1

    def _get_full_numeric_index(self, num_players: int) -> dict[str, int]:
        index = self._critic_index_cache.get(num_players)
        if index is not None:
            return index
        numeric_names = get_full_numeric_feature_names(num_players, self.map_type)
        index = {name: idx for idx, name in enumerate(numeric_names)}
        self._critic_index_cache[num_players] = index
        return index

    def _get_observation_indices(
        self,
        num_players: int,
        level: ActorObservationLevel | CriticObservationLevel,
    ) -> np.ndarray:
        cache_key = (num_players, level)
        indices = self._observation_index_cache.get(cache_key)
        if indices is not None:
            return indices
        indices = get_observation_indices_from_full(num_players, self.map_type, level)
        self._observation_index_cache[cache_key] = indices
        return indices

    def _resolve_root_action(
        self,
        best_action,
        playable_actions,
        num_players: int,
        game_colors: Tuple[Color, ...],
    ):
        best_idx = to_action_space(best_action, num_players, self.map_type, game_colors)
        for action in playable_actions:
            if to_action_space(action, num_players, self.map_type, game_colors) == best_idx:
                return action
        raise ValueError("NN MCTS Player chose unavailable action.")

    def __repr__(self) -> str:
        return super().__repr__().replace("Player", "NNMCTSPlayer")


class NNMCTSGuessedDevCardsPlayer(NNMCTSPlayer):
    """
    NNMCTS player that hides exact opponent dev-card types from the critic.
    """

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("critic_mode", "guessed_dev_cards")
        super().__init__(*args, **kwargs)
