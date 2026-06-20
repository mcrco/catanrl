from __future__ import annotations

import math
import multiprocessing as mp
import queue
import random
import threading
import time
from collections import defaultdict
from concurrent.futures import Future
from dataclasses import dataclass, field, replace
from typing import Callable, Literal, Sequence, Tuple

import numpy as np
import torch
from catanatron.game import Game
from catanatron.models.player import Color, Player, RandomPlayer
from catanatron.players.mcts import MCTSPlayer
from catanatron.players.minimax import AlphaBetaPlayer, SameTurnAlphaBetaPlayer
from catanatron.players.playouts import GreedyPlayoutsPlayer
from catanatron.players.search import VictoryPointPlayer
from catanatron.players.tree_search_utils import execute_spectrum, list_prunned_actions
from catanatron.players.value import ValueFunctionPlayer
from catanatron.players.weighted_random import WeightedRandomPlayer

from catanrl.beliefs.dev_card_belief import DevCardBelief
from catanrl.beliefs.determinize import determinize_game
from catanrl.features.catanatron_utils import (
    ActorObservationLevel,
    CriticObservationLevel,
    full_game_to_features,
    get_observation_indices_from_full,
)
from catanrl.models import PolicyNetworkWrapper, PolicyValueNetworkWrapper, ValueNetworkWrapper
from catanrl.models.inference_utils import forward_policy_value
from catanrl.utils.catanatron_action_space import get_action_space_size, to_action_space

EPSILON = 1e-8

# Populated in the parent immediately before fork-based within-tree workers start.
_FORK_SEARCH_CTX: dict[str, object] = {}


def _fork_search_worker_main(worker_idx: int) -> None:
    """Run claim-and-simulate loop in a fork child (shared address space + tree)."""
    ctx = _FORK_SEARCH_CTX
    player: "NNMCTSPlayer" = ctx["player"]  # type: ignore[assignment]
    root: _Node = ctx["root"]  # type: ignore[assignment]
    child_backends: list[_NNMCTSInferenceBackend] = ctx["child_backends"]  # type: ignore[assignment]
    inference_backend = child_backends[worker_idx]
    completed: mp.sharedctypes.Synchronized = ctx["completed"]  # type: ignore[assignment]
    completed_lock: mp.synchronize.Lock = ctx["completed_lock"]  # type: ignore[assignment]
    num_simulations = int(ctx["num_simulations"])
    first_error: list[BaseException] = ctx.setdefault("first_error", [])  # type: ignore[assignment]

    def claim_simulation() -> bool:
        with completed_lock:
            if int(completed.value) >= num_simulations or first_error:
                return False
            completed.value += 1
            return True

    try:
        while claim_simulation():
            player._run_parallel_simulation(root, inference_backend)
    except BaseException as exc:
        with completed_lock:
            first_error.append(exc)


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

    def __post_init__(self):
        self.children: dict[object, list[tuple["_Node", float]]] = defaultdict(list)
        self.action_priors: dict[object, float] = {}
        # Perspective of the player to move at this node. Values stored in this
        # node's value_sum are from this player's point of view; backprop and
        # selection re-orient across perspective changes (N-player correct).
        self.to_play: Color = self.game.state.current_color()

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
        policy_model: PolicyNetworkWrapper | PolicyValueNetworkWrapper,
        critic_model: ValueNetworkWrapper | None,
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
        with torch.inference_mode():
            actor_tensor = torch.from_numpy(actor_features.reshape(1, -1)).to(self.device)
            critic_tensor = torch.from_numpy(critic_features.reshape(1, -1)).to(self.device)
            logits, value_tensor = forward_policy_value(
                self.policy_model,
                self.critic_model,
                actor_tensor,
                self.model_type,
                critic_states=critic_tensor,
            )
            combined = torch.cat([logits.reshape(-1), value_tensor.reshape(-1)]).to("cpu")

        combined_np = combined.numpy()
        policy_logits = np.array(combined_np[:-1], dtype=np.float32)
        value = float(np.clip(float(combined_np[-1]), -1.0, 1.0))
        return _LeafEvaluation(policy_logits=policy_logits, value=value)


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


class _SyncRemoteNNMCTSInferenceBackend(_NNMCTSInferenceBackend):
    """Blocking remote backend safe for fork children (no listener thread)."""

    def __init__(
        self,
        worker_id: int,
        request_queue: mp.Queue,
        response_queue: mp.Queue,
        response_read_lock: mp.synchronize.Lock | None = None,
        shared_request_id: mp.sharedctypes.Synchronized | None = None,
        shared_request_id_lock: mp.synchronize.Lock | None = None,
        response_queues: Sequence[mp.Queue] | None = None,
    ) -> None:
        self.worker_id = worker_id
        self.request_queue = request_queue
        self.response_queue = response_queue
        self.response_queues = list(response_queues) if response_queues is not None else [response_queue]
        self.response_read_lock = response_read_lock
        self.shared_request_id = shared_request_id
        self.shared_request_id_lock = shared_request_id_lock
        self._closed = False
        self._next_request_id = 0
        self._request_lock = threading.Lock()

    def _allocate_request_id(self) -> int:
        if self.shared_request_id is not None and self.shared_request_id_lock is not None:
            with self.shared_request_id_lock:
                self.shared_request_id.value += 1
                return int(self.shared_request_id.value)
        with self._request_lock:
            request_id = self._next_request_id
            self._next_request_id += 1
            return request_id

    def evaluate_leaf(
        self,
        actor_features: np.ndarray,
        critic_features: np.ndarray,
    ) -> _LeafEvaluation:
        if self._closed:
            raise RuntimeError("Sync remote NN MCTS inference backend is closed.")

        request_id = self._allocate_request_id()

        request = _RemoteLeafEvaluationRequest(
            request_id=request_id,
            worker_id=self.worker_id,
            actor_features=actor_features.astype(np.float32, copy=False),
            critic_features=critic_features.astype(np.float32, copy=False),
        )

        def _await_response() -> _LeafEvaluation:
            while True:
                message = self.response_queue.get()
                if message is None:
                    raise RuntimeError("Sync remote NN MCTS inference backend is closed.")
                if int(message["request_id"]) != request_id:
                    continue
                error = message.get("error")
                if error is not None:
                    raise RuntimeError(str(error))
                return _LeafEvaluation(
                    policy_logits=message["policy_logits"],
                    value=float(message["value"]),
                )

        self.request_queue.put(request)
        if self.response_read_lock is not None:
            with self.response_read_lock:
                return _await_response()
        return _await_response()

    def close(self) -> None:
        self._closed = True


class _CentralNNMCTSInferenceServer(threading.Thread):
    def __init__(
        self,
        policy_model: PolicyNetworkWrapper | PolicyValueNetworkWrapper,
        critic_model: ValueNetworkWrapper | None,
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
                critic_tensor = torch.from_numpy(critic_features).to(self.device)
                logits, values = forward_policy_value(
                    self.policy_model,
                    self.critic_model,
                    actor_tensor,
                    self.model_type,
                    critic_states=critic_tensor,
                )
                policy_logits = logits.detach().cpu().numpy()
                values_np = values.detach().cpu().numpy()

            for request, logits_np, value in zip(batch, policy_logits, values_np):
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
        policy_model: PolicyNetworkWrapper | PolicyValueNetworkWrapper | None,
        critic_model: ValueNetworkWrapper | None,
        map_type: Literal["BASE", "MINI", "TOURNAMENT"] = "BASE",
        num_simulations: int = 64,
        c_puct: float = 1.5,
        prunning: bool = False,
        opponent_policy: str = "self",
        opponent_factory: Callable[[Color], Player] | None = None,
        actor_observation_level: ActorObservationLevel = "private",
        critic_observation_level: CriticObservationLevel = "full",
        num_search_workers: int = 1,
        inference_batch_size: int = 32,
        inference_wait_ms: float = 1.0,
        virtual_loss: float = 1.0,
        dirichlet_alpha: float = 0.3,
        dirichlet_frac: float = 0.25,
        ismcts_determinizations: int = 1,
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
        self.actor_observation_level = actor_observation_level
        self.critic_observation_level = critic_observation_level
        self.num_search_workers = max(1, int(num_search_workers))
        self.inference_batch_size = max(1, int(inference_batch_size))
        self.inference_wait_ms = max(0.0, float(inference_wait_ms))
        self.virtual_loss = float(virtual_loss)
        self._tree_process_lock: mp.synchronize.Lock | None = None
        # Training-only exploration knobs (consumed by run_search_policy; eval's
        # decide() ignores them).
        self.dirichlet_alpha = float(dirichlet_alpha)
        self.dirichlet_frac = float(dirichlet_frac)
        # Information-Set MCTS via determinization (PIMC): run an independent
        # search per sampled/enumerated determinization of opponents' hidden dev
        # cards and aggregate root visit counts weighted by belief probability.
        # A single determinization (the default) means plain perfect-info search.
        self.ismcts_determinizations = max(1, int(ismcts_determinizations))
        self.ismcts = self.ismcts_determinizations > 1
        if self.ismcts and actor_observation_level != "full":
            raise ValueError(
                "Information-Set MCTS (ismcts_determinizations > 1) requires "
                "actor_observation_level='full': it samples a full belief state, so "
                f"privatizing the policy input again is inconsistent (got "
                f"'{actor_observation_level}')."
            )
        self._opponents_by_color: dict[Color, Player] = {}
        self._observation_index_cache: dict[tuple[int, str], np.ndarray] = {}

        if self.opponent_factory is None and self.opponent_policy != "self":
            self.opponent_factory = self._build_opponent_factory(self.opponent_policy)
        self.policy_model = None
        self.critic_model = None
        self._owns_inference_backend = inference_backend is None
        if inference_backend is None:
            if policy_model is None:
                raise ValueError(
                    "policy_model is required when inference_backend is not supplied."
                )
            uses_shared_network = isinstance(policy_model, PolicyValueNetworkWrapper)
            if not uses_shared_network and critic_model is None:
                raise ValueError(
                    "critic_model is required when policy_model is not a PolicyValueNetworkWrapper."
                )
            self.policy_model = policy_model.to(self.device)
            self.critic_model = None if uses_shared_network else critic_model.to(self.device)
            self.policy_model.eval()
            if self.critic_model is not None:
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

        num_players = len(game.state.colors)
        game_colors = tuple(game.state.colors)

        if self.ismcts:
            agg_visits = self._ismcts_aggregate_visits(game, add_noise=False)
            indices = [
                to_action_space(action, num_players, self.map_type, game_colors)
                for action in playable_actions
            ]
            playable_visits = agg_visits[indices]
            if playable_visits.sum() <= 0:
                return playable_actions[0]
            return playable_actions[int(playable_visits.argmax())]

        root = self._search_root(game, add_noise=False)
        if not root.children:
            return playable_actions[0]

        best_action = max(
            root.children.keys(),
            key=lambda action: sum(child.visits for child, _ in root.children[action]),
        )
        return self._resolve_root_action(
            best_action, playable_actions, num_players, game_colors
        )

    def _search_root(self, game, add_noise: bool = False) -> _Node:
        """Build the root, expand it, optionally add Dirichlet noise, and run
        the configured number of simulations. Shared by decide() (eval) and
        run_search_policy() (training data collection)."""
        root = _Node(game=game.copy(), parent=None)
        self._expand(root, self._inference_backend)
        if not root.children:
            return root
        if add_noise and self.dirichlet_frac > 0.0:
            self._add_root_dirichlet_noise(root)
        if self.num_search_workers <= 1:
            for _ in range(self.num_simulations):
                self._run_simulation(root, self._inference_backend)
        else:
            self._run_parallel_search(root)
        return root

    def _add_root_dirichlet_noise(self, root: _Node) -> None:
        priors = root.action_priors
        if not priors:
            return
        actions = list(priors.keys())
        noise = np.random.dirichlet([self.dirichlet_alpha] * len(actions))
        frac = self.dirichlet_frac
        for action, noise_val in zip(actions, noise):
            priors[action] = priors[action] * (1.0 - frac) + float(noise_val) * frac

    def _determinizations(self, game) -> list[tuple[Game, float]]:
        """Belief-weighted determinizations of opponents' hidden dev cards.

        The belief is taken from the current mover's perspective (the player about
        to act). In 1v1 the belief is enumerated exactly and (optionally)
        subsampled to ``num_determinizations``; otherwise it is sampled.
        """
        perspective = game.state.current_color()
        belief = DevCardBelief(game, perspective)
        if belief.num_players == 2:
            hypotheses = belief.enumerate_hypotheses()
            if len(hypotheses) > self.ismcts_determinizations:
                hypotheses = self._subsample_hypotheses(
                    hypotheses, self.ismcts_determinizations
                )
        else:
            hypotheses = belief.sample_hypotheses(self.ismcts_determinizations, random)
        return [
            (determinize_game(belief, hyp, rng=random), hyp.weight)
            for hyp in hypotheses
        ]

    @staticmethod
    def _subsample_hypotheses(hypotheses, k: int):
        """Sample ``k`` hypotheses without replacement by weight and renormalize."""
        weights = np.array([h.weight for h in hypotheses], dtype=np.float64)
        weights = weights / weights.sum()
        chosen_idx = np.random.choice(
            len(hypotheses), size=k, replace=False, p=weights
        )
        chosen = [hypotheses[i] for i in chosen_idx]
        total = sum(hypotheses[i].weight for i in chosen_idx)
        if total <= 0:
            total = 1.0
        return [replace(h, weight=h.weight / total) for h in chosen]

    def _ismcts_aggregate_visits(self, game, add_noise: bool) -> np.ndarray:
        """Run one search per determinization and aggregate belief-weighted visits.

        Returns visit counts over the flat action space (indexed by
        ``to_action_space``), summed across determinizations and weighted by each
        determinization's belief probability.
        """
        num_players = len(game.state.colors)
        game_colors = tuple(game.state.colors)
        action_space_size = get_action_space_size(num_players, self.map_type)
        aggregate = np.zeros(action_space_size, dtype=np.float64)

        for det_game, weight in self._determinizations(game):
            root = self._search_root(det_game, add_noise=add_noise)
            if not root.children:
                continue
            for action, kids in root.children.items():
                idx = to_action_space(action, num_players, self.map_type, game_colors)
                aggregate[idx] += weight * sum(child.visits for child, _ in kids)
        return aggregate

    def _ismcts_search_policy(
        self,
        game,
        temperature: float,
        add_noise: bool,
        playable_actions: list,
    ) -> Tuple[np.ndarray, object]:
        num_players = len(game.state.colors)
        game_colors = tuple(game.state.colors)
        action_space_size = get_action_space_size(num_players, self.map_type)
        policy = np.zeros(action_space_size, dtype=np.float32)

        aggregate = self._ismcts_aggregate_visits(game, add_noise=add_noise)
        indices = [
            to_action_space(action, num_players, self.map_type, game_colors)
            for action in playable_actions
        ]
        visits = aggregate[indices].astype(np.float64)

        if visits.sum() <= 0:
            uniform = 1.0 / len(playable_actions)
            for idx in indices:
                policy[idx] = uniform
            return policy, random.choice(playable_actions)

        if temperature <= 1e-3:
            best = int(visits.argmax())
            policy[indices[best]] = 1.0
            return policy, playable_actions[best]

        adjusted = visits ** (1.0 / temperature)
        total = adjusted.sum()
        adjusted = (
            adjusted / total
            if total > 0
            else np.ones_like(adjusted) / len(adjusted)
        )
        for idx, prob in zip(indices, adjusted):
            policy[idx] = prob
        chosen = playable_actions[int(np.random.choice(len(playable_actions), p=adjusted))]
        return policy, chosen

    def run_search_policy(
        self,
        game,
        temperature: float,
        add_noise: bool = True,
    ) -> Tuple[np.ndarray, object]:
        """Run MCTS and return (visit-count policy target over the flat action
        space, sampled action). Used to collect AlphaZero training targets."""
        num_players = len(game.state.colors)
        game_colors = tuple(game.state.colors)
        action_space_size = get_action_space_size(num_players, self.map_type)
        policy = np.zeros(action_space_size, dtype=np.float32)

        playable_actions = list(game.playable_actions)
        if len(playable_actions) == 1:
            only = playable_actions[0]
            policy[to_action_space(only, num_players, self.map_type, game_colors)] = 1.0
            return policy, only

        if self.ismcts:
            return self._ismcts_search_policy(
                game, temperature, add_noise, playable_actions
            )

        root = self._search_root(game, add_noise=add_noise)
        if not root.children:
            uniform = 1.0 / len(playable_actions)
            for action in playable_actions:
                policy[to_action_space(action, num_players, self.map_type, game_colors)] = uniform
            return policy, random.choice(playable_actions)

        actions = list(root.children.keys())
        visits = np.array(
            [sum(child.visits for child, _ in root.children[action]) for action in actions],
            dtype=np.float64,
        )
        indices = [to_action_space(action, num_players, self.map_type, game_colors) for action in actions]

        if temperature <= 1e-3:
            best = int(visits.argmax())
            policy[indices[best]] = 1.0
            chosen_action = actions[best]
        else:
            adjusted = visits ** (1.0 / temperature)
            total = adjusted.sum()
            if total <= 0:
                adjusted = np.ones_like(adjusted) / len(adjusted)
            else:
                adjusted = adjusted / total
            for idx, prob in zip(indices, adjusted):
                policy[idx] = prob
            chosen_action = actions[int(np.random.choice(len(actions), p=adjusted))]

        resolved = self._resolve_root_action(chosen_action, playable_actions, num_players, game_colors)
        return policy, resolved

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
            leaf_value = 1.0 if winner == node.to_play else -1.0
        else:
            leaf_value = self._expand(node, inference_backend).value

        # leaf_value is from the leaf's to_play perspective; flip on every
        # perspective change as we walk back to the root.
        value = leaf_value
        for idx in range(len(path) - 1, -1, -1):
            path_node = path[idx]
            path_node.visits += 1
            path_node.value_sum += value
            if idx > 0 and path[idx - 1].to_play != path_node.to_play:
                value = -value

    def _run_parallel_search(self, root: _Node) -> None:
        """Within-tree parallel search via forked processes sharing one tree.

        Fork children cannot use threaded inference backends (async remote
        listeners). Local models must be on CPU; remote backends use the
        synchronous request/response path.
        """
        try:
            fork_ctx = mp.get_context("fork")
        except ValueError as exc:
            raise RuntimeError(
                "num_search_workers > 1 requires fork start method support."
            ) from exc

        if self._owns_inference_backend:
            if self.device != "cpu":
                raise RuntimeError(
                    "num_search_workers > 1 with local models requires CPU device "
                    "(fork is unsafe after CUDA init)."
                )
            if self.policy_model is None:
                raise RuntimeError("Local models are not configured for inference.")
            if self.critic_model is None and not isinstance(
                self.policy_model, PolicyValueNetworkWrapper
            ):
                raise RuntimeError("Local critic model is not configured for inference.")
            inference_backend: _NNMCTSInferenceBackend = _LocalNNMCTSInferenceBackend(
                self.policy_model,
                self.critic_model,
                self.model_type,
                self.device,
            )
            close_inference_backend = False
        else:
            backend = self._inference_backend
            if isinstance(backend, _RemoteNNMCTSInferenceBackend):
                inference_backend = _SyncRemoteNNMCTSInferenceBackend(
                    backend.worker_id,
                    backend.request_queue,
                    backend.response_queue,
                )
            elif isinstance(backend, _SyncRemoteNNMCTSInferenceBackend):
                inference_backend = backend
            else:
                inference_backend = backend
            close_inference_backend = False

        child_backends: list[_NNMCTSInferenceBackend]
        if isinstance(inference_backend, _SyncRemoteNNMCTSInferenceBackend):
            if len(inference_backend.response_queues) < self.num_search_workers:
                raise RuntimeError(
                    "Process parallel search requires one response queue per search worker "
                    f"({self.num_search_workers} workers, "
                    f"{len(inference_backend.response_queues)} response queues)."
                )
            child_backends = [
                _SyncRemoteNNMCTSInferenceBackend(
                    worker_id=idx,
                    request_queue=inference_backend.request_queue,
                    response_queue=inference_backend.response_queues[idx],
                )
                for idx in range(self.num_search_workers)
            ]
        else:
            child_backends = [inference_backend] * self.num_search_workers

        completed = fork_ctx.Value("i", 0)
        completed_lock = fork_ctx.Lock()
        self._tree_process_lock = fork_ctx.Lock()
        global _FORK_SEARCH_CTX
        _FORK_SEARCH_CTX = {
            "player": self,
            "root": root,
            "child_backends": child_backends,
            "completed": completed,
            "completed_lock": completed_lock,
            "num_simulations": self.num_simulations,
            "first_error": [],
        }

        processes = [
            fork_ctx.Process(
                target=_fork_search_worker_main,
                args=(idx,),
                name=f"nn-mcts-search-proc-{idx}",
                daemon=False,
            )
            for idx in range(self.num_search_workers)
        ]
        try:
            for process in processes:
                process.start()
            for process in processes:
                process.join()
        finally:
            self._tree_process_lock = None
            if close_inference_backend:
                inference_backend.close()

        first_error = _FORK_SEARCH_CTX.get("first_error", [])
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
                leaf_value = 1.0 if winner == node.to_play else -1.0
                self._backpropagate_parallel(path, leaf_value)
                return

            while node.expanding:
                time.sleep(0)
            with self._node_lock(node):
                if node.expanded:
                    child = self._select_locked(node)
                    child.virtual_visits += 1
                    vl_sign = 1.0 if child.to_play == node.to_play else -1.0
                    child.virtual_value_sum -= vl_sign * self.virtual_loss
                    node = child
                    path.append(child)
                    continue
                node.expanding = True

            try:
                leaf_value = self._expand(node, inference_backend).value
            finally:
                node.expanding = False

            self._backpropagate_parallel(path, leaf_value)
            return

    def _backpropagate_parallel(self, path: list[_Node], leaf_value: float) -> None:
        value = leaf_value
        for idx in range(len(path) - 1, -1, -1):
            path_node = path[idx]
            with self._node_lock(path_node):
                if idx > 0 and path_node.virtual_visits > 0:
                    parent = path[idx - 1]
                    vl_sign = 1.0 if path_node.to_play == parent.to_play else -1.0
                    path_node.virtual_visits -= 1
                    path_node.virtual_value_sum += vl_sign * self.virtual_loss
                path_node.visits += 1
                path_node.value_sum += value
            if idx > 0 and path[idx - 1].to_play != path_node.to_play:
                value = -value

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

    def _select(self, node: _Node) -> _Node:
        lock = self._tree_process_lock
        if lock is not None:
            with lock:
                return self._select_locked(node)
        with node.lock:
            return self._select_locked(node)

    def _node_lock(self, node: _Node):
        lock = self._tree_process_lock
        if lock is not None:
            return lock
        return node.lock

    def _select_locked(self, node: _Node) -> _Node:
        total_visits = sum(
            child.search_visits for kids in node.children.values() for child, _ in kids
        )
        best_score = -float("inf")
        best_action = None

        for action, children in node.children.items():
            action_visits = sum(child.search_visits for child, _ in children)
            if action_visits > 0:
                # Each outcome child's search_value is from that child's to_play
                # perspective; re-orient it to this node's perspective before
                # averaging (negate when the player to move differs).
                q_value = sum(
                    probability
                    * (child.search_value if child.to_play == node.to_play else -child.search_value)
                    for child, probability in children
                ) / sum(probability for _, probability in children)
            else:
                q_value = 0.0

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
        # Evaluate the position from the mover's (to_play) perspective so the
        # returned value is correctly oriented for N-player backprop.
        critic_features = self._observation_features(
            game,
            current_color,
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


class NNMCTSInformationSetPlayer(NNMCTSPlayer):
    """
    NNMCTS player that runs Information-Set MCTS (PIMC) over the belief about
    opponents' hidden development cards instead of searching the true state.

    Defaults to a full-information policy, as required by IS-MCTS.
    """

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("ismcts_determinizations", 8)
        kwargs.setdefault("actor_observation_level", "full")
        super().__init__(*args, **kwargs)

