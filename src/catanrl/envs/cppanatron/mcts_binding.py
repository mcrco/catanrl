from __future__ import annotations

import ctypes
from dataclasses import dataclass

import numpy as np
from catanatron.gym.board_tensor_features import HEIGHT, WIDTH

from catanrl.features.catanatron_utils import get_observation_indices_from_full

from .batch_binding import _position_arrays
from .binding import MapType, NativeGame, _load_library


class _SearchMetrics(ctypes.Structure):
    _fields_ = [
        ("simulations", ctypes.c_uint64),
        ("principal_variation_depth", ctypes.c_uint32),
        ("maximum_depth", ctypes.c_uint32),
        ("mean_depth", ctypes.c_double),
        ("root_value", ctypes.c_double),
        ("retained_root_visits", ctypes.c_uint32),
        ("pruned_actions", ctypes.c_uint64),
        ("coalesced_outcomes", ctypes.c_uint64),
        ("tree_reused", ctypes.c_int32),
    ]


@dataclass(frozen=True)
class NativeMCTSSearchMetrics:
    simulations: int
    principal_variation_depth: int
    maximum_depth: int
    mean_depth: float
    root_value: float
    retained_root_visits: int
    pruned_actions: int
    coalesced_outcomes: int
    tree_reused: bool


class NativeMCTSSearch:
    """Pull-based Python binding for one native cppanatron MCTS tree.

    The caller evaluates the root and each selected leaf with its own inference
    backend. This deliberately keeps PyTorch out of the C++ engine and allows
    leaf requests from independent self-play workers to share the existing
    centrally batched inference server.
    """

    def __init__(
        self,
        game: NativeGame,
        map_type: MapType,
        *,
        c_puct: float = 1.5,
        seed: int = 0,
        canonical_pruning: bool = False,
    ) -> None:
        self._handle = None
        self._library = _load_library()
        self._configure_signatures()
        nodes, edges, tiles = _position_arrays()
        self.observation_size = len(
            get_observation_indices_from_full(
                game.num_players,
                map_type,
                "full",
            )
        )
        self.action_space_size = game.action_space_size
        self._leaf_observation = np.empty(self.observation_size, dtype=np.float32)

        self._handle = self._library.cppanatron_search_create(
            game._handle,
            float(c_puct),
            int(seed),
            int(canonical_pruning),
            WIDTH,
            HEIGHT,
            nodes,
            len(nodes),
            edges,
            len(edges),
            tiles,
            len(tiles),
        )
        if not self._handle:
            self._raise_last_error()

    def _configure_signatures(self) -> None:
        library = self._library
        handle = ctypes.c_void_p
        float_pointer = ctypes.POINTER(ctypes.c_float)
        library.cppanatron_search_create.argtypes = [
            handle,
            ctypes.c_double,
            ctypes.c_uint64,
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_void_p,
            ctypes.c_size_t,
        ]
        library.cppanatron_search_create.restype = handle
        library.cppanatron_search_destroy.argtypes = [handle]
        library.cppanatron_search_initialize_root.argtypes = [
            handle,
            float_pointer,
            ctypes.c_size_t,
        ]
        library.cppanatron_search_add_root_dirichlet_noise.argtypes = [
            handle,
            ctypes.c_double,
            ctypes.c_double,
        ]
        library.cppanatron_search_root_observation.argtypes = [
            handle,
            float_pointer,
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_int32),
        ]
        library.cppanatron_search_select_leaf.argtypes = [
            handle,
            float_pointer,
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_int32),
        ]
        library.cppanatron_search_select_leaf.restype = ctypes.c_int32
        library.cppanatron_search_evaluate_leaf.argtypes = [
            handle,
            float_pointer,
            ctypes.c_size_t,
            ctypes.c_double,
        ]
        library.cppanatron_search_root_visits.argtypes = [
            handle,
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.c_size_t,
        ]
        library.cppanatron_search_root_action_values.argtypes = [
            handle,
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_size_t,
        ]
        library.cppanatron_search_get_metrics.argtypes = [
            handle,
            ctypes.POINTER(_SearchMetrics),
        ]
        library.cppanatron_search_root_expanded.argtypes = [handle]
        library.cppanatron_search_root_expanded.restype = ctypes.c_int32
        library.cppanatron_search_reset_metrics.argtypes = [handle]
        library.cppanatron_search_advance.argtypes = [handle, ctypes.c_size_t]
        library.cppanatron_search_advance.restype = ctypes.c_int32

    def _raise_last_error(self) -> None:
        message = self._library.cppanatron_last_error()
        raise RuntimeError(message.decode("utf-8") if message else "unknown cppanatron error")

    def _check_result(self, result: int) -> None:
        if result != 0:
            self._raise_last_error()

    def _policy_logits(self, policy_logits: np.ndarray) -> np.ndarray:
        logits = np.ascontiguousarray(policy_logits, dtype=np.float32)
        if logits.shape != (self.action_space_size,):
            raise ValueError(
                f"Expected policy logits with shape ({self.action_space_size},), got {logits.shape}"
            )
        return logits

    def initialize_root(self, policy_logits: np.ndarray) -> None:
        logits = self._policy_logits(policy_logits)
        self._check_result(
            self._library.cppanatron_search_initialize_root(
                self._handle,
                logits.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                logits.size,
            )
        )

    def add_root_dirichlet_noise(self, alpha: float, fraction: float) -> None:
        self._check_result(
            self._library.cppanatron_search_add_root_dirichlet_noise(
                self._handle,
                float(alpha),
                float(fraction),
            )
        )

    def root_observation(self) -> tuple[np.ndarray, int]:
        player = ctypes.c_int32()
        self._check_result(
            self._library.cppanatron_search_root_observation(
                self._handle,
                self._leaf_observation.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                self._leaf_observation.size,
                ctypes.byref(player),
            )
        )
        return self._leaf_observation.copy(), int(player.value)

    def select_leaf(self) -> tuple[np.ndarray, int] | None:
        player = ctypes.c_int32()
        result = int(
            self._library.cppanatron_search_select_leaf(
                self._handle,
                self._leaf_observation.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                self._leaf_observation.size,
                ctypes.byref(player),
            )
        )
        if result < 0:
            self._raise_last_error()
        if result == 0:
            return None
        return self._leaf_observation.copy(), int(player.value)

    def evaluate_leaf(self, policy_logits: np.ndarray, value: float) -> None:
        logits = self._policy_logits(policy_logits)
        self._check_result(
            self._library.cppanatron_search_evaluate_leaf(
                self._handle,
                logits.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                logits.size,
                float(value),
            )
        )

    def root_visits(self) -> np.ndarray:
        visits = np.empty(self.action_space_size, dtype=np.uint32)
        self._check_result(
            self._library.cppanatron_search_root_visits(
                self._handle,
                visits.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
                visits.size,
            )
        )
        return visits

    def root_action_values(self) -> np.ndarray:
        """Expected action Q from the root player; missing actions are NaN."""
        values = np.empty(self.action_space_size, dtype=np.float64)
        self._check_result(
            self._library.cppanatron_search_root_action_values(
                self._handle,
                values.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                values.size,
            )
        )
        return values

    @property
    def root_expanded(self) -> bool:
        result = int(self._library.cppanatron_search_root_expanded(self._handle))
        if result < 0:
            self._raise_last_error()
        return bool(result)

    def reset_metrics(self) -> None:
        self._check_result(self._library.cppanatron_search_reset_metrics(self._handle))

    def advance(self, action: int) -> bool:
        if not 0 <= action < self.action_space_size:
            raise ValueError(f"Action index {action} is outside the flat action space")
        result = int(self._library.cppanatron_search_advance(self._handle, action))
        if result < 0:
            self._raise_last_error()
        return bool(result)

    def metrics(self) -> NativeMCTSSearchMetrics:
        metrics = _SearchMetrics()
        self._check_result(
            self._library.cppanatron_search_get_metrics(
                self._handle,
                ctypes.byref(metrics),
            )
        )
        return NativeMCTSSearchMetrics(
            simulations=int(metrics.simulations),
            principal_variation_depth=int(metrics.principal_variation_depth),
            maximum_depth=int(metrics.maximum_depth),
            mean_depth=float(metrics.mean_depth),
            root_value=float(metrics.root_value),
            retained_root_visits=int(metrics.retained_root_visits),
            pruned_actions=int(metrics.pruned_actions),
            coalesced_outcomes=int(metrics.coalesced_outcomes),
            tree_reused=bool(metrics.tree_reused),
        )

    def close(self) -> None:
        if self._handle:
            self._library.cppanatron_search_destroy(self._handle)
            self._handle = None

    def __enter__(self) -> NativeMCTSSearch:
        return self

    def __exit__(self, *_args) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()


__all__ = ["NativeMCTSSearch", "NativeMCTSSearchMetrics"]
