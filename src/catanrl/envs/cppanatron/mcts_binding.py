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
    q_min: float | None
    q_max: float | None


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
        search_selection: str = "puct",
        c_visit: float = 50.0,
        c_scale: float = 1.0,
    ) -> None:
        if search_selection not in ("puct", "completed-q"):
            raise ValueError("search_selection must be 'puct' or 'completed-q'")
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
        self.search_selection = search_selection
        if search_selection == "completed-q":
            configure = getattr(
                self._library,
                "cppanatron_search_enable_completed_q_selection",
                None,
            )
            if configure is None:
                self.close()
                raise RuntimeError(
                    "cppanatron library does not support completed-Q search selection"
                )
            self._check_result(configure(self._handle, float(c_visit), float(c_scale)))

    def _configure_signatures(self) -> None:
        library = self._library
        handle = ctypes.c_void_p
        float_pointer = ctypes.POINTER(ctypes.c_float)
        double_pointer = ctypes.POINTER(ctypes.c_double)
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
        if hasattr(library, "cppanatron_search_set_root_value"):
            library.cppanatron_search_set_root_value.argtypes = [
                handle,
                ctypes.c_double,
            ]
        if hasattr(library, "cppanatron_search_set_root_wdl"):
            library.cppanatron_search_set_root_wdl.argtypes = [
                handle,
                double_pointer,
                ctypes.c_size_t,
            ]
        if hasattr(library, "cppanatron_search_enable_completed_q_selection"):
            library.cppanatron_search_enable_completed_q_selection.argtypes = [
                handle,
                ctypes.c_double,
                ctypes.c_double,
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
        if hasattr(library, "cppanatron_search_select_leaf_batch"):
            library.cppanatron_search_select_leaf_batch.argtypes = [
                ctypes.POINTER(handle),
                ctypes.c_size_t,
                float_pointer,
                ctypes.c_size_t,
                ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_int32),
                ctypes.POINTER(ctypes.c_int32),
            ]
        library.cppanatron_search_evaluate_leaf.argtypes = [
            handle,
            float_pointer,
            ctypes.c_size_t,
            ctypes.c_double,
        ]
        if hasattr(library, "cppanatron_search_evaluate_leaf_wdl"):
            library.cppanatron_search_evaluate_leaf_wdl.argtypes = [
                handle,
                float_pointer,
                ctypes.c_size_t,
                double_pointer,
                ctypes.c_size_t,
            ]
        if hasattr(library, "cppanatron_search_evaluate_leaf_batch"):
            library.cppanatron_search_evaluate_leaf_batch.argtypes = [
                ctypes.POINTER(handle),
                ctypes.c_size_t,
                float_pointer,
                ctypes.c_size_t,
                ctypes.c_size_t,
                double_pointer,
            ]
        if hasattr(library, "cppanatron_search_evaluate_leaf_wdl_batch"):
            library.cppanatron_search_evaluate_leaf_wdl_batch.argtypes = [
                ctypes.POINTER(handle),
                ctypes.c_size_t,
                float_pointer,
                ctypes.c_size_t,
                ctypes.c_size_t,
                double_pointer,
                ctypes.c_size_t,
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
        if hasattr(library, "cppanatron_search_root_wdl"):
            library.cppanatron_search_root_wdl.argtypes = [
                handle,
                double_pointer,
                ctypes.c_size_t,
            ]
        if hasattr(library, "cppanatron_search_q_bounds"):
            library.cppanatron_search_q_bounds.argtypes = [
                handle,
                ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_double),
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
        if hasattr(library, "cppanatron_search_advance_to_game"):
            library.cppanatron_search_advance_to_game.argtypes = [
                handle,
                ctypes.c_size_t,
                handle,
            ]
            library.cppanatron_search_advance_to_game.restype = ctypes.c_int32

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

    @staticmethod
    def _wdl_distribution(wdl: np.ndarray) -> np.ndarray:
        probabilities = np.ascontiguousarray(wdl, dtype=np.float64)
        if probabilities.shape != (3,):
            raise ValueError(f"Expected WDL with shape (3,), got {probabilities.shape}")
        if not np.isfinite(probabilities).all() or bool((probabilities < 0.0).any()):
            raise ValueError("WDL probabilities must be finite and non-negative")
        total = float(probabilities.sum())
        if total <= 0.0:
            raise ValueError("WDL probabilities must have positive mass")
        probabilities /= total
        return probabilities

    def initialize_root(self, policy_logits: np.ndarray) -> None:
        logits = self._policy_logits(policy_logits)
        self._check_result(
            self._library.cppanatron_search_initialize_root(
                self._handle,
                logits.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                logits.size,
            )
        )

    def set_root_value(self, value: float) -> None:
        setter = getattr(self._library, "cppanatron_search_set_root_value", None)
        if setter is None:
            raise RuntimeError("cppanatron library does not support root network values")
        self._check_result(setter(self._handle, float(value)))

    def set_root_wdl(self, wdl: np.ndarray) -> None:
        probabilities = self._wdl_distribution(wdl)
        setter = getattr(self._library, "cppanatron_search_set_root_wdl", None)
        if setter is None:
            self.set_root_value(float(probabilities[0] - probabilities[2]))
            return
        self._check_result(
            setter(
                self._handle,
                probabilities.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                probabilities.size,
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

    @staticmethod
    def select_leaf_batch(
        searches: list[NativeMCTSSearch],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Select one leaf per independent tree with one native call."""

        if not searches:
            raise ValueError("At least one native search is required")
        first = searches[0]
        function = getattr(first._library, "cppanatron_search_select_leaf_batch", None)
        if function is None:
            raise RuntimeError("cppanatron library does not support batched leaf selection")
        observation_size = first.observation_size
        for search in searches:
            if search._handle is None:
                raise RuntimeError("Cannot batch a closed native search")
            if search.observation_size != observation_size:
                raise ValueError("Batched native searches must share an observation size")
        handles = (ctypes.c_void_p * len(searches))(
            *(ctypes.c_void_p(search._handle) for search in searches)
        )
        observations = np.zeros((len(searches), observation_size), dtype=np.float32)
        players = np.full(len(searches), -1, dtype=np.int32)
        statuses = np.zeros(len(searches), dtype=np.int32)
        first._check_result(
            function(
                handles,
                len(searches),
                observations.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                observations.shape[1],
                observation_size,
                players.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
                statuses.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            )
        )
        return observations, players, statuses.astype(np.bool_, copy=False)

    def evaluate_leaf(
        self,
        policy_logits: np.ndarray,
        value: float,
        wdl: np.ndarray | None = None,
    ) -> None:
        logits = self._policy_logits(policy_logits)
        evaluate_wdl = getattr(self._library, "cppanatron_search_evaluate_leaf_wdl", None)
        if wdl is not None and evaluate_wdl is not None:
            probabilities = self._wdl_distribution(wdl)
            self._check_result(
                evaluate_wdl(
                    self._handle,
                    logits.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                    logits.size,
                    probabilities.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                    probabilities.size,
                )
            )
            return
        self._check_result(
            self._library.cppanatron_search_evaluate_leaf(
                self._handle,
                logits.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                logits.size,
                float(value),
            )
        )

    @staticmethod
    def evaluate_leaf_batch(
        searches: list[NativeMCTSSearch],
        policy_logits: np.ndarray,
        values: np.ndarray,
        wdls: np.ndarray | None = None,
    ) -> None:
        """Expand and back up one pending leaf in each independent tree."""

        if not searches:
            raise ValueError("At least one native search is required")
        first = searches[0]
        for search in searches:
            if search._handle is None:
                raise RuntimeError("Cannot batch a closed native search")
            if search.action_space_size != first.action_space_size:
                raise ValueError("Batched native searches must share an action space")
        logits = np.ascontiguousarray(policy_logits, dtype=np.float32)
        value_array = np.ascontiguousarray(values, dtype=np.float64)
        expected_policy_shape = (len(searches), first.action_space_size)
        if logits.shape != expected_policy_shape:
            raise ValueError(
                f"Expected policy logits with shape {expected_policy_shape}, got {logits.shape}"
            )
        if value_array.shape != (len(searches),):
            raise ValueError(
                f"Expected values with shape ({len(searches)},), got {value_array.shape}"
            )
        handles = (ctypes.c_void_p * len(searches))(
            *(ctypes.c_void_p(search._handle) for search in searches)
        )
        if wdls is None:
            function = getattr(first._library, "cppanatron_search_evaluate_leaf_batch", None)
            if function is None:
                raise RuntimeError("cppanatron library does not support batched leaf evaluation")
            first._check_result(
                function(
                    handles,
                    len(searches),
                    logits.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                    logits.shape[1],
                    logits.shape[1],
                    value_array.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                )
            )
            return

        wdl_array = np.ascontiguousarray(wdls, dtype=np.float64)
        if wdl_array.shape != (len(searches), 3):
            raise ValueError(
                f"Expected WDL probabilities with shape ({len(searches)}, 3), got {wdl_array.shape}"
            )
        if not np.isfinite(wdl_array).all() or bool((wdl_array < 0.0).any()):
            raise ValueError("WDL probabilities must be finite and non-negative")
        totals = wdl_array.sum(axis=1, keepdims=True)
        if bool((totals <= 0.0).any()):
            raise ValueError("Each WDL row must have positive mass")
        wdl_array = np.ascontiguousarray(wdl_array / totals)
        function = getattr(first._library, "cppanatron_search_evaluate_leaf_wdl_batch", None)
        if function is None:
            raise RuntimeError("cppanatron library does not support batched WDL evaluation")
        first._check_result(
            function(
                handles,
                len(searches),
                logits.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                logits.shape[1],
                logits.shape[1],
                wdl_array.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                wdl_array.shape[1],
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

    def root_wdl(self) -> np.ndarray:
        getter = getattr(self._library, "cppanatron_search_root_wdl", None)
        if getter is None:
            value = float(self.metrics().root_value)
            return np.asarray([(1.0 + value) * 0.5, 0.0, (1.0 - value) * 0.5])
        probabilities = np.empty(3, dtype=np.float64)
        self._check_result(
            getter(
                self._handle,
                probabilities.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                probabilities.size,
            )
        )
        return probabilities

    @property
    def root_expanded(self) -> bool:
        result = int(self._library.cppanatron_search_root_expanded(self._handle))
        if result < 0:
            self._raise_last_error()
        return bool(result)

    def reset_metrics(self) -> None:
        self._check_result(self._library.cppanatron_search_reset_metrics(self._handle))

    def advance(self, action: int, observed_game: NativeGame | None = None) -> bool:
        if not 0 <= action < self.action_space_size:
            raise ValueError(f"Action index {action} is outside the flat action space")
        advance_to_game = getattr(
            self._library,
            "cppanatron_search_advance_to_game",
            None,
        )
        if observed_game is not None and advance_to_game is not None:
            result = int(advance_to_game(self._handle, action, observed_game._handle))
        else:
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
        q_min_value: float | None = None
        q_max_value: float | None = None
        bounds = getattr(self._library, "cppanatron_search_q_bounds", None)
        if bounds is not None:
            q_min = ctypes.c_double()
            q_max = ctypes.c_double()
            self._check_result(bounds(self._handle, ctypes.byref(q_min), ctypes.byref(q_max)))
            q_min_value = float(q_min.value)
            q_max_value = float(q_max.value)
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
            q_min=q_min_value,
            q_max=q_max_value,
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
