from __future__ import annotations

import ctypes

import numpy as np
from catanatron.gym.board_tensor_features import HEIGHT, WIDTH

from catanrl.features.catanatron_utils import get_observation_indices_from_full

from .batch_binding import _position_arrays
from .binding import MapType, NativeGame, _load_library


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


__all__ = ["NativeMCTSSearch"]
