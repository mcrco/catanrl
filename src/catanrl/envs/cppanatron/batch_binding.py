from __future__ import annotations

import ctypes
from typing import Literal

import numpy as np
from catanatron.gym.board_tensor_features import (
    HEIGHT,
    WIDTH,
    get_node_and_edge_maps,
    get_tile_coordinate_map,
)

from .binding import MapType, _load_library


class _NodePosition(ctypes.Structure):
    _fields_ = [
        ("node", ctypes.c_int32),
        ("x", ctypes.c_int32),
        ("y", ctypes.c_int32),
    ]


class _EdgePosition(ctypes.Structure):
    _fields_ = [
        ("a", ctypes.c_int32),
        ("b", ctypes.c_int32),
        ("x", ctypes.c_int32),
        ("y", ctypes.c_int32),
    ]


class _TilePosition(ctypes.Structure):
    _fields_ = [
        ("x", ctypes.c_int32),
        ("y", ctypes.c_int32),
        ("z", ctypes.c_int32),
        ("board_x", ctypes.c_int32),
        ("board_y", ctypes.c_int32),
    ]


def _position_arrays():
    node_map, edge_map = get_node_and_edge_maps()
    nodes = (_NodePosition * len(node_map))(
        *(_NodePosition(node, x, y) for node, (x, y) in node_map.items())
    )

    unique_edges: dict[tuple[int, int], tuple[int, int]] = {}
    for (a, b), position in edge_map.items():
        unique_edges.setdefault(tuple(sorted((a, b))), position)
    edges = (_EdgePosition * len(unique_edges))(
        *(_EdgePosition(a, b, x, y) for (a, b), (x, y) in unique_edges.items())
    )

    tile_map = get_tile_coordinate_map()
    tiles = (_TilePosition * len(tile_map))(
        *(
            _TilePosition(int(x), int(y), int(z), board_x, board_y)
            for (x, y, z), (board_y, board_x) in tile_map.items()
        )
    )
    return nodes, edges, tiles


class NativeGameBatch:
    """Owned native vector of games writing directly into Puffer buffers."""

    _MAP_TYPES = {"BASE": 0, "MINI": 1, "TOURNAMENT": 2}
    _REWARD_FUNCTIONS = {"shaped": 0, "win": 1}

    def __init__(
        self,
        *,
        num_envs: int,
        num_players: int,
        map_type: MapType,
        discard_limit: int,
        vps_to_win: int,
        reward_function: Literal["shaped", "win"],
        turns_limit: int,
        observations: np.ndarray,
        obs_dtype: np.dtype,
        actions: np.ndarray,
        rewards: np.ndarray,
        terminals: np.ndarray,
        truncations: np.ndarray,
        masks: np.ndarray,
    ) -> None:
        self._handle = None
        self._library = _load_library()
        self._configure_signatures()
        nodes, edges, tiles = _position_arrays()
        self._handle = self._library.cppanatron_batch_create(
            num_envs,
            num_players,
            self._MAP_TYPES[map_type],
            discard_limit,
            0,
            vps_to_win,
            1,
            self._REWARD_FUNCTIONS[reward_function],
            turns_limit,
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

        action_mask_offset = obs_dtype.fields["action_mask"][1]
        observation_offset = obs_dtype.fields["observation"][1]
        self._check_array(observations, np.uint8, "observations")
        self._check_array(actions, np.int32, "actions")
        self._check_array(rewards, np.float32, "rewards")
        self._check_array(terminals, np.bool_, "terminals")
        self._check_array(truncations, np.bool_, "truncations")
        self._check_array(masks, np.bool_, "masks")
        result = self._library.cppanatron_batch_bind_buffers(
            self._handle,
            observations.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            observations.strides[0],
            action_mask_offset,
            observation_offset,
            actions.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            rewards.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            terminals.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            truncations.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            masks.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        )
        self._check_result(result)

    @staticmethod
    def _check_array(array: np.ndarray, dtype, name: str) -> None:
        if array.dtype != dtype or not array.flags.c_contiguous:
            raise ValueError(f"{name} must be contiguous {np.dtype(dtype)}")

    def _configure_signatures(self) -> None:
        library = self._library
        handle = ctypes.c_void_p
        library.cppanatron_batch_create.argtypes = [
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.POINTER(_NodePosition),
            ctypes.c_size_t,
            ctypes.POINTER(_EdgePosition),
            ctypes.c_size_t,
            ctypes.POINTER(_TilePosition),
            ctypes.c_size_t,
        ]
        library.cppanatron_batch_create.restype = handle
        library.cppanatron_batch_destroy.argtypes = [handle]
        library.cppanatron_batch_bind_buffers.argtypes = [
            handle,
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.POINTER(ctypes.c_uint8),
        ]
        library.cppanatron_batch_reset_all.argtypes = [
            handle,
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.c_size_t,
        ]
        library.cppanatron_batch_reset_at.argtypes = [
            handle,
            ctypes.c_int32,
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_int32,
        ]
        library.cppanatron_batch_step.argtypes = [handle]

    def _raise_last_error(self) -> None:
        message = self._library.cppanatron_last_error()
        raise RuntimeError(message.decode("utf-8") if message else "unknown cppanatron error")

    def _check_result(self, result: int) -> None:
        if result != 0:
            self._raise_last_error()

    def reset_all(
        self,
        map_seeds: np.ndarray,
        game_seeds: np.ndarray,
    ) -> None:
        map_seeds = np.ascontiguousarray(map_seeds, dtype=np.uint64)
        game_seeds = np.ascontiguousarray(game_seeds, dtype=np.uint64)
        if map_seeds.shape != game_seeds.shape or map_seeds.ndim != 1:
            raise ValueError("map and game seeds must be matching vectors")
        result = self._library.cppanatron_batch_reset_all(
            self._handle,
            map_seeds.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
            game_seeds.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
            len(map_seeds),
        )
        self._check_result(result)

    def reset_at(
        self,
        env_index: int,
        map_seed: int,
        game_seed: int,
        *,
        preserve_transition: bool,
    ) -> None:
        self._check_result(
            self._library.cppanatron_batch_reset_at(
                self._handle,
                env_index,
                map_seed,
                game_seed,
                preserve_transition,
            )
        )

    def step(self) -> None:
        self._check_result(self._library.cppanatron_batch_step(self._handle))

    def close(self) -> None:
        if self._handle:
            self._library.cppanatron_batch_destroy(self._handle)
            self._handle = None

    def __del__(self) -> None:
        self.close()


__all__ = ["NativeGameBatch"]
