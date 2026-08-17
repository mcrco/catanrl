from __future__ import annotations

import ctypes
from typing import Literal

import numpy as np
from catanatron.gym.board_tensor_features import HEIGHT, WIDTH

from .binding import MapType, _load_library, observation_layout_arrays


class NativeGameBatch:
    """GPU-backed vector of games writing into host Puffer buffers."""

    _MAP_TYPES = {"BASE": 0, "MINI": 1, "TOURNAMENT": 2}
    _REWARD_FUNCTIONS = {"shaped": 0, "win": 1}
    _NUMBER_PLACEMENTS = {"official_spiral": 0, "random": 1}

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
        number_placement: Literal["official_spiral", "random"] = "random",
    ) -> None:
        self._handle = None
        self._library = _load_library()
        self._configure_signatures()
        width, height, nodes, edges, tiles = observation_layout_arrays()
        if width != WIDTH or height != HEIGHT:
            raise ValueError("observation layout dimensions do not match the gym board tensor")
        self._handle = self._library.cudanatron_batch_create(
            num_envs,
            num_players,
            self._MAP_TYPES[map_type],
            discard_limit,
            0,
            vps_to_win,
            self._NUMBER_PLACEMENTS[number_placement],
            self._REWARD_FUNCTIONS[reward_function],
            turns_limit,
            WIDTH,
            HEIGHT,
            ctypes.cast(nodes, ctypes.c_void_p),
            len(nodes),
            ctypes.cast(edges, ctypes.c_void_p),
            len(edges),
            ctypes.cast(tiles, ctypes.c_void_p),
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
        self._check(
            self._library.cudanatron_batch_bind_buffers(
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
        )
        self.action_space_size = int(self._library.cudanatron_batch_action_space_size(self._handle))
        self.observation_size = int(self._library.cudanatron_batch_observation_size(self._handle))

    @staticmethod
    def _check_array(array: np.ndarray, dtype, name: str) -> None:
        if array.dtype != dtype or not array.flags.c_contiguous:
            raise ValueError(f"{name} must be contiguous {np.dtype(dtype)}")

    def _configure_signatures(self) -> None:
        library = self._library
        handle = ctypes.c_void_p
        library.cudanatron_batch_create.argtypes = [
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
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_void_p,
            ctypes.c_size_t,
        ]
        library.cudanatron_batch_create.restype = handle
        library.cudanatron_batch_destroy.argtypes = [handle]
        library.cudanatron_batch_bind_buffers.argtypes = [
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
        library.cudanatron_batch_reset_all.argtypes = [
            handle,
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.c_size_t,
        ]
        library.cudanatron_batch_reset_at.argtypes = [
            handle,
            ctypes.c_int32,
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_int32,
        ]
        library.cudanatron_batch_step.argtypes = [handle]
        library.cudanatron_batch_action_space_size.argtypes = [handle]
        library.cudanatron_batch_action_space_size.restype = ctypes.c_int32
        library.cudanatron_batch_observation_size.argtypes = [handle]
        library.cudanatron_batch_observation_size.restype = ctypes.c_int32

    def _raise_last_error(self) -> None:
        message = self._library.cudanatron_last_error()
        raise RuntimeError(message.decode() if message else "unknown cudanatron error")

    def _check(self, result: int) -> None:
        if result != 0:
            self._raise_last_error()

    def reset_all(self, map_seeds: np.ndarray, game_seeds: np.ndarray) -> None:
        map_seeds = np.ascontiguousarray(map_seeds, dtype=np.uint64)
        game_seeds = np.ascontiguousarray(game_seeds, dtype=np.uint64)
        if map_seeds.shape != game_seeds.shape or map_seeds.ndim != 1:
            raise ValueError("map and game seeds must be matching vectors")
        self._check(
            self._library.cudanatron_batch_reset_all(
                self._handle,
                map_seeds.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
                game_seeds.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
                len(map_seeds),
            )
        )

    def reset_at(
        self,
        env_index: int,
        map_seed: int,
        game_seed: int,
        *,
        preserve_transition: bool,
    ) -> None:
        self._check(
            self._library.cudanatron_batch_reset_at(
                self._handle,
                env_index,
                map_seed,
                game_seed,
                int(preserve_transition),
            )
        )

    def step(self) -> None:
        self._check(self._library.cudanatron_batch_step(self._handle))

    def close(self) -> None:
        if self._handle:
            self._library.cudanatron_batch_destroy(self._handle)
            self._handle = None

    def __enter__(self) -> NativeGameBatch:
        return self

    def __exit__(self, *_args) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()
