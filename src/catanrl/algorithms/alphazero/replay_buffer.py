"""Replay storage implementations for AlphaZero training targets."""

from __future__ import annotations

import mmap
import random
import shutil
import tempfile
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .parallel_self_play import SelfPlayExperience


class DiskReplayBuffer(Sequence[SelfPlayExperience]):
    """Fixed-capacity replay ring backed by exact-dtype memory-mapped arrays.

    The Catanatron observation contract is substantially wider than Canopy's.
    Keeping 500k expanded observations in ordinary NumPy objects makes inactive
    replay pages compete with self-play workers for RAM. This buffer preserves
    the samples exactly while allowing the operating system to evict clean pages
    to an explicitly selected disk filesystem.
    """

    _WRITE_CHUNK_SIZE = 4096

    def __init__(self, capacity: int, storage_dir: str, *, shared_states: bool) -> None:
        if capacity < 1:
            raise ValueError("Replay capacity must be at least one")
        if not storage_dir:
            raise ValueError("Disk replay requires an explicit storage directory")
        base = Path(storage_dir).expanduser().resolve()
        base.mkdir(parents=True, exist_ok=True)
        self._directory = Path(tempfile.mkdtemp(prefix="catanrl-replay-", dir=base))
        self._capacity = capacity
        self._start = 0
        self._size = 0
        self._arrays: dict[str, np.memmap] = {}
        self._shapes: dict[str, tuple[int, ...]] = {}
        self._shared_states = shared_states
        self._closed = False

    @property
    def storage_path(self) -> str:
        return str(self._directory)

    def __len__(self) -> int:
        return self._size

    def __getitem__(self, index: int | slice) -> SelfPlayExperience | list[SelfPlayExperience]:
        if isinstance(index, slice):
            return [self[position] for position in range(*index.indices(self._size))]
        if index < 0:
            index += self._size
        if index < 0 or index >= self._size:
            raise IndexError("Replay index out of range")
        physical = (self._start + index) % self._capacity
        value_wdl = (
            np.asarray(self._arrays["value_wdl"][physical])
            if bool(self._arrays["has_value_wdl"][physical])
            else None
        )
        aux_value_targets = (
            np.asarray(self._arrays["aux_value_targets"][physical])
            if "aux_value_targets" in self._arrays
            and bool(self._arrays["has_aux_value_targets"][physical])
            else None
        )
        actor_state = np.asarray(self._arrays["actor_state"][physical])
        critic_state = (
            actor_state
            if self._shared_states
            else np.asarray(self._arrays["critic_state"][physical])
        )
        return SelfPlayExperience(
            actor_state=actor_state,
            critic_state=critic_state,
            policy=np.asarray(self._arrays["policy"][physical]),
            action_mask=np.asarray(self._arrays["action_mask"][physical]),
            value=float(self._arrays["value"][physical]),
            full_search=bool(self._arrays["full_search"][physical]),
            value_wdl=value_wdl,
            aux_value_targets=aux_value_targets,
        )

    def __iter__(self) -> Iterator[SelfPlayExperience]:
        for index in range(self._size):
            yield self[index]

    def extend(self, experiences: Iterable[SelfPlayExperience]) -> None:
        rows = list(experiences)
        if not rows:
            return
        if not self._arrays:
            self._initialize(rows[0])
        self._validate(rows)
        if len(rows) >= self._capacity:
            rows = rows[-self._capacity :]
            self._start = 0
            self._size = 0
        for offset in range(0, len(rows), self._WRITE_CHUNK_SIZE):
            self._append_chunk(rows[offset : offset + self._WRITE_CHUNK_SIZE])
        self.release_pages()

    def sample_indices(self, count: int) -> list[int]:
        if count > self._size:
            raise ValueError("Sample larger than replay population")
        return random.sample(range(self._size), count)

    def batch(self, indices: Sequence[int]) -> list[SelfPlayExperience]:
        return [self[index] for index in indices]

    def release_pages(self) -> None:
        """Flush writes and make clean replay pages immediately reclaimable."""
        for array in self._arrays.values():
            array.flush()
            mapped = getattr(array, "_mmap", None)
            if mapped is not None and hasattr(mapped, "madvise"):
                mapped.madvise(mmap.MADV_DONTNEED)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        # Do not force-close mappings while the caller may still hold the final
        # batch's ndarray views. Linux safely keeps unlinked mappings alive until
        # those views are released.
        self._arrays.clear()
        shutil.rmtree(self._directory, ignore_errors=True)

    def _initialize(self, sample: SelfPlayExperience) -> None:
        actor_state = self._as_array(sample.actor_state, np.float32, "actor_state")
        critic_state = self._as_array(sample.critic_state, np.float32, "critic_state")
        policy = self._as_array(sample.policy, np.float32, "policy")
        action_mask = self._as_array(sample.action_mask, np.bool_, "action_mask")
        if self._shared_states and actor_state.shape != critic_state.shape:
            raise ValueError("Shared replay actor and critic state shapes differ")
        self._create_array("actor_state", np.float32, actor_state.shape)
        if not self._shared_states:
            self._create_array("critic_state", np.float32, critic_state.shape)
        self._create_array("policy", np.float32, policy.shape)
        self._create_array("action_mask", np.bool_, action_mask.shape)
        self._create_array("value", np.float32, ())
        self._create_array("full_search", np.bool_, ())
        self._create_array("value_wdl", np.float32, (3,))
        self._create_array("has_value_wdl", np.bool_, ())
        if sample.aux_value_targets is not None:
            aux = self._as_array(
                sample.aux_value_targets,
                np.float32,
                "aux_value_targets",
            )
            self._create_array("aux_value_targets", np.float32, aux.shape)
            self._create_array("has_aux_value_targets", np.bool_, ())

    def _create_array(self, name: str, dtype: Any, trailing_shape: tuple[int, ...]) -> None:
        path = self._directory / f"{name}.bin"
        shape = (self._capacity, *trailing_shape)
        self._arrays[name] = np.memmap(path, mode="w+", dtype=dtype, shape=shape)
        self._shapes[name] = trailing_shape

    def _validate(self, rows: Sequence[SelfPlayExperience]) -> None:
        for row in rows:
            self._check_shape(row.actor_state, "actor_state")
            if not self._shared_states:
                self._check_shape(row.critic_state, "critic_state")
            self._check_shape(row.policy, "policy")
            self._check_shape(row.action_mask, "action_mask")
            if row.value_wdl is not None and np.asarray(row.value_wdl).shape != (3,):
                raise ValueError("Replay value_wdl shape changed")
            if "aux_value_targets" in self._arrays:
                if row.aux_value_targets is None:
                    continue
                self._check_shape(row.aux_value_targets, "aux_value_targets")
            elif row.aux_value_targets is not None:
                raise ValueError("Replay auxiliary target shape changed")

    def _check_shape(self, value: np.ndarray, name: str) -> None:
        if np.asarray(value).shape != self._shapes[name]:
            raise ValueError(f"Replay {name} shape changed")

    def _append_chunk(self, rows: Sequence[SelfPlayExperience]) -> None:
        count = len(rows)
        write_start = (self._start + self._size) % self._capacity
        overflow = max(0, self._size + count - self._capacity)
        if overflow:
            self._start = (self._start + overflow) % self._capacity
            self._size -= overflow
            write_start = (self._start + self._size) % self._capacity
        first_count = min(count, self._capacity - write_start)
        self._write_slice(write_start, rows[:first_count])
        if first_count < count:
            self._write_slice(0, rows[first_count:])
        self._size += count

    def _write_slice(self, start: int, rows: Sequence[SelfPlayExperience]) -> None:
        if not rows:
            return
        stop = start + len(rows)
        self._arrays["actor_state"][start:stop] = np.stack(
            [self._as_array(row.actor_state, np.float32, "actor_state") for row in rows]
        )
        if not self._shared_states:
            self._arrays["critic_state"][start:stop] = np.stack(
                [self._as_array(row.critic_state, np.float32, "critic_state") for row in rows]
            )
        self._arrays["policy"][start:stop] = np.stack(
            [self._as_array(row.policy, np.float32, "policy") for row in rows]
        )
        self._arrays["action_mask"][start:stop] = np.stack(
            [self._as_array(row.action_mask, np.bool_, "action_mask") for row in rows]
        )
        self._arrays["value"][start:stop] = np.asarray(
            [row.value for row in rows], dtype=np.float32
        )
        self._arrays["full_search"][start:stop] = np.asarray(
            [row.full_search for row in rows], dtype=np.bool_
        )
        has_wdl = np.asarray([row.value_wdl is not None for row in rows], dtype=np.bool_)
        self._arrays["has_value_wdl"][start:stop] = has_wdl
        wdl_rows = np.zeros((len(rows), 3), dtype=np.float32)
        for index, row in enumerate(rows):
            if row.value_wdl is not None:
                wdl_rows[index] = np.asarray(row.value_wdl, dtype=np.float32)
        self._arrays["value_wdl"][start:stop] = wdl_rows
        if "aux_value_targets" in self._arrays:
            aux_shape = self._shapes["aux_value_targets"]
            aux_rows = np.zeros((len(rows), *aux_shape), dtype=np.float32)
            has_aux = np.zeros(len(rows), dtype=np.bool_)
            for index, row in enumerate(rows):
                if row.aux_value_targets is not None:
                    aux_rows[index] = np.asarray(row.aux_value_targets, dtype=np.float32)
                    has_aux[index] = True
            self._arrays["aux_value_targets"][start:stop] = aux_rows
            self._arrays["has_aux_value_targets"][start:stop] = has_aux

    @staticmethod
    def _as_array(value: np.ndarray, dtype: Any, name: str) -> np.ndarray:
        array = np.asarray(value)
        if array.dtype != np.dtype(dtype):
            raise ValueError(f"Replay {name} dtype must be {np.dtype(dtype)}")
        return array

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
