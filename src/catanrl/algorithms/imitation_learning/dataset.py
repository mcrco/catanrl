"""
Aggregated dataset for DAgger imitation learning with configurable eviction strategy.
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, Literal, Tuple

import numpy as np
import numpy.typing as npt
import torch
from torch.utils.data import Dataset

from catanrl.features.catanatron_utils import (
    ActorObservationLevel,
    CriticObservationLevel,
    get_actor_indices_from_full,
    get_observation_indices_from_full,
)
from catanrl.utils.catanatron_action_space import get_action_array, get_action_space_size


class EvictionStrategy(str, Enum):
    """Strategy for evicting samples when dataset is full."""
    RANDOM = "random"  # Randomly replace existing samples
    FIFO = "fifo"      # Ring buffer - evict oldest samples first
    CORRECT = "correct"  # Evict samples the model predicts correctly (not yet implemented)


class AggregatedDataset(Dataset):
    """In-memory (cpu) dataset for aggregated DAgger trajectories with configurable eviction."""

    def __init__(
        self,
        full_state_dim: int,
        num_players: int,
        map_type: Literal["BASE", "MINI", "TOURNAMENT"],
        actor_observation_level: ActorObservationLevel = "private",
        critic_observation_level: CriticObservationLevel = "full",
        max_size: int = 100_000_000,
        eviction_strategy: EvictionStrategy = EvictionStrategy.RANDOM,
    ):
        self.full_state_dim = full_state_dim
        self.num_players = num_players
        self.map_type = map_type
        self.actor_observation_level = actor_observation_level
        self.critic_observation_level = critic_observation_level
        self.max_size = max_size
        self.capacity = max_size
        self.eviction_strategy = eviction_strategy
        self.action_space_size = get_action_space_size(num_players, map_type)
        self.actor_observation_indices = get_actor_indices_from_full(
            num_players,
            map_type,
            level=actor_observation_level,
        )
        self.critic_observation_indices = get_observation_indices_from_full(
            num_players,
            map_type,
            level=critic_observation_level,
        )

        # Preallocate full capacity.
        self.full_states: npt.NDArray[np.float32] = np.zeros(
            (self.max_size, full_state_dim), dtype=np.float32
        )
        self.actions: npt.NDArray[np.int64] = np.zeros((self.max_size,), dtype=np.int64)
        self.returns: npt.NDArray[np.float32] = np.zeros((self.max_size,), dtype=np.float32)
        self.is_single_action: npt.NDArray[np.bool_] = np.zeros(
            (self.max_size,), dtype=np.bool_
        )
        self.action_masks: npt.NDArray[np.bool_] = np.zeros(
            (self.max_size, self.action_space_size), dtype=np.bool_
        )

        self.size = 0
        self.head = 0  # Write pointer for FIFO eviction

    def __len__(self) -> int:
        return self.size

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if idx >= self.size:
            raise IndexError(f"Index {idx} out of bounds for size {self.size}")

        full_state = self.full_states[idx]
        actor = full_state[self.actor_observation_indices]
        critic = full_state[self.critic_observation_indices]

        return (
            torch.from_numpy(actor.copy()),
            torch.from_numpy(critic.copy()),
            torch.tensor(np.take(self.actions, idx), dtype=torch.long),
            torch.tensor(np.take(self.returns, idx), dtype=torch.float32),
            torch.tensor(np.take(self.is_single_action, idx), dtype=torch.bool),
            torch.from_numpy(np.take(self.action_masks, idx, axis=0)),
        )

    def add_samples(
        self,
        full_states: npt.NDArray[np.float32] | list[npt.NDArray[np.float32]],
        expert_actions: npt.NDArray[np.int64] | list[int],
        returns: npt.NDArray[np.float32] | list[float],
        is_single_action: npt.NDArray[np.bool_] | list[bool],
        action_masks: npt.NDArray[np.bool_] | list[npt.NDArray[np.bool_]],
    ) -> None:
        """Add precomputed transition samples using the configured eviction strategy."""
        if self.eviction_strategy == EvictionStrategy.CORRECT:
            raise NotImplementedError(
                "CORRECT eviction strategy requires model access and is not yet implemented"
            )

        full_states = np.asarray(full_states, dtype=np.float32)
        expert_actions = np.asarray(expert_actions, dtype=np.int64)
        returns = np.asarray(returns, dtype=np.float32)
        is_single_action = np.asarray(is_single_action, dtype=np.bool_)
        action_masks = np.asarray(action_masks, dtype=np.bool_)
        num_samples = len(full_states)
        if num_samples == 0:
            return
        if (
            full_states.shape[1] != self.full_state_dim
            or len(expert_actions) != num_samples
            or len(returns) != num_samples
            or len(is_single_action) != num_samples
            or len(action_masks) != num_samples
        ):
            raise ValueError("Mismatched sample counts in add_samples inputs.")

        available_space = self.capacity - self.size
        if num_samples <= available_space:
            start = self.size
            end = start + num_samples
            self.full_states[start:end] = full_states
            self.actions[start:end] = expert_actions
            self.returns[start:end] = returns
            self.is_single_action[start:end] = is_single_action
            self.action_masks[start:end] = action_masks
            self.size += num_samples
            self.head = self.size % self.capacity  # Update head for when we switch to eviction
            return

        # Fill remaining space first
        if available_space > 0:
            start = self.size
            end = start + available_space
            self.full_states[start:end] = full_states[:available_space]
            self.actions[start:end] = expert_actions[:available_space]
            self.returns[start:end] = returns[:available_space]
            self.is_single_action[start:end] = is_single_action[:available_space]
            self.action_masks[start:end] = action_masks[:available_space]
            self.size = self.capacity

        # Evict and replace remaining samples
        remaining = num_samples - available_space
        if remaining <= 0:
            return

        offset = available_space
        if self.eviction_strategy == EvictionStrategy.RANDOM:
            replace_indices = np.random.choice(self.capacity, size=remaining, replace=False)
        elif self.eviction_strategy == EvictionStrategy.FIFO:
            # Ring buffer: write at head and advance
            replace_indices = np.arange(self.head, self.head + remaining) % self.capacity
            self.head = (self.head + remaining) % self.capacity
        else:
            raise ValueError(f"Unknown eviction strategy: {self.eviction_strategy}")

        self.full_states[replace_indices] = full_states[offset:]
        self.actions[replace_indices] = expert_actions[offset:]
        self.returns[replace_indices] = returns[offset:]
        self.is_single_action[replace_indices] = is_single_action[offset:]
        self.action_masks[replace_indices] = action_masks[offset:]

    def summarize_imitation_stats(self) -> Tuple[float, float, Dict[str, int]]:
        """Fraction of non-forced steps, mean valid-action count, expert action-type counts."""
        if self.size == 0:
            return 0.0, 0.0, {}
        sl = slice(0, self.size)
        is_single = self.is_single_action[sl]
        masks = self.action_masks[sl]
        actions = self.actions[sl]
        fraction_non_single = float((~is_single).mean())
        mean_mask_size = float(masks.sum(axis=1).mean())

        action_array = get_action_array(self.num_players, self.map_type)
        counts: Dict[str, int] = {}
        for a in actions:
            at = action_array[int(a)][0]
            name = at.name
            counts[name] = counts.get(name, 0) + 1
        return fraction_non_single, mean_mask_size, counts


__all__ = ["AggregatedDataset", "EvictionStrategy"]
