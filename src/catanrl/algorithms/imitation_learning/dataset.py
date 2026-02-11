"""
Aggregated dataset for DAgger imitation learning with configurable eviction strategy.
"""

from __future__ import annotations

from enum import Enum
from typing import List, Literal, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from catanrl.features.catanatron_utils import get_actor_indices_from_critic


class EvictionStrategy(str, Enum):
    """Strategy for evicting samples when dataset is full."""
    RANDOM = "random"  # Randomly replace existing samples
    FIFO = "fifo"      # Ring buffer - evict oldest samples first
    CORRECT = "correct"  # Evict samples the model predicts correctly (not yet implemented)


class AggregatedDataset(Dataset):
    """In-memory dataset for aggregated DAgger trajectories with configurable eviction.
    
    Supports multiple eviction strategies when the dataset is full:
    - RANDOM: Randomly replace existing samples
    - FIFO: Ring buffer approach, evict oldest samples first
    - CORRECT: Evict samples the model predicts correctly (not yet implemented)
    """

    def __init__(
        self,
        critic_dim: int,
        num_players: int,
        map_type: Literal["BASE", "MINI", "TOURNAMENT"],
        max_size: int = 100_000_000,
        eviction_strategy: EvictionStrategy = EvictionStrategy.RANDOM,
    ):
        self.critic_dim = critic_dim
        self.num_players = num_players
        self.map_type = map_type
        self.max_size = max_size
        self.capacity = max_size
        self.eviction_strategy = eviction_strategy

        # Actor state is subset of critic state; these indices convert critic -> actor 
        # so we don't double store overlapping board state.
        self.actor_indices = get_actor_indices_from_critic(num_players, map_type)

        # Preallocate full capacity - only store critic states
        self.critic_states = np.zeros((self.max_size, critic_dim), dtype=np.float32)
        self.actions = np.zeros((self.max_size,), dtype=np.int64)
        self.returns = np.zeros((self.max_size,), dtype=np.float32)
        self.is_single_action = np.zeros((self.max_size,), dtype=np.bool_)

        self.size = 0
        self.head = 0  # Write pointer for FIFO eviction

    def __len__(self) -> int:
        return self.size

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if idx >= self.size:
            raise IndexError(f"Index {idx} out of bounds for size {self.size}")

        critic = self.critic_states[idx]
        # derive actor state on demand
        actor = critic[self.actor_indices]

        return (
            torch.from_numpy(actor.copy()),
            torch.from_numpy(critic),
            torch.tensor(self.actions[idx], dtype=torch.long),
            torch.tensor(self.returns[idx], dtype=torch.float32),
            torch.tensor(self.is_single_action[idx], dtype=torch.bool),
        )

    def add_samples(
        self,
        critic_states: np.ndarray | List[np.ndarray],
        expert_actions: np.ndarray | List[int],
        returns: np.ndarray | List[float],
        is_single_action: np.ndarray | List[bool],
    ) -> None:
        """Add precomputed transition samples using the configured eviction strategy."""
        if self.eviction_strategy == EvictionStrategy.CORRECT:
            raise NotImplementedError(
                "CORRECT eviction strategy requires model access and is not yet implemented"
            )

        critic_states = np.asarray(critic_states, dtype=np.float32)
        expert_actions = np.asarray(expert_actions, dtype=np.int64)
        returns = np.asarray(returns, dtype=np.float32)
        is_single_action = np.asarray(is_single_action, dtype=np.bool_)

        self._add_samples(
            critic_states=critic_states,
            expert_actions=expert_actions,
            returns=returns,
            is_single_action=is_single_action,
        )

    def _add_samples(
        self,
        critic_states: np.ndarray,
        expert_actions: np.ndarray,
        returns: np.ndarray,
        is_single_action: np.ndarray,
    ) -> None:
        num_samples = len(critic_states)
        if num_samples == 0:
            return
        if (
            len(expert_actions) != num_samples
            or len(returns) != num_samples
            or len(is_single_action) != num_samples
        ):
            raise ValueError("Mismatched sample counts in add_samples inputs.")

        available_space = self.capacity - self.size
        if num_samples <= available_space:
            start = self.size
            end = start + num_samples
            self.critic_states[start:end] = critic_states
            self.actions[start:end] = expert_actions
            self.returns[start:end] = returns
            self.is_single_action[start:end] = is_single_action
            self.size += num_samples
            self.head = self.size % self.capacity  # Update head for when we switch to eviction
            return

        # Fill remaining space first
        if available_space > 0:
            start = self.size
            end = start + available_space
            self.critic_states[start:end] = critic_states[:available_space]
            self.actions[start:end] = expert_actions[:available_space]
            self.returns[start:end] = returns[:available_space]
            self.is_single_action[start:end] = is_single_action[:available_space]
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

        self.critic_states[replace_indices] = critic_states[offset:]
        self.actions[replace_indices] = expert_actions[offset:]
        self.returns[replace_indices] = returns[offset:]
        self.is_single_action[replace_indices] = is_single_action[offset:]


__all__ = ["AggregatedDataset", "EvictionStrategy"]
