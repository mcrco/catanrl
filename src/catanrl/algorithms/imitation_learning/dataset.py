"""
Aggregated dataset for DAgger imitation learning with random eviction strategy.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


def _discount_rewards(rewards: List[float], gamma: float) -> List[float]:
    """Return discounted returns for one episode."""
    discounted: List[float] = [0.0] * len(rewards)
    running = 0.0
    for idx in reversed(range(len(rewards))):
        running = rewards[idx] + gamma * running
        discounted[idx] = running
    return discounted


class AggregatedDataset(Dataset):
    """In-memory dataset for aggregated DAgger trajectories with random eviction.
    
    When the dataset is full, new samples randomly replace existing samples
    rather than following a FIFO (ring buffer) approach. This maintains better
    diversity of game states across different games instead of only keeping
    the most recent experiences.
    """

    def __init__(
        self,
        actor_dim: int,
        critic_dim: int,
        max_size: int = 100_000_000,
    ):
        self.actor_dim = actor_dim
        self.critic_dim = critic_dim
        self.max_size = max_size
        self.capacity = max_size

        # Preallocate full capacity.
        self.actor_states = np.zeros((self.max_size, actor_dim), dtype=np.float32)
        self.critic_states = np.zeros((self.max_size, critic_dim), dtype=np.float32)
        self.actions = np.zeros((self.max_size,), dtype=np.int64)
        self.returns = np.zeros((self.max_size,), dtype=np.float32)
        self.is_single_action = np.zeros((self.max_size,), dtype=np.bool_)

        self.size = 0

    def __len__(self) -> int:
        return self.size

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if idx >= self.size:
            raise IndexError(f"Index {idx} out of bounds for size {self.size}")

        return (
            torch.from_numpy(self.actor_states[idx]),
            torch.from_numpy(self.critic_states[idx]),
            torch.tensor(self.actions[idx], dtype=torch.long),
            torch.tensor(self.returns[idx], dtype=torch.float32),
            torch.tensor(self.is_single_action[idx], dtype=torch.bool),
        )

    def add_episode(
        self,
        actor_states: np.ndarray | List[np.ndarray],
        critic_states: np.ndarray | List[np.ndarray],
        expert_actions: np.ndarray | List[int],
        rewards: List[float],
        gamma: float,
        is_single_action: np.ndarray | List[bool],
    ) -> None:
        """Add a full episode to the dataset using random eviction when full.
        
        When the dataset is at capacity, new samples randomly replace existing
        samples to maintain diversity across different games rather than just
        keeping the most recent experiences.
        """
        returns_list = _discount_rewards(rewards, gamma)
        returns = np.array(returns_list, dtype=np.float32)

        actor_states = np.asarray(actor_states, dtype=np.float32)
        critic_states = np.asarray(critic_states, dtype=np.float32)
        expert_actions = np.asarray(expert_actions, dtype=np.int64)
        is_single_action = np.asarray(is_single_action, dtype=np.bool_)

        num_samples = len(actor_states)
        if num_samples == 0:
            return

        available_space = self.capacity - self.size
        if num_samples <= available_space:
            start = self.size
            end = start + num_samples
            self.actor_states[start:end] = actor_states
            self.critic_states[start:end] = critic_states
            self.actions[start:end] = expert_actions
            self.returns[start:end] = returns
            self.is_single_action[start:end] = is_single_action
            self.size += num_samples
        else:
            # Fill remaining space first
            if available_space > 0:
                start = self.size
                end = start + available_space
                self.actor_states[start:end] = actor_states[:available_space]
                self.critic_states[start:end] = critic_states[:available_space]
                self.actions[start:end] = expert_actions[:available_space]
                self.returns[start:end] = returns[:available_space]
                self.is_single_action[start:end] = is_single_action[:available_space]
                self.size = self.capacity
            
            # Random eviction for remaining samples
            remaining = num_samples - available_space
            if remaining > 0:
                replace_indices = np.random.choice(
                    self.capacity, size=remaining, replace=False
                )
                offset = available_space
                
                self.actor_states[replace_indices] = actor_states[offset:]
                self.critic_states[replace_indices] = critic_states[offset:]
                self.actions[replace_indices] = expert_actions[offset:]
                self.returns[replace_indices] = returns[offset:]
                self.is_single_action[replace_indices] = is_single_action[offset:]

    def _add_sample(self, *args):
        raise NotImplementedError("Use add_episode with batched data instead.")


__all__ = ["AggregatedDataset", "_discount_rewards"]
