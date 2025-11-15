from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np


class OnPolicyBuffer:
    """Simple on-policy buffer for single-agent PPO with action masking."""

    def __init__(self):
        self.states: List[np.ndarray] = []
        self.actions: List[int] = []
        self.rewards: List[float] = []
        self.values: List[float] = []
        self.log_probs: List[float] = []
        self.valid_actions: List[np.ndarray] = []
        self.dones: List[bool] = []

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        value: float,
        log_prob: float,
        valid_actions: np.ndarray,
        done: bool,
    ):
        self.states.append(state.astype(np.float32, copy=False))
        self.actions.append(int(action))
        self.rewards.append(float(reward))
        self.values.append(float(value))
        self.log_probs.append(float(log_prob))
        self.valid_actions.append(valid_actions.astype(np.int64, copy=False))
        self.dones.append(bool(done))

    def get(self) -> Tuple[np.ndarray, ...]:
        return (
            np.array(self.states, dtype=np.float32),
            np.array(self.actions, dtype=np.int64),
            np.array(self.rewards, dtype=np.float32),
            np.array(self.values, dtype=np.float32),
            np.array(self.log_probs, dtype=np.float32),
            self.valid_actions,
            np.array(self.dones, dtype=np.bool_),
        )

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()
        self.valid_actions.clear()
        self.dones.clear()

    def __len__(self) -> int:
        return len(self.states)


class MultiAgentExperienceBuffer:
    """Buffer for PPO training across multiple PettingZoo agents."""

    def __init__(self):
        self.states: List[np.ndarray] = []
        self.actions: List[int] = []
        self.rewards: List[float] = []
        self.values: List[float] = []
        self.log_probs: List[float] = []
        self.valid_actions: List[np.ndarray] = []
        self.dones: List[bool] = []

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        value: float,
        log_prob: float,
        valid_actions: np.ndarray,
        done: bool,
    ):
        self.states.append(state.astype(np.float32, copy=False))
        self.actions.append(int(action))
        self.rewards.append(float(reward))
        self.values.append(float(value))
        self.log_probs.append(float(log_prob))
        self.valid_actions.append(valid_actions.astype(np.int64, copy=False))
        self.dones.append(bool(done))

    def get(self) -> Tuple[np.ndarray, ...]:
        return (
            np.stack(self.states).astype(np.float32),
            np.array(self.actions, dtype=np.int64),
            np.array(self.rewards, dtype=np.float32),
            np.array(self.values, dtype=np.float32),
            np.array(self.log_probs, dtype=np.float32),
            self.valid_actions,
            np.array(self.dones, dtype=np.bool_),
        )

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()
        self.valid_actions.clear()
        self.dones.clear()

    def __len__(self) -> int:
        return len(self.states)

