from __future__ import annotations

from typing import Tuple

import numpy as np
import numpy.typing as npt

from catanatron.gym.envs.catanatron_env import ACTION_SPACE_SIZE


class ExperienceBuffer:
    """On-policy buffer that keeps a time x env dimension for vectorized envs."""

    def __init__(
        self,
        num_rollouts: int,
        state_dim: int,
        action_space_size: int = ACTION_SPACE_SIZE,
        num_envs: int = 1,
    ):
        self.capacity = num_rollouts
        self.num_envs = num_envs
        self.states = np.empty((num_rollouts, num_envs, state_dim), dtype=np.float32)
        self.actions = np.empty((num_rollouts, num_envs), dtype=np.int64)
        self.rewards = np.empty((num_rollouts, num_envs), dtype=np.float32)
        self.values = np.empty((num_rollouts, num_envs), dtype=np.float32)
        self.log_probs = np.empty((num_rollouts, num_envs), dtype=np.float32)
        self.valid_action_masks = np.empty((num_rollouts, num_envs, action_space_size), dtype=np.bool_)
        self.dones = np.empty((num_rollouts, num_envs), dtype=np.bool_)
        self.index = 0

    def _ensure_capacity(self, count: int):
        if self.index + count > self.capacity:
            raise RuntimeError(
                f"ExperienceBuffer capacity exceeded: {self.index + count}/{self.capacity}"
            )

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        value: float,
        log_prob: float,
        valid_action_masks: npt.NDArray[np.bool_],
        done: bool,
        env_id: int = 0,
    ):
        """Add a single transition (defaults to env 0)."""
        self._ensure_capacity(1)
        if env_id < 0 or env_id >= self.num_envs:
            raise ValueError(f"env_id {env_id} out of range [0, {self.num_envs})")
        idx = self.index
        self.states[idx, env_id] = state.astype(np.float32, copy=False)
        self.actions[idx, env_id] = int(action)
        self.rewards[idx, env_id] = float(reward)
        self.values[idx, env_id] = float(value)
        self.log_probs[idx, env_id] = float(log_prob)
        self.valid_action_masks[idx, env_id] = valid_action_masks.astype(np.bool_, copy=False)
        self.dones[idx, env_id] = bool(done)
        self.index += 1
        return idx

    def add_batch(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        values: np.ndarray,
        log_probs: np.ndarray,
        valid_action_masks: npt.NDArray[np.bool_],
        dones: np.ndarray,
    ):
        """Add a batch of transitions to the buffer."""
        states = np.asarray(states)
        num_envs = len(states)
        if num_envs != self.num_envs:
            raise ValueError(f"Expected {self.num_envs} envs, got {num_envs}")
        self._ensure_capacity(1)
        self.states[self.index] = states.astype(np.float32, copy=False)
        self.actions[self.index] = actions.astype(np.int64, copy=False)
        self.rewards[self.index] = rewards.astype(np.float32, copy=False)
        self.values[self.index] = values.astype(np.float32, copy=False)
        self.log_probs[self.index] = log_probs.astype(np.float32, copy=False)
        self.valid_action_masks[self.index] = valid_action_masks.astype(np.bool_, copy=False)
        self.dones[self.index] = dones.astype(np.bool_, copy=False)
        self.index += 1

    def get(self) -> Tuple[np.ndarray, ...]:
        """Return time-major arrays shaped [T, num_envs, ...]."""
        end = self.index
        return (
            np.array(self.states[:end], dtype=np.float32),
            np.array(self.actions[:end], dtype=np.int64),
            np.array(self.rewards[:end], dtype=np.float32),
            np.array(self.values[:end], dtype=np.float32),
            np.array(self.log_probs[:end], dtype=np.float32),
            np.array(self.valid_action_masks[:end], dtype=np.bool_),
            np.array(self.dones[:end], dtype=np.bool_),
        )

    def clear(self):
        self.index = 0

    def __len__(self) -> int:
        return self.index * self.num_envs


class CentralCriticExperienceBuffer(ExperienceBuffer):
    """Stores actor and critic observations for multi-agent PPO updates."""

    def __init__(
        self,
        num_rollouts: int,
        actor_state_dim: int,
        critic_state_dim: int,
        action_space_size: int = ACTION_SPACE_SIZE,
        num_envs: int = 1,
    ):
        super().__init__(num_rollouts, actor_state_dim, action_space_size, num_envs=num_envs)
        self.critic_states = np.empty((num_rollouts, num_envs, critic_state_dim), dtype=np.float32)

    def add(
        self,
        actor_state: np.ndarray,
        critic_state: np.ndarray,
        action: int,
        reward: float,
        value: float,
        log_prob: float,
        valid_action_masks: npt.NDArray[np.bool_],
        done: bool,
        env_id: int = 0,
    ):
        idx = super().add(
            actor_state, action, reward, value, log_prob, valid_action_masks, done, env_id=env_id
        )
        self.critic_states[idx, env_id] = critic_state.astype(np.float32, copy=False)

    def add_batch(
        self,
        actor_states: np.ndarray,
        critic_states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        values: np.ndarray,
        log_probs: np.ndarray,
        valid_action_masks: npt.NDArray[np.bool_],
        dones: np.ndarray,
    ):
        actor_states = np.asarray(actor_states)
        critic_states = np.asarray(critic_states)
        num_envs = len(actor_states)
        if num_envs != self.num_envs:
            raise ValueError(f"Expected {self.num_envs} envs, got {num_envs}")
        start = self.index
        super().add_batch(actor_states, actions, rewards, values, log_probs, valid_action_masks, dones)
        self.critic_states[start] = critic_states.astype(np.float32, copy=False)

    def get(self) -> Tuple[np.ndarray, ...]:
        end = self.index
        return (
            np.array(self.states[:end], dtype=np.float32),
            np.array(self.critic_states[:end], dtype=np.float32),
            np.array(self.actions[:end], dtype=np.int64),
            np.array(self.rewards[:end], dtype=np.float32),
            np.array(self.values[:end], dtype=np.float32),
            np.array(self.log_probs[:end], dtype=np.float32),
            np.array(self.valid_action_masks[:end], dtype=np.bool_),
            np.array(self.dones[:end], dtype=np.bool_),
        )
