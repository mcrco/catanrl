"""
Dataset Aggregation (DAgger) imitation-learning trainer.
"""

from __future__ import annotations

import os
import random
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import wandb
from pufferlib.emulation import nativize

from catanatron.gym.envs.catanatron_env import ACTION_SPACE_SIZE

from ...envs.gym.single_env import (
    compute_single_agent_dims,
    make_puffer_vectorized_envs,
)
from ...models.backbones import BackboneConfig, MLPBackboneConfig, CrossDimensionalBackboneConfig
from ...models.models import (
    build_flat_policy_network,
    build_hierarchical_policy_network,
    build_value_network,
)
from ...models.wrappers import PolicyNetworkWrapper, ValueNetworkWrapper
from ...eval.training_eval import evaluate_against_baselines


def _mask_logits(
    logits: torch.Tensor, valid_actions: np.ndarray, device: torch.device
) -> torch.Tensor:
    """Apply -inf mask to invalid actions."""
    mask = torch.full_like(logits, float("-inf"))
    if valid_actions.size == 0:
        return logits
    mask[:, valid_actions] = 0.0
    masked_logits = torch.clamp(logits + mask, min=-100, max=100)
    return masked_logits


def _observation_to_state_vector(observation: Dict[str, np.ndarray]) -> np.ndarray:
    """Flatten mixed observation (numeric + board tensor) into the training vector."""
    numeric = observation["numeric"].astype(np.float32, copy=False)
    board = observation["board"].astype(np.float32, copy=False)
    board_ch_last = np.transpose(board, (1, 2, 0))
    board_flat = board_ch_last.reshape(-1)
    return np.concatenate([numeric, board_flat], axis=0)


def _discount_rewards(rewards: List[float], gamma: float) -> List[float]:
    """Return discounted returns for one episode."""
    discounted: List[float] = [0.0] * len(rewards)
    running = 0.0
    for idx in reversed(range(len(rewards))):
        running = rewards[idx] + gamma * running
        discounted[idx] = running
    return discounted


def _extract_valid_actions(info: Dict[str, Any] | None) -> np.ndarray:
    valid_actions = None
    if info is not None:
        valid_actions = info.get("valid_actions")
    if valid_actions is None or len(valid_actions) == 0:
        return np.arange(ACTION_SPACE_SIZE, dtype=np.int64)
    return np.array(valid_actions, dtype=np.int64)


def _get_policy_logits(
    policy_model: PolicyNetworkWrapper,
    states: torch.Tensor,
    model_type: str,
) -> torch.Tensor:
    """Get flat action logits from policy model."""
    if model_type == "flat":
        return policy_model(states)
    elif model_type == "hierarchical":
        action_type_logits, param_logits = policy_model(states)
        return policy_model.get_flat_action_logits(action_type_logits, param_logits)
    else:
        raise ValueError(f"Unknown model_type '{model_type}'")


def _select_policy_action(
    policy_model: PolicyNetworkWrapper,
    state_vec: np.ndarray,
    valid_actions: np.ndarray,
    model_type: str,
    device: torch.device,
    deterministic: bool = False,
) -> int:
    """Sample or pick the highest-probability action under the current policy."""
    state_t = torch.from_numpy(state_vec).float().unsqueeze(0).to(device)
    was_training = policy_model.training
    policy_model.eval()

    with torch.no_grad():
        policy_logits = _get_policy_logits(policy_model, state_t, model_type)
        masked_logits = _mask_logits(policy_logits, valid_actions, device)
        valid_logits = masked_logits[:, valid_actions]
        if torch.isnan(valid_logits).any():
            action_idx = random.choice(valid_actions.tolist())
        else:
            if deterministic:
                local_idx = torch.argmax(valid_logits, dim=-1).item()
            else:
                probs = F.softmax(valid_logits, dim=-1)
                probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-12)
                dist = torch.distributions.Categorical(probs=probs.squeeze(0))
                local_idx = dist.sample().item()
            action_idx = int(valid_actions[local_idx])

    if was_training:
        policy_model.train()
    return action_idx


def _select_policy_actions_batch(
    policy_model: PolicyNetworkWrapper,
    states: torch.Tensor,
    valid_action_masks: np.ndarray,
    model_type: str,
    device: torch.device,
    deterministic: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorized action selection for multiple parallel environments.

    Returns:
        actions: Selected action indices (batch_size,)
        log_probs: Log probabilities of selected actions (batch_size,)
    """
    was_training = policy_model.training
    policy_model.eval()

    with torch.no_grad():
        policy_logits = _get_policy_logits(policy_model, states, model_type)

        # Mask invalid actions
        mask_tensor = torch.as_tensor(
            valid_action_masks, dtype=torch.bool, device=device
        )
        if mask_tensor.ndim == 1:
            mask_tensor = mask_tensor.unsqueeze(0)
        # Handle case where no actions are valid (shouldn't happen)
        no_valid = ~mask_tensor.any(dim=1, keepdim=True)
        mask_tensor = torch.where(no_valid, torch.ones_like(mask_tensor), mask_tensor)

        masked_logits = torch.where(
            mask_tensor,
            policy_logits,
            torch.full_like(policy_logits, float("-inf")),
        )
        masked_logits = torch.clamp(masked_logits, min=-100, max=100)

        probs = torch.softmax(masked_logits, dim=-1)
        log_probs_all = torch.log_softmax(masked_logits, dim=-1)

        if deterministic:
            action_tensor = torch.argmax(masked_logits, dim=-1)
        else:
            dist = torch.distributions.Categorical(probs=probs)
            action_tensor = dist.sample()

        log_prob_tensor = log_probs_all.gather(1, action_tensor.unsqueeze(1)).squeeze(1)

        actions = action_tensor.detach().cpu().numpy().astype(np.int64)
        log_probs = log_prob_tensor.detach().cpu().numpy().astype(np.float32)

    if was_training:
        policy_model.train()

    return actions, log_probs


class AggregatedDataset(Dataset):
    """In-memory dataset for aggregated DAgger trajectories with separate actor/critic states."""

    def __init__(
        self,
        actor_dim: int,
        critic_dim: int,
        max_size: int = 100_000_000,
    ):
        self.actor_dim = actor_dim
        self.critic_dim = critic_dim
        # Handle case where None is passed explicitly
        if max_size is None:
            max_size = 100_000_000
        self.max_size = max_size
        self.capacity = max_size

        # Preallocate full capacity.
        # NOTE: With max_size=100M, this will attempt to allocate ~1.5TB+ of RAM depending on dims.
        self.actor_states = np.zeros((self.max_size, actor_dim), dtype=np.float32)
        self.critic_states = np.zeros((self.max_size, critic_dim), dtype=np.float32)
        self.actions = np.zeros((self.max_size,), dtype=np.int64)
        self.returns = np.zeros((self.max_size,), dtype=np.float32)
        self.is_single_action = np.zeros((self.max_size,), dtype=np.bool_)

        self.pos = 0
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
        """Add a full episode to the dataset."""
        # Convert rewards to returns
        returns_list = _discount_rewards(rewards, gamma)
        returns = np.array(returns_list, dtype=np.float32)

        # Ensure inputs are arrays
        actor_states = np.asarray(actor_states, dtype=np.float32)
        critic_states = np.asarray(critic_states, dtype=np.float32)
        expert_actions = np.asarray(expert_actions, dtype=np.int64)
        is_single_action = np.asarray(is_single_action, dtype=np.bool_)

        num_samples = len(actor_states)
        if num_samples == 0:
            return

        # Ring buffer logic: write data, wrapping around if necessary
        remaining = num_samples
        src_idx = 0
        while remaining > 0:
            space = self.capacity - self.pos
            to_write = min(remaining, space)

            start = self.pos
            end = start + to_write

            self.actor_states[start:end] = actor_states[src_idx : src_idx + to_write]
            self.critic_states[start:end] = critic_states[src_idx : src_idx + to_write]
            self.actions[start:end] = expert_actions[src_idx : src_idx + to_write]
            self.returns[start:end] = returns[src_idx : src_idx + to_write]
            self.is_single_action[start:end] = is_single_action[
                src_idx : src_idx + to_write
            ]

            self.pos = (self.pos + to_write) % self.capacity
            # size grows up to capacity
            self.size = min(self.size + to_write, self.capacity)

            remaining -= to_write
            src_idx += to_write

    def _add_sample(self, *args):
        raise NotImplementedError("Use add_episode with batched data instead.")


@dataclass
class DAggerCollectStats:
    episodes: int
    steps: int
    mean_reward: float
    expert_fraction: float
    dataset_size: int


def _flatten_puffer_observation(obs: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """Convert Puffer observation dict to flattened actor and critic vectors."""
    actor_obs = obs["observation"]
    numeric = np.asarray(actor_obs["numeric"], dtype=np.float32).reshape(-1)
    board = np.asarray(actor_obs["board"], dtype=np.float32)
    # board is (C, W, H), transpose to (W, H, C) then flatten
    board_wh_last = np.transpose(board, (1, 2, 0))
    board_flat = board_wh_last.reshape(-1)
    actor_vec = np.concatenate([numeric, board_flat], axis=0)
    critic_vec = np.asarray(obs["critic"], dtype=np.float32).reshape(-1)
    return actor_vec, critic_vec


def _get_action_mask_from_obs(obs: Dict[str, Any]) -> np.ndarray:
    """Extract action mask from Puffer observation."""
    return np.asarray(obs["action_mask"], dtype=np.int8) > 0


def _extract_expert_actions_from_infos(infos: Any, batch_size: int) -> np.ndarray:
    """Extract expert actions from PufferLib vectorized env infos.

    The infos from puffer vectorized envs can be structured as a dict of arrays
    or as a list of dicts. This handles both cases.
    """
    expert_actions = np.zeros(batch_size, dtype=np.int64)

    if isinstance(infos, dict) and "expert_action" in infos:
        # Dict of arrays format
        expert_arr = infos["expert_action"]
        if hasattr(expert_arr, "__len__") and len(expert_arr) == batch_size:
            expert_actions[:] = expert_arr
        else:
            expert_actions[:] = int(expert_arr)
    elif isinstance(infos, (list, tuple)):
        # List of dicts format
        for idx, info in enumerate(infos):
            if isinstance(info, dict) and "expert_action" in info:
                expert_actions[idx] = int(info["expert_action"])
    elif hasattr(infos, "__iter__"):
        # Try to iterate over infos
        for idx, info in enumerate(infos):
            if idx >= batch_size:
                break
            if isinstance(info, dict) and "expert_action" in info:
                expert_actions[idx] = int(info["expert_action"])

    return expert_actions


def _collect_dagger_rollouts_vectorized(
    envs: Any,
    policy_model: PolicyNetworkWrapper,
    dataset: AggregatedDataset,
    num_steps: int,
    beta: float,
    gamma: float,
    model_type: str,
    device: torch.device,
    actor_dim: int,
    critic_dim: int,
) -> DAggerCollectStats:
    """Run batched policy/environment interaction and aggregate expert-labeled data.

    This version works with PufferLib vectorized environments where expert actions
    are provided in the info dict (via expert_player config on the env).
    """
    num_envs = envs.num_envs
    driver_env = envs.driver_env
    # For GymnasiumPufferEnv, get observation space from the wrapped env
    if hasattr(driver_env, "env_single_observation_space"):
        obs_space = driver_env.env_single_observation_space
    else:
        obs_space = driver_env.observation_space
    if hasattr(driver_env, "obs_dtype"):
        obs_dtype = driver_env.obs_dtype
    else:
        obs_dtype = np.float32

    # Preallocated buffers for each environment
    # Each buffer is resized if needed, but we start with a reasonable size
    initial_buffer_size = 2000  # TURNS_LIMIT=1000, ~2 steps per turn max

    # We maintain a list of buffers (dictionaries of arrays)
    ep_buffers = []
    for _ in range(num_envs):
        ep_buffers.append(
            {
                "actor": np.zeros((initial_buffer_size, actor_dim), dtype=np.float32),
                "critic": np.zeros((initial_buffer_size, critic_dim), dtype=np.float32),
                "label": np.zeros(initial_buffer_size, dtype=np.int64),
                "reward": np.zeros(initial_buffer_size, dtype=np.float32),
                "is_single": np.zeros(initial_buffer_size, dtype=np.bool_),
                "size": 0,
                "capacity": initial_buffer_size,
            }
        )

    episode_rewards: List[float] = []
    steps_collected = 0
    expert_actions_used = 0
    total_actions = 0

    observations, infos = envs.reset()

    for _ in range(num_steps):
        batch_size = observations.shape[0]
        actor_batch = np.zeros((batch_size, actor_dim), dtype=np.float32)
        critic_batch = np.zeros((batch_size, critic_dim), dtype=np.float32)
        action_masks = np.zeros((batch_size, ACTION_SPACE_SIZE), dtype=np.bool_)

        # Decode observations
        for idx in range(batch_size):
            structured = nativize(observations[idx], obs_space, obs_dtype)
            actor_batch[idx], critic_batch[idx] = _flatten_puffer_observation(structured)
            action_masks[idx] = _get_action_mask_from_obs(structured)

        # Get policy actions
        actor_tensor = torch.from_numpy(actor_batch).float().to(device)
        policy_actions, _ = _select_policy_actions_batch(
            policy_model,
            actor_tensor,
            action_masks,
            model_type,
            device,
            deterministic=False,
        )

        # Get expert actions from info dict (computed by the env)
        expert_actions = _extract_expert_actions_from_infos(infos, batch_size)

        # Determine which steps are single-action steps (only 1 valid action)
        num_valid_actions = action_masks.sum(axis=1)
        is_single_action_batch = num_valid_actions <= 1

        # Decide which action to execute (expert or policy)
        # Vectorized beta sampling
        use_expert = np.random.random(batch_size) < beta
        actions_to_play = np.where(use_expert, expert_actions, policy_actions)

        expert_actions_used += np.sum(use_expert)
        total_actions += batch_size

        # Store experience in buffers
        for idx in range(batch_size):
            buf = ep_buffers[idx]
            pos = buf["size"]

            # Resize if needed
            if pos >= buf["capacity"]:
                new_cap = buf["capacity"] * 2

                new_actor = np.zeros((new_cap, actor_dim), dtype=np.float32)
                new_actor[: buf["capacity"]] = buf["actor"]
                buf["actor"] = new_actor

                new_critic = np.zeros((new_cap, critic_dim), dtype=np.float32)
                new_critic[: buf["capacity"]] = buf["critic"]
                buf["critic"] = new_critic

                new_label = np.zeros(new_cap, dtype=np.int64)
                new_label[: buf["capacity"]] = buf["label"]
                buf["label"] = new_label

                new_reward = np.zeros(new_cap, dtype=np.float32)
                new_reward[: buf["capacity"]] = buf["reward"]
                buf["reward"] = new_reward

                new_is_single = np.zeros(new_cap, dtype=np.bool_)
                new_is_single[: buf["capacity"]] = buf["is_single"]
                buf["is_single"] = new_is_single

                buf["capacity"] = new_cap

            # Write to buffer
            buf["actor"][pos] = actor_batch[idx]
            buf["critic"][pos] = critic_batch[idx]
            buf["label"][pos] = expert_actions[idx]
            buf["is_single"][pos] = is_single_action_batch[idx]
            buf["size"] += 1

        # Step environments
        next_observations, rewards, terminations, truncations, infos = envs.step(
            actions_to_play
        )
        rewards = rewards.astype(np.float32)
        dones = np.logical_or(terminations, truncations)

        for idx in range(batch_size):
            # Assign reward to the previous step (which is at size-1)
            buf = ep_buffers[idx]
            pos = buf["size"] - 1
            buf["reward"][pos] = rewards[idx]

            steps_collected += 1

            if dones[idx]:
                # Episode finished - add to dataset
                size = buf["size"]
                dataset.add_episode(
                    actor_states=buf["actor"][:size],
                    critic_states=buf["critic"][:size],
                    expert_actions=buf["label"][:size],
                    rewards=buf["reward"][:size],
                    gamma=gamma,
                    is_single_action=buf["is_single"][:size],
                )
                episode_rewards.append(float(np.sum(buf["reward"][:size])))

                # Reset buffer
                buf["size"] = 0

        observations = next_observations

    mean_reward = float(np.mean(episode_rewards)) if episode_rewards else 0.0
    expert_fraction = expert_actions_used / total_actions if total_actions else 0.0
    return DAggerCollectStats(
        episodes=len(episode_rewards),
        steps=steps_collected,
        mean_reward=mean_reward,
        expert_fraction=expert_fraction,
        dataset_size=len(dataset),
    )


def _train_on_dataset(
    dataset: AggregatedDataset,
    policy_model: PolicyNetworkWrapper,
    critic_model: ValueNetworkWrapper,
    policy_optimizer: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer,
    epochs: int,
    batch_size: int,
    model_type: str,
    device: torch.device,
    max_grad_norm: float = 5.0,
) -> Dict[str, float]:
    """Run supervised updates on the aggregated dataset with separate policy/critic."""
    if len(dataset) == 0 or epochs <= 0:
        return {"total_loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0, "accuracy": 0.0}

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    policy_model.train()
    critic_model.train()

    total_loss = 0.0
    total_policy_loss = 0.0
    total_value_loss = 0.0
    correct = 0
    total_samples = 0
    n_updates = 0

    for _ in range(epochs):
        for actor_features, critic_features, actions, returns, is_single_action in loader:
            actor_features = actor_features.to(device)
            critic_features = critic_features.to(device)
            actions = actions.to(device)
            returns = returns.to(device)
            is_single_action = is_single_action.to(device)

            # Policy update (only on non-single-action steps since end-turn steps dominate otherwise)
            policy_optimizer.zero_grad()
            policy_logits = _get_policy_logits(policy_model, actor_features, model_type)
            non_single_mask = ~is_single_action
            if non_single_mask.any():
                # Compute loss only for non-single-action steps
                policy_loss_per_sample = F.cross_entropy(
                    policy_logits, actions, reduction="none"
                )
                policy_loss = policy_loss_per_sample[non_single_mask].mean()
                policy_loss.backward()
                torch.nn.utils.clip_grad_norm_(policy_model.parameters(), max_norm=max_grad_norm)
                policy_optimizer.step()
            else:
                # All samples are single-action steps, skip policy update
                policy_loss = torch.tensor(0.0, device=device)

            # Critic update
            critic_optimizer.zero_grad()
            value_pred = critic_model(critic_features).squeeze(-1)
            value_loss = F.mse_loss(value_pred, returns)
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(critic_model.parameters(), max_norm=max_grad_norm)
            critic_optimizer.step()

            # Metrics
            batch_total_loss = policy_loss + value_loss
            total_loss += float(batch_total_loss.item())
            total_policy_loss += float(policy_loss.item())
            total_value_loss += float(value_loss.item())

            predictions = torch.argmax(policy_logits, dim=-1)
            # Once again, only compute accuracy for non-single-action steps since end-turn steps dominate otherwise
            if non_single_mask.any():
                correct += int(((predictions == actions) & non_single_mask).sum().item())
                total_samples += int(non_single_mask.sum().item())
            n_updates += 1

    n_updates = max(n_updates, 1)
    accuracy = correct / total_samples if total_samples else 0.0
    return {
        "total_loss": total_loss / n_updates,
        "policy_loss": total_policy_loss / n_updates,
        "value_loss": total_value_loss / n_updates,
        "accuracy": accuracy,
    }


def train(
    num_actions: int = ACTION_SPACE_SIZE,
    model_type: str = "flat",
    backbone_type: str = "mlp",
    policy_hidden_dims: Sequence[int] = (512, 512),
    critic_hidden_dims: Sequence[int] = (512, 512),
    load_policy_weights: Optional[str] = None,
    load_critic_weights: Optional[str] = None,
    n_iterations: int = 10,
    steps_per_iteration: int = 2048,
    train_epochs: int = 1,
    batch_size: int = 256,
    policy_lr: float = 1e-4,
    critic_lr: float = 1e-4,
    gamma: float = 0.99,
    expert_config: str = "AB:2",
    opponent_configs: Optional[Sequence[str]] = None,
    map_type: Literal["BASE", "MINI", "TOURNAMENT"] = "BASE",
    beta_init: float = 1.0,
    beta_decay: float = 0.9,
    beta_min: float = 0.05,
    max_dataset_size: Optional[int] = None,
    save_path: Optional[str] = None,
    device: Optional[str] = None,
    wandb_config: Optional[Dict[str, Any]] = None,
    eval_games_per_opponent: int = 250,
    seed: int = 42,
    num_envs: int = 4,
    reward_function: Literal["shaped", "win"] = "shaped",
    max_grad_norm: float = 5.0,
) -> Tuple[PolicyNetworkWrapper, ValueNetworkWrapper]:
    """
    Train separate policy and critic networks with DAgger using vectorized environments.

    The policy network is trained via imitation learning (cross-entropy with expert labels).
    The critic network is trained to predict discounted returns using full game information.

    Args:
        num_actions: Size of action space
        model_type: "flat" or "hierarchical" policy head
        backbone_type: "mlp" or "xdim" (cross-dimensional with CNN for board)
        policy_hidden_dims: Hidden layer sizes for policy backbone
        critic_hidden_dims: Hidden layer sizes for critic backbone
        load_policy_weights: Path to load initial policy weights
        load_critic_weights: Path to load initial critic weights
        n_iterations: Number of DAgger iterations
        steps_per_iteration: Environment steps to collect per iteration
        train_epochs: Training epochs per iteration
        batch_size: Batch size for training
        policy_lr: Policy learning rate
        critic_lr: Critic learning rate
        gamma: Discount factor for computing returns
        expert_config: Expert player config string (e.g. "AB:2", "MCTS:200")
        opponent_configs: List of opponent config strings
        map_type: Catan map type
        beta_init: Initial probability of using expert action
        beta_decay: Multiplicative decay for beta per iteration
        beta_min: Minimum beta value
        max_dataset_size: Maximum samples to keep in replay buffer
        save_path: Directory to save checkpoints
        device: Torch device ("cuda" or "cpu")
        wandb_config: W&B initialization config
        eval_games_per_opponent: Number of games to play against each baseline opponent
        seed: Random seed
        num_envs: Number of parallel environments
        reward_function: Reward function type ("shaped" or "win")
        max_grad_norm: Maximum gradient norm for clipping

    Returns:
        Tuple of (policy_model, critic_model)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_device = torch.device(device)

    opponent_configs = list(opponent_configs) if opponent_configs else ["random"]
    save_path = save_path or "weights/dagger"

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if wandb_config:
        wandb.init(**wandb_config)
    else:
        wandb.init(mode="disabled")

    assert backbone_type in ("mlp", "xdim"), f"Unknown backbone_type '{backbone_type}'"

    # Compute dimensions
    num_players = len(opponent_configs) + 1
    dims = compute_single_agent_dims(num_players, map_type)
    actor_dim = dims["actor_dim"]
    critic_dim = dims["critic_dim"]
    numeric_dim = dims["numeric_dim"]
    full_numeric_dim = dims["full_numeric_dim"]
    board_channels = dims["board_channels"]

    # Board shape for xdim backbone (C, W, H)
    BOARD_WIDTH = 21
    BOARD_HEIGHT = 11

    print(f"\n{'=' * 60}")
    print("DAgger Training with Separate Policy/Critic Networks")
    print(f"{'=' * 60}")
    print(f"Device: {device}")
    print(f"Map type: {map_type} | Players: {num_players}")
    print(f"Backbone: {backbone_type} | Model type: {model_type}")
    print(f"Actor input dim: {actor_dim} | Critic input dim: {critic_dim}")
    if backbone_type == "xdim":
        print(f"Board shape (C, W, H): ({board_channels}, {BOARD_WIDTH}, {BOARD_HEIGHT}) | Numeric dim: {numeric_dim}")
    print(f"Parallel environments: {num_envs}")
    print(f"Steps per iteration: {steps_per_iteration}")

    # Build policy backbone config based on backbone_type
    if backbone_type == "mlp":
        policy_backbone_config = BackboneConfig(
            architecture="mlp",
            args=MLPBackboneConfig(input_dim=actor_dim, hidden_dims=list(policy_hidden_dims)),
        )
    else:  # xdim
        policy_backbone_config = BackboneConfig(
            architecture="cross_dimensional",
            args=CrossDimensionalBackboneConfig(
                board_height=BOARD_HEIGHT,
                board_width=BOARD_WIDTH,
                board_channels=board_channels,
                numeric_dim=numeric_dim,
                cnn_channels=[64, 128, 128],
                cnn_kernel_size=(3, 5),
                numeric_hidden_dims=list(policy_hidden_dims),
                fusion_hidden_dim=policy_hidden_dims[-1] if policy_hidden_dims else 256,
                output_dim=policy_hidden_dims[-1] if policy_hidden_dims else 256,
            ),
        )

    if model_type == "flat":
        policy_model = build_flat_policy_network(
            backbone_config=policy_backbone_config, num_actions=num_actions
        ).to(torch_device)
    elif model_type == "hierarchical":
        policy_model = build_hierarchical_policy_network(
            backbone_config=policy_backbone_config
        ).to(torch_device)
    else:
        raise ValueError(f"Unknown model_type '{model_type}'")

    # Build critic backbone config
    if backbone_type == "mlp":
        critic_backbone_config = BackboneConfig(
            architecture="mlp",
            args=MLPBackboneConfig(input_dim=critic_dim, hidden_dims=list(critic_hidden_dims)),
        )
    else:  # xdim
        critic_backbone_config = BackboneConfig(
            architecture="cross_dimensional",
            args=CrossDimensionalBackboneConfig(
                board_height=BOARD_HEIGHT,
                board_width=BOARD_WIDTH,
                board_channels=board_channels,
                numeric_dim=full_numeric_dim,  # critic uses full numeric features
                cnn_channels=[64, 128, 128],
                cnn_kernel_size=(3, 5),
                numeric_hidden_dims=list(critic_hidden_dims),
                fusion_hidden_dim=critic_hidden_dims[-1] if critic_hidden_dims else 256,
                output_dim=critic_hidden_dims[-1] if critic_hidden_dims else 256,
            ),
        )
    critic_model = build_value_network(backbone_config=critic_backbone_config).to(torch_device)

    # Load weights if provided
    if load_policy_weights and os.path.exists(load_policy_weights):
        state = torch.load(load_policy_weights, map_location=torch_device)
        policy_model.load_state_dict(state)
        print(f"[DAgger] Loaded policy weights from {load_policy_weights}")
    elif load_policy_weights:
        print(f"[DAgger] Warning: policy weights file '{load_policy_weights}' not found.")

    if load_critic_weights and os.path.exists(load_critic_weights):
        state = torch.load(load_critic_weights, map_location=torch_device)
        critic_model.load_state_dict(state)
        print(f"[DAgger] Loaded critic weights from {load_critic_weights}")
    elif load_critic_weights:
        print(f"[DAgger] Warning: critic weights file '{load_critic_weights}' not found.")

    policy_optimizer = torch.optim.Adam(policy_model.parameters(), lr=policy_lr)
    critic_optimizer = torch.optim.Adam(critic_model.parameters(), lr=critic_lr)

    # Create vectorized environments with expert player embedded
    # Each env will compute expert actions and include them in info dict
    envs = make_puffer_vectorized_envs(
        reward_function=reward_function,
        map_type=map_type,
        opponent_configs=list(opponent_configs),
        num_envs=num_envs,
        expert_config=expert_config,
    )

    dataset = AggregatedDataset(
        actor_dim=actor_dim, critic_dim=critic_dim, max_size=max_dataset_size
    )

    beta = beta_init
    best_eval_win_rate = float("-inf")
    total_steps = n_iterations * steps_per_iteration
    global_step = 0

    try:
        with tqdm(total=total_steps, desc="DAgger", unit="step", unit_scale=True) as pbar:
            for iteration in range(1, n_iterations + 1):
                pbar.set_postfix(
                    {"iter": f"{iteration}/{n_iterations}", "beta": f"{beta:.2f}"}
                )

                collect_stats = _collect_dagger_rollouts_vectorized(
                    envs=envs,
                    policy_model=policy_model,
                    dataset=dataset,
                    num_steps=steps_per_iteration,
                    beta=beta,
                    gamma=gamma,
                    model_type=model_type,
                    device=torch_device,
                    actor_dim=actor_dim,
                    critic_dim=critic_dim,
                )
                global_step += collect_stats.steps
                pbar.update(collect_stats.steps)

                train_stats = _train_on_dataset(
                    dataset=dataset,
                    policy_model=policy_model,
                    critic_model=critic_model,
                    policy_optimizer=policy_optimizer,
                    critic_optimizer=critic_optimizer,
                    epochs=train_epochs,
                    batch_size=batch_size,
                    model_type=model_type,
                    device=torch_device,
                    max_grad_norm=max_grad_norm,
                )

                # Evaluate against baselines (same as MARL PPO)
                policy_model.eval()
                with torch.no_grad():
                    eval_metrics = evaluate_against_baselines(
                        policy_model=policy_model,
                        model_type=model_type,
                        map_type=map_type,
                        num_games=eval_games_per_opponent,
                        seed=random.randint(0, sys.maxsize),
                        log_to_wandb=True,
                        global_step=global_step,
                    )
                policy_model.train()

                eval_win_rate = float(eval_metrics.get("eval/win_rate_vs_value", 0.0))
                pbar.set_postfix(
                    {
                        "iter": f"{iteration}/{n_iterations}",
                        "beta": f"{beta:.2f}",
                        "acc": f"{train_stats['accuracy'] * 100:.1f}%",
                        "wr_val": f"{eval_win_rate * 100:.1f}%",
                    }
                )

                if eval_win_rate > best_eval_win_rate:
                    best_eval_win_rate = eval_win_rate
                    os.makedirs(save_path, exist_ok=True)
                    policy_path = os.path.join(save_path, "policy_best.pt")
                    critic_path = os.path.join(save_path, "critic_best.pt")
                    torch.save(policy_model.state_dict(), policy_path)
                    torch.save(critic_model.state_dict(), critic_path)

                # Save periodic checkpoints
                os.makedirs(save_path, exist_ok=True)
                policy_path = os.path.join(save_path, f"policy_iter_{iteration}.pt")
                critic_path = os.path.join(save_path, f"critic_iter_{iteration}.pt")
                torch.save(policy_model.state_dict(), policy_path)
                torch.save(critic_model.state_dict(), critic_path)

                wandb.log(
                    {
                        "dagger/dataset_size": len(dataset),
                        "dagger/collect_reward": collect_stats.mean_reward,
                        "dagger/expert_fraction": collect_stats.expert_fraction,
                        "dagger/train_total_loss": train_stats["total_loss"],
                        "dagger/train_policy_loss": train_stats["policy_loss"],
                        "dagger/train_value_loss": train_stats["value_loss"],
                        "dagger/train_accuracy": train_stats["accuracy"],
                        "dagger/beta": beta,
                        **eval_metrics,
                    },
                    step=global_step,
                )

                beta = max(beta * beta_decay, beta_min)

    finally:
        envs.close()

    if wandb.run is not None:
        wandb.run.summary["best_eval_win_rate_vs_value"] = best_eval_win_rate

    print(
        f"\n[DAgger] Training complete. "
        f"Best eval win rate vs value: {best_eval_win_rate:.3f}"
    )
    return policy_model, critic_model


__all__ = ["train"]
