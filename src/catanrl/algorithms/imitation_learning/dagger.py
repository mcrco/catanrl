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
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
from sklearn.metrics import f1_score
from pufferlib.emulation import nativize

from catanatron.gym.envs.catanatron_env import ACTION_SPACE_SIZE

from .dataset import AggregatedDataset, EvictionStrategy

from ...envs.gym.single_env import (
    compute_single_agent_dims,
    make_puffer_vectorized_envs,
)
from ...envs.gym.puffer_rollout_utils import (
    extract_expert_actions_from_infos,
    flatten_puffer_observation,
    get_action_mask_from_obs,
)
from ...models.backbones import BackboneConfig, MLPBackboneConfig, CrossDimensionalBackboneConfig
from ...models.models import (
    build_flat_policy_network,
    build_hierarchical_policy_network,
    build_value_network,
)
from ...models.wrappers import PolicyNetworkWrapper, ValueNetworkWrapper
from ...eval.training_eval import eval_policy_value_against_baselines


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
        mask_tensor = torch.as_tensor(valid_action_masks, dtype=torch.bool, device=device)
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


@dataclass
class DAggerCollectStats:
    episodes: int
    steps: int
    mean_reward: float
    expert_fraction: float
    dataset_size: int


def _collect_dagger_rollouts_vectorized(
    envs: Any,
    policy_model: PolicyNetworkWrapper,
    critic_model: ValueNetworkWrapper,
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

    # Preallocated buffers for the full collection phase (time x env)
    critic_states_tmj = np.zeros((num_steps, num_envs, critic_dim), dtype=np.float32)
    expert_actions_tmj = np.zeros((num_steps, num_envs), dtype=np.int64)
    rewards_tmj = np.zeros((num_steps, num_envs), dtype=np.float32)
    is_single_action_tmj = np.zeros((num_steps, num_envs), dtype=np.bool_)
    dones_tmj = np.zeros((num_steps, num_envs), dtype=np.bool_)

    episode_rewards: List[float] = []
    episode_reward_accum = np.zeros(num_envs, dtype=np.float32)
    steps_collected = 0
    expert_actions_used = 0
    total_actions = 0

    observations, infos = envs.reset()

    for step_idx in range(num_steps):
        batch_size = observations.shape[0]
        actor_batch = np.zeros((batch_size, actor_dim), dtype=np.float32)
        critic_batch = np.zeros((batch_size, critic_dim), dtype=np.float32)
        action_masks = np.zeros((batch_size, ACTION_SPACE_SIZE), dtype=np.bool_)

        # Decode observations
        for idx in range(batch_size):
            structured = nativize(observations[idx], obs_space, obs_dtype)
            actor_batch[idx], critic_batch[idx] = flatten_puffer_observation(structured)
            action_masks[idx] = get_action_mask_from_obs(structured)

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
        expert_actions = extract_expert_actions_from_infos(infos, batch_size)

        # Determine which steps are single-action steps (only 1 valid action)
        num_valid_actions = action_masks.sum(axis=1)
        is_single_action_batch = num_valid_actions <= 1

        # Decide which action to execute (expert or policy)
        # Vectorized beta sampling
        use_expert = np.random.random(batch_size) < beta
        actions_to_play = np.where(use_expert, expert_actions, policy_actions)

        expert_actions_used += np.sum(use_expert)
        total_actions += batch_size

        # Store experience in time-major buffers
        critic_states_tmj[step_idx] = critic_batch
        expert_actions_tmj[step_idx] = expert_actions
        is_single_action_tmj[step_idx] = is_single_action_batch

        # Step environments
        next_observations, rewards, terminations, truncations, infos = envs.step(actions_to_play)
        rewards = rewards.astype(np.float32)
        dones = np.logical_or(terminations, truncations)

        rewards_tmj[step_idx] = rewards
        dones_tmj[step_idx] = dones

        episode_reward_accum += rewards
        if np.any(dones):
            episode_rewards.extend(episode_reward_accum[dones].tolist())
            episode_reward_accum[dones] = 0.0

        steps_collected += batch_size

        observations = next_observations

    # Bootstrap returns for unfinished episodes using critic value
    was_training = critic_model.training
    critic_model.eval()
    last_critic_states = np.zeros((num_envs, critic_dim), dtype=np.float32)
    for idx in range(num_envs):
        structured = nativize(observations[idx], obs_space, obs_dtype)
        _, last_critic_states[idx] = flatten_puffer_observation(structured)

    with torch.no_grad():
        last_critic_tensor = torch.from_numpy(last_critic_states).float().to(device)
        bootstrap_values = critic_model(last_critic_tensor).squeeze(-1).cpu().numpy()
    bootstrap_values = bootstrap_values.astype(np.float32, copy=False)
    bootstrap_values = np.where(dones_tmj[-1], 0.0, bootstrap_values)
    if was_training:
        critic_model.train()

    # Compute discounted returns (time-major), resetting at episode boundaries
    returns_tmj = np.zeros_like(rewards_tmj)
    running_returns = bootstrap_values.copy()
    for step_idx in reversed(range(num_steps)):
        running_returns = rewards_tmj[step_idx] + gamma * running_returns
        returns_tmj[step_idx] = running_returns
        running_returns = np.where(dones_tmj[step_idx], 0.0, running_returns)

    dataset.add_samples(
        critic_states=critic_states_tmj.reshape(-1, critic_dim),
        expert_actions=expert_actions_tmj.reshape(-1),
        returns=returns_tmj.reshape(-1),
        is_single_action=is_single_action_tmj.reshape(-1),
    )

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
        return {
            "total_loss": 0.0,
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "accuracy": 0.0,
            "f1_score": 0.0,
        }

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    policy_model.train()
    critic_model.train()

    for _ in range(epochs):
        # Reset all metric counters each epoch (only report last epoch)
        total_loss = 0.0
        total_policy_loss = 0.0
        total_value_loss = 0.0
        correct = 0
        total_samples = 0
        all_labels = []
        all_preds = []
        n_updates = 0
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
                policy_loss_per_sample = F.cross_entropy(policy_logits, actions, reduction="none")
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
                # Collect labels and predictions for F1 calculation
                all_labels.extend(actions[non_single_mask].cpu().numpy().tolist())
                all_preds.extend(predictions[non_single_mask].cpu().numpy().tolist())
            n_updates += 1

    n_updates = max(n_updates, 1)
    accuracy = correct / total_samples if total_samples else 0.0
    # Compute macro F1 score across all action classes
    if all_labels and all_preds:
        f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0.0)
    else:
        f1 = 0.0
    return {
        "total_loss": total_loss / n_updates,
        "policy_loss": total_policy_loss / n_updates,
        "value_loss": total_value_loss / n_updates,
        "accuracy": accuracy,
        "f1_score": f1,
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
    eviction_strategy: EvictionStrategy = EvictionStrategy.RANDOM,
    save_path: Optional[str] = None,
    device: Optional[str] = None,
    wandb_config: Optional[Dict[str, Any]] = None,
    eval_games_per_opponent: int = 250,
    eval_compare_to_expert: bool = True,
    eval_expert_config: Optional[str] = None,
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
        eviction_strategy: Strategy for evicting samples when buffer is full
        save_path: Directory to save checkpoints
        device: Torch device ("cuda" or "cpu")
        wandb_config: W&B initialization config
        eval_games_per_opponent: Number of games to play against each baseline opponent
        eval_compare_to_expert: Whether to compute expert-accuracy/F1 during eval
        eval_expert_config: Expert config to use for eval labels (defaults to expert_config)
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
        print(
            f"Board shape (C, W, H): ({board_channels}, {BOARD_WIDTH}, {BOARD_HEIGHT}) | Numeric dim: {numeric_dim}"
        )
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
        policy_model = build_hierarchical_policy_network(backbone_config=policy_backbone_config).to(
            torch_device
        )
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

    if max_dataset_size is None:
        max_dataset_size = 100_000_000

    dataset = AggregatedDataset(
        critic_dim=critic_dim,
        num_players=num_players,
        map_type=map_type,
        max_size=max_dataset_size,
        eviction_strategy=eviction_strategy,
    )

    beta = beta_init
    best_eval_win_rate = float("-inf")
    best_eval_critic_mse = float("inf")
    total_steps = n_iterations * num_envs * steps_per_iteration
    global_step = 0

    try:
        with tqdm(total=total_steps, desc="DAgger", unit="step", unit_scale=True) as pbar:
            for iteration in range(1, n_iterations + 1):
                pbar.set_postfix({"iter": f"{iteration}/{n_iterations}", "beta": f"{beta:.2f}"})

                collect_stats = _collect_dagger_rollouts_vectorized(
                    envs=envs,
                    policy_model=policy_model,
                    critic_model=critic_model,
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

                # Evaluate policy against baselines and critic value predictions
                policy_model.eval()
                critic_model.eval()
                eval_expert_cfg = (
                    eval_expert_config if eval_expert_config is not None else expert_config
                )
                with torch.no_grad():
                    eval_metrics = eval_policy_value_against_baselines(
                        policy_model=policy_model,
                        critic_model=critic_model,
                        model_type=model_type,
                        map_type=map_type,
                        num_games=eval_games_per_opponent,
                        gamma=gamma,
                        seed=random.randint(0, sys.maxsize),
                        log_to_wandb=False,
                        global_step=global_step,
                        device=device,
                        compare_to_expert=eval_compare_to_expert,
                        expert_config=eval_expert_cfg,
                    )
                policy_model.train()
                critic_model.train()

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
                    torch.save(policy_model.state_dict(), policy_path)

                eval_critic_mse = eval_metrics.get("eval/value_mse", float("inf"))
                if eval_critic_mse < best_eval_critic_mse:
                    best_eval_critic_mse = eval_critic_mse
                    os.makedirs(save_path, exist_ok=True)
                    critic_path = os.path.join(save_path, "critic_best.pt")
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
                        "dagger/train_f1_score": train_stats["f1_score"],
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
        f"\n[DAgger] Training complete. " f"Best eval win rate vs value: {best_eval_win_rate:.3f}"
    )
    return policy_model, critic_model


__all__ = ["train"]
