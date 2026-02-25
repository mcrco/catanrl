from __future__ import annotations

import os
import sys
import random
from collections import deque
from typing import Dict, Optional, Sequence, Tuple, Literal, cast

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import wandb
from tqdm import tqdm

from catanatron.gym.envs.catanatron_env import ACTION_SPACE_SIZE, ACTIONS_ARRAY, ACTION_TYPES
from pufferlib.emulation import nativize

from ...envs import compute_multiagent_input_dim, flatten_marl_observation
from ...envs.zoo.multi_env import make_vectorized_envs as make_marl_vectorized_envs
from ...features.catanatron_utils import (
    get_full_numeric_feature_names,
    get_numeric_feature_names,
)
from ...eval.training_eval import eval_policy_value_against_baselines
from ...models.backbones import BackboneConfig, MLPBackboneConfig, CrossDimensionalBackboneConfig
from ...models.models import (
    build_flat_policy_network,
    build_hierarchical_policy_network,
    build_value_network,
)
from ...models.wrappers import PolicyNetworkWrapper, ValueNetworkWrapper
from .gae import compute_gae_batched
from .buffers import CentralCriticExperienceBuffer


def _build_action_type_mapping() -> Tuple[Dict[str, list], np.ndarray]:
    """Build a mapping from action types to action indices and vice versa.

    Returns:
        action_type_to_indices: Dict mapping action type name to list of action indices
        action_to_type_idx: Array mapping action index to action type index
    """
    action_type_to_indices: Dict[str, list] = {at.name: [] for at in ACTION_TYPES}
    action_to_type_idx = np.zeros(ACTION_SPACE_SIZE, dtype=np.int32)

    for action_idx, (action_type, _) in enumerate(ACTIONS_ARRAY):
        action_type_to_indices[action_type.name].append(action_idx)
        action_to_type_idx[action_idx] = ACTION_TYPES.index(action_type)

    return action_type_to_indices, cast(np.ndarray, action_to_type_idx)


# Pre-compute action type mapping at module load time
ACTION_TYPE_TO_INDICES, ACTION_TO_TYPE_IDX = _build_action_type_mapping()
ACTION_TYPE_NAMES = [at.name for at in ACTION_TYPES]


def compute_action_distributions(
    actions: np.ndarray,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Compute action type and action distributions from taken actions. Used for logging agent decisions over time."""
    n_actions = len(actions)
    if n_actions == 0:
        return {}, {}

    # Compute action type distribution
    action_types_taken = ACTION_TO_TYPE_IDX[actions]
    type_counts = np.bincount(action_types_taken, minlength=len(ACTION_TYPES))
    action_type_dist = {
        name: float(count / n_actions) for name, count in zip(ACTION_TYPE_NAMES, type_counts)
    }

    # Compute raw action distribution
    action_counts = np.bincount(actions, minlength=ACTION_SPACE_SIZE)
    action_dist = {
        f"action_{idx}": float(count / n_actions)
        for idx, count in enumerate(action_counts)
        if count > 0
    }

    return action_type_dist, action_dist


class PolicyAgent:
    """Policy wrapper that only handles action selection and log-prob evaluation."""

    def __init__(
        self,
        model: PolicyNetworkWrapper,
        model_type: str,
        device: str,
    ):
        self.model = model
        self.model_type = model_type
        self.device = device

    def _policy_logits(self, states: torch.Tensor) -> torch.Tensor:
        if self.model_type == "flat":
            policy_logits = self.model(states)
        elif self.model_type == "hierarchical":
            action_type_logits, param_logits = self.model(states)
            policy_logits = self.model.get_flat_action_logits(action_type_logits, param_logits)
        else:  # pragma: no cover - defensive guard
            raise ValueError(f"Unknown model_type '{self.model_type}'")
        return policy_logits

    def _mask_logits(
        self, policy_logits: torch.Tensor, valid_action_masks: np.ndarray
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mask_tensor = torch.as_tensor(
            valid_action_masks, dtype=torch.bool, device=policy_logits.device
        )
        if mask_tensor.ndim == 1:
            mask_tensor = mask_tensor.unsqueeze(0)
        no_valid = ~mask_tensor.any(dim=1, keepdim=True)
        mask_tensor = torch.where(no_valid, torch.ones_like(mask_tensor), mask_tensor)
        masked_logits = torch.where(
            mask_tensor,
            policy_logits,
            torch.full_like(policy_logits, float("-inf")),
        )
        return torch.clamp(masked_logits, min=-100, max=100), mask_tensor

    def select_action(
        self,
        state_vec: np.ndarray,
        valid_action_mask: np.ndarray,
        deterministic: bool = False,
    ) -> Tuple[int, float]:
        """Select an action for the current PettingZoo agent."""
        state_tensor = torch.from_numpy(state_vec).float().unsqueeze(0).to(self.device)
        self.model.eval()
        with torch.no_grad():
            policy_logits = self._policy_logits(state_tensor)
            masked_logits, mask_tensor = self._mask_logits(policy_logits, valid_action_mask)

            probs = torch.softmax(masked_logits, dim=-1)
            log_probs_all = torch.log_softmax(masked_logits, dim=-1)
            valid_indices = torch.nonzero(mask_tensor[0], as_tuple=False).squeeze(-1)
            if valid_indices.size == 0:
                valid_indices = torch.arange(ACTION_SPACE_SIZE, device=probs.device)

            if deterministic:
                local_idx = torch.argmax(probs[0, valid_indices]).item()
                action = int(valid_indices[local_idx].item())
            else:
                valid_probs = probs[0, valid_indices]
                valid_probs = valid_probs / (valid_probs.sum() + 1e-12)
                dist = torch.distributions.Categorical(valid_probs)
                action = int(valid_indices[dist.sample()].item())

            log_prob = float(log_probs_all[0, action].item())

        return action, log_prob

    def select_actions_batch(
        self,
        states: torch.Tensor,
        valid_action_masks: np.ndarray,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Vectorized action selection for multiple parallel environments."""
        self.model.eval()
        with torch.no_grad():
            policy_logits = self._policy_logits(states)

            # Track raw policy preference for logging purposes.
            raw_argmax_tensor = torch.argmax(policy_logits, dim=-1)

            masked_logits, _ = self._mask_logits(policy_logits, valid_action_masks)
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
            raw_argmax = raw_argmax_tensor.detach().cpu().numpy().astype(np.int64)

        return actions, log_probs, raw_argmax

    def evaluate_actions(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        valid_action_masks: np.ndarray,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate log-probs, entropy, and raw policy logits for PPO updates.

        Returns:
            log_probs: Log probabilities of the taken actions
            entropy: Entropy of the masked action distribution
            policy_logits: Raw (unmasked) policy logits for activity regularization
        """
        policy_logits = self._policy_logits(states)
        masked_logits, _ = self._mask_logits(policy_logits, valid_action_masks)
        log_probs_all = torch.log_softmax(masked_logits, dim=-1)
        probs = torch.softmax(masked_logits, dim=-1)

        log_probs = log_probs_all.gather(1, actions.unsqueeze(1)).squeeze(1)
        entropy = -(probs * log_probs_all).sum(dim=-1)
        return log_probs, entropy, policy_logits


def set_global_seeds(seed: Optional[int]):
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ppo_update(
    policy_agent: PolicyAgent,
    critic_model: ValueNetworkWrapper,
    policy_optimizer: optim.Optimizer,
    critic_optimizer: optim.Optimizer,
    buffer: CentralCriticExperienceBuffer,
    clip_epsilon: float,
    value_coef: float,
    entropy_coef: float,
    activity_coef: float,
    n_epochs: int,
    batch_size: int,
    device: str,
    last_critic_states: np.ndarray,
    gamma: float,
    gae_lambda: float,
    max_grad_norm: float,
    target_kl: Optional[float] = None,
) -> Dict[str, float]:
    """Run PPO updates over the centralized buffer."""
    if len(buffer) == 0:
        return {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy_loss": 0.0,
            "activity_loss": 0.0,
            "total_loss": 0.0,
            "approx_kl": 0.0,
            "clipfrac": 0.0,
            "ratio_mean": 0.0,
            "ratio_std": 0.0,
            "early_stop": 0.0,
        }

    (
        actor_states_tmj,
        critic_states_tmj,
        actions_tmj,
        rewards_tmj,
        old_values_tmj,
        old_log_probs_tmj,
        valid_action_masks_tmj,
        dones_tmj,
    ) = buffer.get()
    time_steps, num_envs = actions_tmj.shape

    critic_model.eval()
    with torch.no_grad():
        next_critic_state = torch.from_numpy(last_critic_states).float().to(device)
        next_values = critic_model(next_critic_state).squeeze(-1).detach().cpu().numpy()
    critic_model.train()

    advantages_tmj, returns_tmj = compute_gae_batched(
        rewards_tmj,
        old_values_tmj,
        dones_tmj,
        next_values,
        gamma,
        gae_lambda,
    )

    # experiences start time-major (T, E, ...); flatten to (T*E, ...) for batching
    actor_states = actor_states_tmj.reshape(time_steps * num_envs, -1)
    critic_states = critic_states_tmj.reshape(time_steps * num_envs, -1)
    actions = actions_tmj.reshape(-1)
    old_log_probs = old_log_probs_tmj.reshape(-1)
    advantages = advantages_tmj.reshape(-1)
    returns = returns_tmj.reshape(-1)
    valid_action_masks = valid_action_masks_tmj.reshape(time_steps * num_envs, -1)

    # Identify decision points: steps where the agent had >1 available action.
    # Only train actor on decision points.
    is_decision = valid_action_masks.sum(axis=-1) > 1
    if not np.any(is_decision):
        return {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy_loss": 0.0,
            "activity_loss": 0.0,
            "total_loss": 0.0,
            "approx_kl": 0.0,
            "clipfrac": 0.0,
            "ratio_mean": 0.0,
            "ratio_std": 0.0,
            "early_stop": 0.0,
        }

    adv_decision = advantages[is_decision]
    if adv_decision.std() > 1e-8:
        adv_decision = (adv_decision - adv_decision.mean()) / (adv_decision.std() + 1e-8)
    advantages_norm = np.zeros_like(advantages, dtype=np.float32)
    advantages_norm[is_decision] = adv_decision.astype(np.float32, copy=False)

    # Keep data on CPU, only move mini-batches to GPU during training
    # This allows scaling to large rollout buffers without GPU OOM
    actor_states_t = torch.from_numpy(actor_states).float()
    critic_states_t = torch.from_numpy(critic_states).float()
    actions_t = torch.from_numpy(actions).long()
    old_log_probs_t = torch.from_numpy(old_log_probs).float()
    advantages_t = torch.from_numpy(advantages_norm).float()
    returns_t = torch.from_numpy(returns).float()

    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_entropy_loss = 0.0
    total_activity_loss = 0.0
    total_loss = 0.0
    total_approx_kl = 0.0
    total_clipfrac = 0.0
    total_ratio_mean = 0.0
    total_ratio_std = 0.0
    n_updates = 0
    early_stop = False

    for _ in range(n_epochs):
        indices = np.random.permutation(len(actor_states))
        for start in range(0, len(actor_states), batch_size):
            end = start + batch_size
            batch_idx = indices[start:end]
            if len(batch_idx) < 2:
                continue

            # Move only this mini-batch to GPU
            batch_actor_states = actor_states_t[batch_idx].to(device)
            batch_critic_states = critic_states_t[batch_idx].to(device)
            batch_actions = actions_t[batch_idx].to(device)
            batch_old_log_probs = old_log_probs_t[batch_idx].to(device)
            batch_advantages = advantages_t[batch_idx].to(device)
            batch_returns = returns_t[batch_idx].to(device)
            batch_valid_masks = valid_action_masks[batch_idx]
            batch_is_decision = is_decision[batch_idx]

            # Policy update: only on decision points (where >1 action was available).
            # Non-decision points are still used for critic bootstrapping/training.
            decision_count = int(np.sum(batch_is_decision))
            if decision_count > 0:
                log_probs, entropy, policy_logits = policy_agent.evaluate_actions(
                    batch_actor_states, batch_actions, batch_valid_masks
                )
                decision_mask_t = torch.as_tensor(batch_is_decision, device=device, dtype=torch.bool)

                log_probs_d = log_probs[decision_mask_t]
                entropy_d = entropy[decision_mask_t]
                logits_d = policy_logits[decision_mask_t]
                old_log_probs_d = batch_old_log_probs[decision_mask_t]
                advantages_d = batch_advantages[decision_mask_t]

                ratio = torch.exp(log_probs_d - old_log_probs_d)
                surr1 = ratio * advantages_d
                surr2 = (
                    torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * advantages_d
                )
                policy_loss = -torch.min(surr1, surr2).mean()
                entropy_loss = -entropy_d.mean()
                approx_kl = float((old_log_probs_d - log_probs_d).mean().item())
                clipfrac = float(((ratio - 1.0).abs() > clip_epsilon).float().mean().item())
                ratio_mean = float(ratio.mean().item())
                ratio_std = float(ratio.std(unbiased=False).item())

                # Activity loss: L2 penalty on raw policy logits to prevent extreme values.
                activity_loss = (logits_d**2).mean()

                policy_optimizer.zero_grad()
                (policy_loss + entropy_coef * entropy_loss + activity_coef * activity_loss).backward()
                torch.nn.utils.clip_grad_norm_(policy_agent.model.parameters(), max_grad_norm)
                if any(
                    param.grad is not None and torch.isnan(param.grad).any()
                    for param in policy_agent.model.parameters()
                ):
                    policy_optimizer.zero_grad()
                    print("NaN grad in policy optimizer")
                    continue
                policy_optimizer.step()
            else:
                # No decision points in this mini-batch (e.g., all forced END_TURN steps).
                policy_loss = torch.tensor(0.0, device=device)
                entropy_loss = torch.tensor(0.0, device=device)
                activity_loss = torch.tensor(0.0, device=device)
                approx_kl = 0.0
                clipfrac = 0.0
                ratio_mean = 0.0
                ratio_std = 0.0

            critic_optimizer.zero_grad()
            values = critic_model(batch_critic_states).squeeze(-1)
            value_loss = F.mse_loss(values, batch_returns)
            (value_coef * value_loss).backward()
            torch.nn.utils.clip_grad_norm_(critic_model.parameters(), max_grad_norm)
            if any(
                param.grad is not None and torch.isnan(param.grad).any()
                for param in critic_model.parameters()
            ):
                critic_optimizer.zero_grad()
                print("NaN grad in critic optimizer")
                continue
            critic_optimizer.step()

            total_policy_loss += float(policy_loss.item())
            total_value_loss += float(value_loss.item())
            total_entropy_loss += float(entropy_loss.item())
            total_activity_loss += float(activity_loss.item())
            total_loss += float(
                policy_loss.item()
                + entropy_coef * entropy_loss.item()
                + activity_coef * activity_loss.item()
                + value_coef * value_loss.item()
            )
            total_approx_kl += approx_kl
            total_clipfrac += clipfrac
            total_ratio_mean += ratio_mean
            total_ratio_std += ratio_std
            n_updates += 1

            if target_kl is not None and approx_kl > 1.5 * target_kl:
                # PPO safety valve to avoid destructive updates.
                early_stop = True
                break
        if early_stop:
            break

    if n_updates == 0:
        return {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy_loss": 0.0,
            "activity_loss": 0.0,
            "total_loss": 0.0,
            "approx_kl": 0.0,
            "clipfrac": 0.0,
            "ratio_mean": 0.0,
            "ratio_std": 0.0,
            "early_stop": 1.0 if early_stop else 0.0,
        }

    return {
        "policy_loss": total_policy_loss / n_updates,
        "value_loss": total_value_loss / n_updates,
        "entropy_loss": total_entropy_loss / n_updates,
        "activity_loss": total_activity_loss / n_updates,
        "total_loss": total_loss / n_updates,
        "approx_kl": total_approx_kl / n_updates,
        "clipfrac": total_clipfrac / n_updates,
        "ratio_mean": total_ratio_mean / n_updates,
        "ratio_std": total_ratio_std / n_updates,
        "early_stop": 1.0 if early_stop else 0.0,
    }


def train(
    num_players: int = 2,
    map_type: Literal["BASE", "TOURNAMENT", "MINI"] = "BASE",
    model_type: str = "flat",
    backbone_type: str = "mlp",
    total_timesteps: int = 1_000_000,
    rollout_steps: int = 4096,
    policy_lr: float = 3e-4,
    critic_lr: float = 3e-4,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_epsilon: float = 0.2,
    value_coef: float = 0.5,
    entropy_coef: float = 0.01,
    activity_coef: float = 0.0,
    ppo_epochs: int = 4,
    batch_size: int = 256,
    policy_hidden_dims: Sequence[int] = (512, 512),
    critic_hidden_dims: Sequence[int] = (512, 512),
    save_path: Optional[str] = "weights/marl_central_critic",
    load_policy_weights: Optional[str] = None,
    load_critic_weights: Optional[str] = None,
    wandb_config: Optional[Dict] = None,
    seed: Optional[int] = 42,
    max_grad_norm: float = 0.5,
    deterministic_policy: bool = False,
    eval_games_per_opponent: int = 250,
    trend_eval_games_per_opponent: Optional[int] = None,
    trend_eval_seed: Optional[int] = 42,
    eval_every_updates: int = 1,
    target_kl: Optional[float] = None,
    num_envs: int = 2,
    reward_function: Literal["shaped", "win"] = "shaped",
    metric_window: int = 200,
) -> Tuple[PolicyNetworkWrapper, ValueNetworkWrapper]:
    """Train a shared policy with a centralized critic using PPO."""

    assert 2 <= num_players <= 4, "num_players must be between 2 and 4"
    assert num_envs >= 1, "num_envs must be >= 1"
    assert backbone_type in ("mlp", "xdim"), f"Unknown backbone_type '{backbone_type}'"

    set_global_seeds(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    actor_input_dim, board_shape, numeric_dim = compute_multiagent_input_dim(num_players, map_type)
    numeric_dim = numeric_dim or len(get_numeric_feature_names(num_players, map_type))
    board_dim = actor_input_dim - numeric_dim
    full_numeric_len = len(get_full_numeric_feature_names(num_players, map_type))
    critic_input_dim = full_numeric_len + board_dim

    print(f"\n{'=' * 60}")
    print("Centralized Critic Multi-Agent PPO Training")
    print(f"{'=' * 60}")
    print(f"Device: {device}")
    print(f"Map type: {map_type} | Players: {num_players}")
    print(f"Backbone: {backbone_type} | Model type: {model_type}")
    print(f"Actor input dim: {actor_input_dim} | Critic input dim: {critic_input_dim}")
    if backbone_type == "xdim":
        print(f"Board shape (C, W, H): {board_shape} | Numeric dim: {numeric_dim}")
    print(f"Total timesteps: {total_timesteps:,} | Rollout steps: {rollout_steps}")
    print(f"Parallel environments: {num_envs}")

    if wandb.run is None:
        if wandb_config:
            wandb.init(**wandb_config)
        else:
            wandb.init(mode="disabled")

    default_save_dir = "weights/marl_central_critic"
    if save_path == default_save_dir and wandb.run is not None and getattr(wandb.run, "name", None):
        save_path = os.path.join("weights", wandb.run.name)

    # Build backbone config based on backbone_type
    if backbone_type == "mlp":
        policy_backbone_config = BackboneConfig(
            architecture="mlp",
            args=MLPBackboneConfig(input_dim=actor_input_dim, hidden_dims=list(policy_hidden_dims)),
        )
    else:  # xdim
        # board_shape is (C, W, H) from catanatron, but we need (H, W, C)
        assert board_shape is not None
        board_channels, board_width, board_height = board_shape
        policy_backbone_config = BackboneConfig(
            architecture="cross_dimensional",
            args=CrossDimensionalBackboneConfig(
                board_height=board_height,
                board_width=board_width,
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
            backbone_config=policy_backbone_config, num_actions=ACTION_SPACE_SIZE
        ).to(device)
    elif model_type == "hierarchical":
        policy_model = build_hierarchical_policy_network(backbone_config=policy_backbone_config).to(
            device
        )
    else:
        raise ValueError(f"Unknown model_type '{model_type}'")

    # Build critic backbone config
    if backbone_type == "mlp":
        critic_backbone_config = BackboneConfig(
            architecture="mlp",
            args=MLPBackboneConfig(input_dim=critic_input_dim, hidden_dims=list(critic_hidden_dims)),
        )
    else:  # xdim
        assert board_shape is not None
        board_channels, board_width, board_height = board_shape
        critic_backbone_config = BackboneConfig(
            architecture="cross_dimensional",
            args=CrossDimensionalBackboneConfig(
                board_height=board_height,
                board_width=board_width,
                board_channels=board_channels,
                numeric_dim=full_numeric_len,  # critic uses full numeric features
                cnn_channels=[64, 128, 128],
                cnn_kernel_size=(3, 5),
                numeric_hidden_dims=list(critic_hidden_dims),
                fusion_hidden_dim=critic_hidden_dims[-1] if critic_hidden_dims else 256,
                output_dim=critic_hidden_dims[-1] if critic_hidden_dims else 256,
            ),
        )
    critic_model = build_value_network(backbone_config=critic_backbone_config).to(device)

    if load_policy_weights and os.path.exists(load_policy_weights):
        print(f"Loading policy weights from {load_policy_weights}")
        policy_model.load_state_dict(torch.load(load_policy_weights, map_location=device))
    if load_critic_weights and os.path.exists(load_critic_weights):
        print(f"Loading critic weights from {load_critic_weights}")
        critic_model.load_state_dict(torch.load(load_critic_weights, map_location=device))

    policy_agent = PolicyAgent(policy_model, model_type, device)
    policy_optimizer = optim.Adam(policy_model.parameters(), lr=policy_lr)
    critic_optimizer = optim.Adam(critic_model.parameters(), lr=critic_lr)

    buffer = CentralCriticExperienceBuffer(
        num_rollouts=rollout_steps,
        actor_state_dim=actor_input_dim,
        critic_state_dim=critic_input_dim,
        action_space_size=ACTION_SPACE_SIZE,
        num_envs=num_envs * num_players,
    )

    envs = make_marl_vectorized_envs(
        num_players=num_players,
        map_type=map_type,
        shared_critic=True,
        reward_function=reward_function,
        num_envs=num_envs,
    )
    driver_env = envs.driver_env
    obs_space = driver_env.env_single_observation_space
    obs_dtype = driver_env.obs_dtype
    agents_per_env = getattr(envs, "agents_per_env", [driver_env.num_agents] * num_envs)
    env_offsets = []
    ptr = 0
    for count in agents_per_env:
        env_offsets.append(ptr)
        ptr += count

    def decode_observations(flat_obs: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        batch_size = flat_obs.shape[0]
        actor_batch = np.zeros((batch_size, actor_input_dim), dtype=np.float32)
        critic_batch = np.zeros((batch_size, critic_input_dim), dtype=np.float32)
        action_masks = np.zeros((batch_size, ACTION_SPACE_SIZE), dtype=np.bool_)
        for idx in range(batch_size):
            structured = nativize(flat_obs[idx], obs_space, obs_dtype)
            actor_batch[idx] = flatten_marl_observation(structured)
            critic_batch[idx] = np.asarray(structured["critic"], dtype=np.float32).reshape(-1)
            mask = np.asarray(structured["action_mask"], dtype=np.int8).reshape(-1)
            action_masks[idx] = mask > 0
        return actor_batch, critic_batch, action_masks

    metric_window = max(1, metric_window)
    eval_every_updates = max(1, int(eval_every_updates))
    trend_eval_games = (
        eval_games_per_opponent
        if trend_eval_games_per_opponent is None
        else max(1, trend_eval_games_per_opponent)
    )
    best_eval_win_rate = -float("inf")
    best_eval_critic_mse = float("inf")
    global_step = 0
    total_episodes = 0
    ppo_update_count = 0
    episode_lengths: list[int] = [0 for _ in range(num_envs)]
    length_window = deque(maxlen=metric_window)

    # Track raw policy preferences (argmax before masking) for diagnostics
    raw_policy_argmax_buffer: list[np.ndarray] = []

    observations, infos = envs.reset()

    try:
        with tqdm(total=total_timesteps, desc="Steps", unit="step", unit_scale=True) as pbar:
            while global_step < total_timesteps:
                actor_batch, critic_batch, valid_masks = decode_observations(observations)
                actor_tensor = torch.from_numpy(actor_batch).float().to(device)
                actions, log_probs, raw_argmax = policy_agent.select_actions_batch(
                    actor_tensor, valid_masks, deterministic=deterministic_policy
                )
                raw_policy_argmax_buffer.append(raw_argmax)

                critic_inputs = torch.from_numpy(critic_batch).float().to(device)
                critic_model.eval()
                with torch.no_grad():
                    values = critic_model(critic_inputs).squeeze(-1).detach().cpu().numpy()
                critic_model.train()

                next_observations, rewards, terminations, truncations, infos = envs.step(actions)
                rewards = rewards.astype(np.float32)
                dones = np.logical_or(terminations, truncations)

                buffer.add_batch(
                    actor_batch,
                    critic_batch,
                    actions,
                    rewards,
                    values,
                    log_probs,
                    valid_masks,
                    dones,
                )
                global_step += num_envs
                pbar.update(num_envs)

                for env_idx, start in enumerate(env_offsets):
                    end = start + agents_per_env[env_idx]
                    idx = int(env_idx)
                    episode_lengths[idx] += 1
                    if dones[start:end].all():
                        total_episodes += 1
                        length_window.append(episode_lengths[idx])
                        avg_length = float(np.mean(length_window))
                        wandb.log(
                            {
                                "episode/avg_length": avg_length,
                                "episode/count": total_episodes,
                            },
                            step=global_step,
                        )
                        episode_lengths[idx] = 0

                observations = next_observations

                if buffer.steps_collected >= rollout_steps and len(buffer) >= batch_size * 2:
                    # Log raw policy preference distribution (argmax before masking)
                    # This reveals what the network inherently prefers, regardless of action validity
                    all_raw_argmax = np.concatenate(raw_policy_argmax_buffer)
                    raw_action_type_dist, _ = compute_action_distributions(all_raw_argmax)
                    raw_policy_log = {
                        f"raw_policy/type_{name}": freq
                        for name, freq in raw_action_type_dist.items()
                    }
                    raw_policy_log.update(
                        {
                            "train/rollout_steps_collected": int(buffer.steps_collected),
                            "train/rollout_samples_collected": int(len(buffer)),
                        }
                    )
                    wandb.log(raw_policy_log, step=global_step)
                    raw_policy_argmax_buffer.clear()

                    _, bootstrap_critic_batch, _ = decode_observations(observations)
                    policy_agent.model.train()
                    metrics = ppo_update(
                        policy_agent=policy_agent,
                        critic_model=critic_model,
                        policy_optimizer=policy_optimizer,
                        critic_optimizer=critic_optimizer,
                        buffer=buffer,
                        clip_epsilon=clip_epsilon,
                        value_coef=value_coef,
                        entropy_coef=entropy_coef,
                        activity_coef=activity_coef,
                        n_epochs=ppo_epochs,
                        batch_size=batch_size,
                        device=device,
                        last_critic_states=bootstrap_critic_batch,
                        gamma=gamma,
                        gae_lambda=gae_lambda,
                        max_grad_norm=max_grad_norm,
                        target_kl=target_kl,
                    )
                    buffer.clear()
                    policy_agent.model.eval()
                    critic_model.eval()
                    ppo_update_count += 1

                    wandb.log(
                        {
                            "train/policy_loss": metrics["policy_loss"],
                            "train/value_loss": metrics["value_loss"],
                            "train/entropy": -metrics["entropy_loss"],
                            "train/activity_loss": metrics["activity_loss"],
                            "train/total_loss": metrics["total_loss"],
                            "train/approx_kl": metrics["approx_kl"],
                            "train/clipfrac": metrics["clipfrac"],
                            "train/ratio_mean": metrics["ratio_mean"],
                            "train/ratio_std": metrics["ratio_std"],
                            "train/early_stop_kl": metrics["early_stop"],
                        },
                        step=global_step,
                    )

                    if save_path:
                        save_dir = save_path
                        os.makedirs(save_dir, exist_ok=True)
                        snapshot_name = f"policy_update_{ppo_update_count + 1}.pt"
                        policy_path = os.path.join(save_dir, snapshot_name)
                        critic_path = os.path.join(
                            save_dir,
                            f"{os.path.splitext(snapshot_name)[0]}_critic.pt",
                        )
                        torch.save(policy_model.state_dict(), policy_path)
                        torch.save(critic_model.state_dict(), critic_path)

                    # Evaluate policy against catanatron bots and critic value predictions.
                    if ppo_update_count % eval_every_updates == 0:
                        policy_agent.model.eval()
                        critic_model.eval()
                        with torch.no_grad():
                            fresh_eval_metrics = eval_policy_value_against_baselines(
                                policy_model=policy_model,
                                critic_model=critic_model,
                                model_type=model_type,
                                map_type=map_type,
                                num_games=eval_games_per_opponent,
                                gamma=gamma,
                                seed=random.randint(0, sys.maxsize),  # different random games each time
                                log_to_wandb=False,
                                global_step=global_step,
                                device=device,
                            )
                            trend_eval_metrics = eval_policy_value_against_baselines(
                                policy_model=policy_model,
                                critic_model=critic_model,
                                model_type=model_type,
                                map_type=map_type,
                                num_games=trend_eval_games,
                                gamma=gamma,
                                seed=trend_eval_seed if trend_eval_seed is not None else 0,
                                log_to_wandb=False,
                                global_step=global_step,
                                device=device,
                            )

                        fresh_log = {}
                        for key, value in fresh_eval_metrics.items():
                            suffix = key.split("/", 1)[1] if "/" in key else key
                            fresh_log[f"eval_fresh/{suffix}"] = value

                        trend_log = {}
                        for key, value in trend_eval_metrics.items():
                            suffix = key.split("/", 1)[1] if "/" in key else key
                            trend_log[f"eval_trend/{suffix}"] = value
                        trend_log["eval_trend/seed"] = (
                            trend_eval_seed if trend_eval_seed is not None else 0
                        )
                        trend_log["eval_trend/games_per_opponent"] = trend_eval_games
                        wandb.log({**fresh_log, **trend_log}, step=global_step)

                        if save_path:
                            eval_win_rate = float(
                                trend_eval_metrics.get("eval/win_rate_vs_value", 0.0)
                            )
                            if eval_win_rate > best_eval_win_rate:
                                best_eval_win_rate = eval_win_rate
                                save_dir = save_path
                                os.makedirs(save_dir, exist_ok=True)
                                best_policy_path = os.path.join(save_dir, "policy_best.pt")
                                torch.save(policy_model.state_dict(), best_policy_path)
                                if wandb.run is not None:
                                    wandb.run.summary["best_eval_win_rate_vs_value"] = (
                                        best_eval_win_rate
                                    )
                                print(
                                    f"  → Saved best policy (eval win rate vs value: {best_eval_win_rate:.3f})"
                                )

                            eval_critic_mse = trend_eval_metrics.get("eval/value_mse", float("inf"))
                            if eval_critic_mse < best_eval_critic_mse:
                                best_eval_critic_mse = eval_critic_mse
                                save_dir = save_path
                                os.makedirs(save_dir, exist_ok=True)
                                best_critic_path = os.path.join(save_dir, "critic_best.pt")
                                torch.save(critic_model.state_dict(), best_critic_path)
                                if wandb.run is not None:
                                    wandb.run.summary["best_eval_critic_mse"] = best_eval_critic_mse
                                print(
                                    f"  → Saved best critic (eval MSE: {best_eval_critic_mse:.4f})"
                                )

    finally:
        envs.close()

    if best_eval_win_rate > -float("inf"):
        print(
            f"\nTraining complete. {global_step:,} steps, {total_episodes} episodes. "
            f"Best eval win rate vs value: {best_eval_win_rate:.3f}"
        )
    else:
        print(
            f"\nTraining complete. {global_step:,} steps, {total_episodes} episodes. "
            "Best eval win rate vs value: n/a"
        )
    return policy_model, critic_model
