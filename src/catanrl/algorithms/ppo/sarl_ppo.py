#!/usr/bin/env python3
"""
Single Agent Reinforcement Learning training script for Catan Policy-Value Network.
Uses the Catanatron gym environment for training with PPO.
"""

import os
import random
import sys
from collections import deque
from typing import Any, Dict, List, Literal, Sequence, Tuple, cast

import numpy as np
import numpy.typing as npt
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from catanatron.gym.envs.catanatron_env import ACTION_SPACE_SIZE, ACTIONS_ARRAY, ACTION_TYPES
from tqdm import tqdm

import wandb

from ...envs.gym.single_env import (
    BOARD_HEIGHT,
    BOARD_WIDTH,
    compute_single_agent_dims,
    create_opponents,
    make_vectorized_envs,
)
from ...eval.training_eval import eval_policy_against_baselines
from ...models import (
    PolicyValueNetworkWrapper,
    build_flat_policy_value_network,
    build_hierarchical_policy_value_network,
    policy_value_to_policy_only,
)
from ...models.backbones import BackboneConfig, CrossDimensionalBackboneConfig, MLPBackboneConfig
from .buffers import ExperienceBuffer
from .gae import compute_gae_batched


def _build_action_type_mapping() -> Tuple[Dict[str, list], np.ndarray]:
    """Build action-type lookup tables for action-distribution diagnostics."""
    action_type_to_indices: Dict[str, list] = {at.name: [] for at in ACTION_TYPES}
    action_to_type_idx = np.zeros(ACTION_SPACE_SIZE, dtype=np.int32)
    for action_idx, (action_type, _) in enumerate(ACTIONS_ARRAY):
        action_type_to_indices[action_type.name].append(action_idx)
        action_to_type_idx[action_idx] = ACTION_TYPES.index(action_type)
    return action_type_to_indices, cast(np.ndarray, action_to_type_idx)


ACTION_TYPE_TO_INDICES, ACTION_TO_TYPE_IDX = _build_action_type_mapping()
ACTION_TYPE_NAMES = [at.name for at in ACTION_TYPES]


def compute_action_distributions(
    actions: np.ndarray,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Compute action-type and raw-action distributions from chosen actions."""
    n_actions = len(actions)
    if n_actions == 0:
        return {}, {}

    action_types_taken = ACTION_TO_TYPE_IDX[actions]
    type_counts = np.bincount(action_types_taken, minlength=len(ACTION_TYPES))
    action_type_dist = {
        name: float(count / n_actions) for name, count in zip(ACTION_TYPE_NAMES, type_counts)
    }

    action_counts = np.bincount(actions, minlength=ACTION_SPACE_SIZE)
    action_dist = {
        f"action_{idx}": float(count / n_actions) for idx, count in enumerate(action_counts) if count > 0
    }
    return action_type_dist, action_dist


class SARLAgent:
    """Agent wrapper around a policy/value network for single-agent PPO training."""

    def __init__(
        self,
        model: PolicyValueNetworkWrapper,
        model_type: str,
        device: str | torch.device,
    ):
        self.model = model
        self.model_type = model_type
        self.device = device

    def _policy_logits(self, states: torch.Tensor) -> torch.Tensor:
        if self.model_type == "flat":
            policy_logits, _ = self.model(states)
        elif self.model_type == "hierarchical":
            action_type_logits, param_logits, _ = self.model(states)
            policy_logits = self.model.get_flat_action_logits(action_type_logits, param_logits)
        else:  # pragma: no cover
            raise ValueError(f"Unknown model_type '{self.model_type}'")
        return policy_logits

    def _mask_logits(
        self, policy_logits: torch.Tensor, valid_action_masks: npt.NDArray[np.bool_]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mask = torch.as_tensor(valid_action_masks, dtype=torch.bool, device=policy_logits.device)
        if mask.ndim == 1:
            mask = mask.unsqueeze(0)
        no_valid = ~mask.any(dim=1, keepdim=True)
        mask = torch.where(no_valid, torch.ones_like(mask), mask)
        masked_logits = torch.where(mask, policy_logits, torch.full_like(policy_logits, float("-inf")))
        return torch.clamp(masked_logits, min=-100, max=100), mask

    def select_actions_batch(
        self,
        states: torch.Tensor,
        valid_action_masks: npt.NDArray[np.bool_],
        deterministic: bool = False,
    ) -> Tuple[
        npt.NDArray[np.int16],
        npt.NDArray[np.float32],
        npt.NDArray[np.float32],
        npt.NDArray[np.int64],
    ]:
        """Vectorized action selection for multiple parallel environments."""
        self.model.eval()

        with torch.no_grad():
            if self.model_type == "flat":
                policy_logits, values = self.model(states)
            elif self.model_type == "hierarchical":
                action_type_logits, param_logits, values = self.model(states)
                policy_logits = self.model.get_flat_action_logits(action_type_logits, param_logits)
            else:  # pragma: no cover
                raise ValueError(f"Unknown model_type '{self.model_type}'")

            batch_size, num_actions = policy_logits.shape

            raw_argmax_tensor = torch.argmax(policy_logits, dim=-1)
            masked_logits, _ = self._mask_logits(policy_logits, valid_action_masks)
            probs = F.softmax(masked_logits, dim=-1)
            log_probs_all = F.log_softmax(masked_logits, dim=-1)

            if deterministic:
                action_tensor = torch.argmax(masked_logits, dim=-1)
            else:
                dist = torch.distributions.Categorical(probs=probs)
                action_tensor = dist.sample()

            log_prob_tensor = log_probs_all.gather(1, action_tensor.unsqueeze(1)).squeeze(1)
            value_tensor = values.reshape(batch_size)

            actions = action_tensor.detach().cpu().numpy().astype(np.int16, copy=False)
            log_probs_out = log_prob_tensor.detach().cpu().numpy().astype(np.float32, copy=False)
            values_out = value_tensor.detach().cpu().numpy().astype(np.float32, copy=False)
            raw_argmax = raw_argmax_tensor.detach().cpu().numpy().astype(np.int64, copy=False)

        return actions, log_probs_out, values_out, raw_argmax

    def evaluate_actions(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        valid_action_masks: npt.NDArray[np.bool_],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate actions for PPO updates."""
        if self.model_type == "flat":
            policy_logits, values = self.model(states)
        elif self.model_type == "hierarchical":
            action_type_logits, param_logits, values = self.model(states)
            policy_logits = cast(Any, self.model).get_flat_action_logits(
                action_type_logits, param_logits
            )
        else:  # pragma: no cover
            raise ValueError(f"Unknown model_type '{self.model_type}'")

        masked_logits, _ = self._mask_logits(policy_logits, valid_action_masks)
        log_probs_all = F.log_softmax(masked_logits, dim=-1)
        probs = F.softmax(masked_logits, dim=-1)

        log_probs = log_probs_all.gather(1, actions.unsqueeze(1)).squeeze(1)
        entropy = -(probs * log_probs_all).sum(dim=-1)

        return log_probs, values.squeeze(-1), entropy, policy_logits


def ppo_update(
    agent: SARLAgent,
    optimizer: torch.optim.Optimizer,
    buffer: ExperienceBuffer,
    clip_epsilon: float,
    value_coef: float,
    entropy_coef: float,
    activity_coef: float,
    n_epochs: int,
    batch_size: int,
    device: str | torch.device,
    last_states: np.ndarray,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    max_grad_norm: float = 0.5,
    target_kl: float | None = None,
) -> Dict[str, float]:
    """
    Run PPO updates over the collected on-policy buffer for SARL training.
    """
    if len(buffer) == 0:
        print("No experiences found for PPO update. Skipping update.")
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
            "single_action_fraction": 0.0,
            "num_updates": 0,
            "early_stop": 0.0,
        }

    (
        states_tmj,
        actions_tmj,
        rewards_tmj,
        old_values_tmj,
        old_log_probs_tmj,
        valid_action_masks_tmj,
        dones_tmj,
    ) = buffer.get()
    time_steps, num_envs = actions_tmj.shape

    agent.model.eval()
    with torch.no_grad():
        next_state = torch.from_numpy(last_states).float().to(device)  # [num_envs, state_dim]
        if agent.model_type == "flat":
            _, next_value = agent.model(next_state)
        elif agent.model_type == "hierarchical":
            _, _, next_value = agent.model(next_state)
        next_values = next_value.squeeze(-1).detach().cpu().numpy()
    agent.model.train()

    advantages_tmj, returns_tmj = compute_gae_batched(
        rewards_tmj, old_values_tmj, dones_tmj, next_values, gamma, gae_lambda
    )

    # experiences start time-major (T, E, ...); flatten to (T*E, ...) for batching
    states = states_tmj.reshape(time_steps * num_envs, -1)
    actions = actions_tmj.reshape(-1)
    old_log_probs = old_log_probs_tmj.reshape(-1)
    advantages = advantages_tmj.reshape(-1)
    returns = returns_tmj.reshape(-1)
    valid_action_masks = valid_action_masks_tmj.reshape(time_steps * num_envs, -1)

    # Keep all samples for critic training, but only train actor on decision points.
    single_action_fraction = float((valid_action_masks.sum(axis=1) <= 1).mean())
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
            "single_action_fraction": single_action_fraction,
            "num_updates": 0,
            "early_stop": 0.0,
        }

    adv_decision = advantages[is_decision]
    if adv_decision.std() > 1e-8:
        adv_decision = (adv_decision - adv_decision.mean()) / (adv_decision.std() + 1e-8)
    advantages_norm = np.zeros_like(advantages, dtype=np.float32)
    advantages_norm[is_decision] = adv_decision.astype(np.float32, copy=False)

    # Keep rollout tensors on CPU; move only mini-batches to the training device.
    states_t = torch.from_numpy(states).float()
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
        indices = np.random.permutation(len(states))
        for start in range(0, len(states), batch_size):
            end = start + batch_size
            batch_indices = indices[start:end]
            if len(batch_indices) <= 1:
                print(f"Skipping small batch: {len(batch_indices)}")
                continue

            batch_states = states_t[batch_indices].to(device)
            batch_actions = actions_t[batch_indices].to(device)
            batch_old_log_probs = old_log_probs_t[batch_indices].to(device)
            batch_advantages = advantages_t[batch_indices].to(device)
            batch_returns = returns_t[batch_indices].to(device)
            batch_valid_action_masks = valid_action_masks[batch_indices]
            batch_is_decision = is_decision[batch_indices]

            log_probs, values, entropy, policy_logits = agent.evaluate_actions(
                batch_states, batch_actions, batch_valid_action_masks
            )

            decision_count = int(np.sum(batch_is_decision))
            if decision_count > 0:
                decision_mask_t = torch.as_tensor(batch_is_decision, device=device, dtype=torch.bool)

                log_probs_d = log_probs[decision_mask_t]
                entropy_d = entropy[decision_mask_t]
                logits_d = policy_logits[decision_mask_t]
                old_log_probs_d = batch_old_log_probs[decision_mask_t]
                advantages_d = batch_advantages[decision_mask_t]

                ratio = torch.exp(log_probs_d - old_log_probs_d)
                surr1 = ratio * advantages_d
                surr2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * advantages_d
                policy_loss = -torch.min(surr1, surr2).mean()
                entropy_loss = -entropy_d.mean()
                activity_loss = (logits_d**2).mean()
                approx_kl = float((old_log_probs_d - log_probs_d).mean().item())
                clipfrac = float(((ratio - 1.0).abs() > clip_epsilon).float().mean().item())
                ratio_mean = float(ratio.mean().item())
                ratio_std = float(ratio.std(unbiased=False).item())
            else:
                policy_loss = torch.tensor(0.0, device=device)
                entropy_loss = torch.tensor(0.0, device=device)
                activity_loss = torch.tensor(0.0, device=device)
                approx_kl = 0.0
                clipfrac = 0.0
                ratio_mean = 0.0
                ratio_std = 0.0

            value_loss = F.mse_loss(values, batch_returns)
            loss = (
                policy_loss
                + value_coef * value_loss
                + entropy_coef * entropy_loss
                + activity_coef * activity_loss
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.model.parameters(), max_grad_norm)

            has_nan_grad = any(
                param.grad is not None and torch.isnan(param.grad).any()
                for param in agent.model.parameters()
            )
            if has_nan_grad:
                print("  WARNING: NaN gradients detected! Skipping batch update.")
                optimizer.zero_grad()
                continue

            optimizer.step()

            total_approx_kl += approx_kl
            total_clipfrac += clipfrac
            total_ratio_mean += ratio_mean
            total_ratio_std += ratio_std

            total_policy_loss += float(policy_loss.item())
            total_value_loss += float(value_loss.item())
            total_entropy_loss += float(entropy_loss.item())
            total_activity_loss += float(activity_loss.item())
            total_loss += float(loss.item())
            n_updates += 1

            if target_kl is not None and approx_kl > 1.5 * target_kl:
                # PPO safety valve to prevent overly large policy updates and collapse.
                early_stop = True
                break
        if early_stop:
            break

    if n_updates == 0:
        print("  WARNING: n_updates was 0! Loop didn't run or all batches were too small.")
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
            "single_action_fraction": single_action_fraction,
            "num_updates": 0,
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
        "single_action_fraction": single_action_fraction,
        "num_updates": n_updates,
        "early_stop": 1.0 if early_stop else 0.0,
    }


def train(
    input_dim: int | None = None,
    num_actions: int = ACTION_SPACE_SIZE,
    model_type: str = "flat",
    backbone_type: str = "mlp",
    xdim_cnn_channels: Sequence[int] = (64, 128, 128),
    xdim_cnn_kernel_size: Tuple[int, int] = (3, 5),
    xdim_fusion_hidden_dim: int | None = None,
    hidden_dims: Sequence[int] = (512, 512),
    load_weights: str | None = None,
    total_timesteps: int = 1_000_000,
    n_episodes: int | None = None,
    rollout_steps: int = 4096,
    lr: float = 1e-4,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_epsilon: float = 0.2,
    value_coef: float = 0.5,
    entropy_coef: float = 0.01,
    activity_coef: float = 0.0,
    n_epochs: int = 4,
    batch_size: int = 64,
    save_path: str | None = "weights/sarl_ppo",
    save_every_updates: int = 1,
    device: str | torch.device = "cuda" if torch.cuda.is_available() else "cpu",
    wandb_config: dict | None = None,
    map_type: Literal["BASE", "TOURNAMENT", "MINI"] = "BASE",
    opponent_configs: List[str] | None = None,
    reward_function: str = "shaped",
    num_envs: int = 4,
    use_lr_scheduler: bool = False,
    lr_scheduler_kwargs: dict | None = None,
    metric_window: int = 200,
    eval_games_per_opponent: int = 0,
    trend_eval_games_per_opponent: int | None = None,
    trend_eval_seed: int | None = 42,
    eval_every_updates: int = 0,
    deterministic_policy: bool = False,
    max_grad_norm: float = 0.5,
    target_kl: float | None = None,
):
    """Train the shared policy/value network using single-agent PPO."""
    if total_timesteps <= 0:
        raise ValueError("total_timesteps must be > 0")
    if num_envs <= 0:
        raise ValueError("num_envs must be > 0")

    print(f"\n{'=' * 60}")
    print("Training Policy-Value Network with Single Agent Reinforcement Learning (PPO)")
    print(f"{'=' * 60}")
    print(f"Device: {device}")
    print(f"Map type: {map_type}")
    print(f"Total timesteps: {total_timesteps:,}")
    if n_episodes is not None:
        print(f"Episode cap: {n_episodes}")
    print(f"Rollout steps: {rollout_steps}")
    print(f"Parallel environments: {num_envs}")

    opponent_configs = opponent_configs or ["random"]
    opponents = create_opponents(opponent_configs)
    num_players = len(opponents) + 1  # add the learning agent
    dims = compute_single_agent_dims(num_players, map_type)
    actor_input_dim = dims["actor_dim"]
    numeric_dim = dims["numeric_dim"]
    board_channels = dims["board_channels"]
    if input_dim is not None and int(input_dim) != actor_input_dim:
        print(
            f"Warning: input_dim={input_dim} does not match computed actor dim {actor_input_dim}. "
            f"Using computed dim {actor_input_dim}."
        )

    print(f"Number of players: {num_players}")
    print(f"Opponents: {[repr(o) for o in opponents]}")
    print(f"Backbone: {backbone_type} | Model type: {model_type}")

    if wandb.run is None:
        if wandb_config:
            wandb.init(**wandb_config)
            print(
                f"Initialized wandb: {wandb_config['project']}/"
                f"{wandb_config.get('name', wandb.run.name if wandb.run is not None else 'run')}"
            )
        else:
            wandb.init(mode="disabled")

    default_save_dir = "weights/sarl_ppo"
    if save_path == default_save_dir and wandb.run is not None and getattr(wandb.run, "name", None):
        save_path = os.path.join("weights", wandb.run.name)

    xdim_cnn_channels = list(xdim_cnn_channels)
    if backbone_type not in ("mlp", "xdim", "xdim_res"):
        raise ValueError(f"Unknown backbone_type '{backbone_type}'")
    if backbone_type != "mlp" and not xdim_cnn_channels:
        raise ValueError("xdim_cnn_channels cannot be empty")

    if backbone_type == "mlp":
        backbone_config = BackboneConfig(
            architecture="mlp",
            args=MLPBackboneConfig(input_dim=actor_input_dim, hidden_dims=list(hidden_dims)),
        )
    else:
        xdim_output_dim = hidden_dims[-1] if hidden_dims else 256
        fusion_hidden_dim = (
            xdim_fusion_hidden_dim if xdim_fusion_hidden_dim is not None else xdim_output_dim
        )
        xdim_architecture = (
            "residual_cross_dimensional"
            if backbone_type == "xdim_res"
            else "cross_dimensional"
        )
        backbone_config = BackboneConfig(
            architecture=xdim_architecture,
            args=CrossDimensionalBackboneConfig(
                board_height=BOARD_HEIGHT,
                board_width=BOARD_WIDTH,
                board_channels=board_channels,
                numeric_dim=numeric_dim,
                cnn_channels=xdim_cnn_channels,
                cnn_kernel_size=xdim_cnn_kernel_size,
                numeric_hidden_dims=list(hidden_dims),
                fusion_hidden_dim=fusion_hidden_dim,
                output_dim=xdim_output_dim,
            ),
        )
    if model_type == "flat":
        model = build_flat_policy_value_network(
            backbone_config=backbone_config, num_actions=num_actions
        ).to(device)
    elif model_type == "hierarchical":
        model = build_hierarchical_policy_value_network(backbone_config=backbone_config).to(device)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    if load_weights and os.path.exists(load_weights):
        print(f"Loading weights from: {load_weights}")
        state_dict = torch.load(load_weights, map_location=device)
        model.load_state_dict(state_dict)
        print("  Weights loaded succesfully.")

    agent = SARLAgent(model, model_type, device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Create learning rate scheduler
    scheduler = None
    scheduler_total_iters = None
    scheduler_steps_taken = 0
    if use_lr_scheduler:
        scheduler_kwargs = lr_scheduler_kwargs or {}
        start_factor = scheduler_kwargs.get("start_factor", 1.0)
        end_factor = scheduler_kwargs.get("end_factor", 0.0)
        default_total_iters = (
            max(1, total_timesteps // max(1, num_envs))
            if n_episodes is None
            else max(1, n_episodes)
        )
        total_iters = scheduler_kwargs.get("total_iters", default_total_iters)
        scheduler_total_iters = total_iters
        scheduler = lr_scheduler.LinearLR(
            optimizer, start_factor=start_factor, end_factor=end_factor, total_iters=total_iters
        )
        print(
            f"LR Scheduler: LinearLR (start_factor={start_factor}, end_factor={end_factor}, total_iters={total_iters})"
        )
    else:
        print("LR Scheduler: None (constant learning rate)")

    model.eval()

    buffer = ExperienceBuffer(
        num_rollouts=rollout_steps,
        state_dim=actor_input_dim,
        action_space_size=num_actions,
        num_envs=num_envs,
    )
    pending_scheduler_steps = 0
    episode_rewards = deque(maxlen=metric_window)
    episode_lengths = deque(maxlen=metric_window)
    best_eval_win_rate = -float("inf")
    best_avg_reward = float("-inf")
    global_step = 0
    total_episodes = 0
    ppo_update_count = 0
    eval_every_updates = max(0, int(eval_every_updates))
    save_every_updates = max(1, int(save_every_updates))
    if eval_every_updates > 0 and save_every_updates % eval_every_updates != 0:
        raise ValueError(
            "save_every_updates must be a multiple of eval_every_updates "
            f"({eval_every_updates})"
        )
    if eval_every_updates > 0:
        print(f"Eval cadence: every {eval_every_updates} update(s)")
    else:
        print("Eval cadence: disabled")
    print(f"Save cadence: every {save_every_updates} update(s)")
    do_eval = eval_every_updates > 0 and (
        eval_games_per_opponent > 0
        or (trend_eval_games_per_opponent is not None and trend_eval_games_per_opponent > 0)
    )
    trend_eval_games = (
        eval_games_per_opponent
        if trend_eval_games_per_opponent is None
        else max(1, trend_eval_games_per_opponent)
    )
    raw_policy_argmax_buffer: list[np.ndarray] = []

    envs = make_vectorized_envs(
        reward_function=reward_function,
        map_type=map_type,
        opponent_configs=opponent_configs,
        num_envs=num_envs,
        representation="mixed",
    )

    def flatten_observations(observation_batch: Dict[str, np.ndarray]) -> np.ndarray:
        numeric_obs = observation_batch["numeric"].astype(np.float32)
        board_obs = observation_batch["board"].astype(np.float32)
        if board_obs.ndim != 4:
            raise ValueError(f"Expected board batch rank 4, got shape {board_obs.shape}")
        # Canonical flatten order is [numeric, board(W,H,C).reshape(-1)].
        if board_obs.shape[2] == BOARD_WIDTH and board_obs.shape[3] == BOARD_HEIGHT:
            board_obs_wh_last = np.transpose(board_obs, (0, 2, 3, 1))  # [N, C, W, H] -> [N, W, H, C]
        elif board_obs.shape[1] == BOARD_WIDTH and board_obs.shape[2] == BOARD_HEIGHT:
            board_obs_wh_last = board_obs  # already [N, W, H, C]
        else:
            raise ValueError(
                f"Unrecognized board layout {board_obs.shape}; expected [N,C,W,H] or [N,W,H,C]"
            )
        board_flat = board_obs_wh_last.reshape(board_obs_wh_last.shape[0], -1)
        return np.concatenate([numeric_obs, board_flat], axis=1)

    env_episode_rewards = np.zeros(num_envs)
    env_episode_lengths = np.zeros(num_envs, dtype=int)

    print("\nStarting training...")
    observations, infos = envs.reset()

    with tqdm(total=total_timesteps, desc="Steps", unit="step", unit_scale=True) as pbar:
        while global_step < total_timesteps and (n_episodes is None or total_episodes < n_episodes):
            states = flatten_observations(observations)

            info_valid = infos.get("valid_actions")
            valid_action_masks: npt.NDArray[np.bool_] = np.zeros(
                (num_envs, ACTION_SPACE_SIZE), dtype=np.bool_
            )
            for i, v in enumerate(info_valid):
                valid_action_masks[i, v] = True
            states_t = torch.from_numpy(states).float().to(device)
            actions, log_probs, values, raw_argmax = agent.select_actions_batch(
                states_t, valid_action_masks, deterministic=deterministic_policy
            )
            raw_policy_argmax_buffer.append(raw_argmax)

            next_observations, rewards, terminations, truncations, infos = envs.step(actions)

            dones = np.logical_or(terminations, truncations)
            env_episode_rewards += rewards
            env_episode_lengths += 1

            buffer.add_batch(
                states,
                actions,
                rewards,
                values,
                log_probs,
                cast(npt.NDArray[np.bool_], valid_action_masks),
                dones,
            )
            global_step += num_envs
            pbar.update(num_envs)

            done_indices = np.where(dones)[0]
            if len(done_indices) > 0:
                batch_completed_episodes_rewards = env_episode_rewards[done_indices]
                batch_completed_episodes_lengths = env_episode_lengths[done_indices]
                completed_episodes = len(done_indices)

                episode_rewards.extend(batch_completed_episodes_rewards.tolist())
                episode_lengths.extend(batch_completed_episodes_lengths.tolist())
                total_episodes += completed_episodes

                avg_reward = float(np.mean(episode_rewards)) if episode_rewards else 0.0
                avg_length = float(np.mean(episode_lengths)) if episode_lengths else 0.0

                wandb.log(
                    {
                        "episode/batch_completed_episodes": completed_episodes,
                        "episode/batch_completed_episodes_reward_mean": float(
                            np.mean(batch_completed_episodes_rewards)
                        ),
                        "episode/batch_completed_episodes_reward_min": float(
                            np.min(batch_completed_episodes_rewards)
                        ),
                        "episode/batch_completed_episodes_reward_max": float(
                            np.max(batch_completed_episodes_rewards)
                        ),
                        "episode/batch_completed_episodes_length_mean": float(
                            np.mean(batch_completed_episodes_lengths)
                        ),
                        "episode/batch_completed_episodes_length_min": float(
                            np.min(batch_completed_episodes_lengths)
                        ),
                        "episode/batch_completed_episodes_length_max": float(
                            np.max(batch_completed_episodes_lengths)
                        ),
                        "episode/avg_episode_reward": avg_reward,
                        "episode/avg_episode_length": avg_length,
                        "episode": total_episodes,
                    },
                    step=global_step,
                )

                env_episode_rewards[done_indices] = 0.0
                env_episode_lengths[done_indices] = 0

                if scheduler is not None and (
                    scheduler_total_iters is None or scheduler_steps_taken < scheduler_total_iters
                ):
                    pending_scheduler_steps += completed_episodes

                if save_path and avg_reward > best_avg_reward:
                    best_avg_reward = avg_reward
                    os.makedirs(save_path, exist_ok=True)
                    best_train_path = os.path.join(save_path, "policy_best_train_reward.pt")
                    torch.save(model.state_dict(), best_train_path)
                    if wandb.run is not None:
                        wandb.run.summary["best_avg_reward"] = best_avg_reward

            observations = next_observations

            if len(buffer) >= rollout_steps:
                bootstrap_states = flatten_observations(observations)

                # Log raw policy preference distribution (argmax before action-mask application).
                if raw_policy_argmax_buffer:
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

                metrics = ppo_update(
                    agent=agent,
                    optimizer=optimizer,
                    buffer=buffer,
                    clip_epsilon=clip_epsilon,
                    value_coef=value_coef,
                    entropy_coef=entropy_coef,
                    activity_coef=activity_coef,
                    n_epochs=n_epochs,
                    batch_size=batch_size,
                    device=device,
                    last_states=bootstrap_states,
                    gamma=gamma,
                    gae_lambda=gae_lambda,
                    max_grad_norm=max_grad_norm,
                    target_kl=target_kl,
                )
                buffer.clear()
                model.eval()
                ppo_update_count += 1

                # Step learning rate scheduler based on completed episodes since last update
                updates_performed = metrics.get("num_updates", 0)
                lr_before_scheduler = optimizer.param_groups[0]["lr"]
                lr_after_scheduler = lr_before_scheduler
                if updates_performed > 0 and scheduler is not None and pending_scheduler_steps > 0:
                    remaining_steps = (
                        (scheduler_total_iters - scheduler_steps_taken)
                        if scheduler_total_iters is not None
                        else pending_scheduler_steps
                    )
                    if remaining_steps <= 0:
                        pending_scheduler_steps = 0
                    else:
                        steps_to_apply = min(pending_scheduler_steps, remaining_steps)
                        for _ in range(steps_to_apply):
                            scheduler.step()
                        scheduler_steps_taken += steps_to_apply
                        pending_scheduler_steps -= steps_to_apply
                        lr_after_scheduler = optimizer.param_groups[0]["lr"]
                        if lr_after_scheduler != lr_before_scheduler:
                            print(
                                f"  → LR updated: {lr_before_scheduler:.6f} → "
                                f"{lr_after_scheduler:.6f} ({steps_to_apply} episode step"
                                f"{'s' if steps_to_apply != 1 else ''})"
                            )
                current_lr = lr_after_scheduler

                print(
                    f"\n  → PPO Update | "
                    f"Policy Loss: {metrics['policy_loss']:.4f} | "
                    f"Value Loss: {metrics['value_loss']:.4f} | "
                    f"Entropy: {-metrics['entropy_loss']:.4f} | "
                    f"Activity: {metrics['activity_loss']:.4f} | "
                    f"LR: {current_lr:.6f}"
                )

                log_dict = {
                    "train/policy_loss": metrics["policy_loss"],
                    "train/value_loss": metrics["value_loss"],
                    "train/entropy": -metrics["entropy_loss"],
                    "train/activity_loss": metrics["activity_loss"],
                    "train/total_loss": metrics["total_loss"],
                    "train/learning_rate": current_lr,
                    "train/approx_kl": metrics.get("approx_kl", 0.0),
                    "train/clipfrac": metrics.get("clipfrac", 0.0),
                    "train/ratio_mean": metrics.get("ratio_mean", 0.0),
                    "train/ratio_std": metrics.get("ratio_std", 0.0),
                    "train/early_stop_kl": metrics.get("early_stop", 0.0),
                    "train/single_action_fraction": metrics.get("single_action_fraction", 0.0),
                }
                wandb.log(log_dict, step=global_step)

                if save_path and ppo_update_count % save_every_updates == 0:
                    os.makedirs(save_path, exist_ok=True)
                    snapshot_path = os.path.join(save_path, f"policy_update_{ppo_update_count}.pt")
                    torch.save(model.state_dict(), snapshot_path)

                if do_eval and ppo_update_count % eval_every_updates == 0:
                    policy_only = policy_value_to_policy_only(model)
                    fresh_eval_metrics = eval_policy_against_baselines(
                        policy_model=policy_only,
                        model_type=model_type,
                        map_type=map_type,
                        num_games=eval_games_per_opponent,
                        seed=random.randint(0, sys.maxsize),
                        log_to_wandb=False,
                        global_step=global_step,
                    )
                    trend_eval_metrics = eval_policy_against_baselines(
                        policy_model=policy_only,
                        model_type=model_type,
                        map_type=map_type,
                        num_games=trend_eval_games,
                        seed=trend_eval_seed if trend_eval_seed is not None else 0,
                        log_to_wandb=False,
                        global_step=global_step,
                    )
                    fresh_log = {}
                    for key, value in fresh_eval_metrics.items():
                        suffix = key.split("/", 1)[1] if "/" in key else key
                        fresh_log[f"eval_fresh/{suffix}"] = value
                    trend_log = {}
                    for key, value in trend_eval_metrics.items():
                        suffix = key.split("/", 1)[1] if "/" in key else key
                        trend_log[f"eval_trend/{suffix}"] = value
                    trend_log["eval_trend/seed"] = trend_eval_seed if trend_eval_seed is not None else 0
                    trend_log["eval_trend/games_per_opponent"] = trend_eval_games
                    wandb.log({**fresh_log, **trend_log}, step=global_step)

                    if save_path:
                        eval_win_rate = 0.0
                        if (
                            trend_eval_games_per_opponent is not None
                            and trend_eval_games_per_opponent > 0
                        ):
                            eval_win_rate = float(trend_eval_metrics.get("eval/win_rate_vs_value", 0.0))
                        elif eval_games_per_opponent is not None and eval_games_per_opponent > 0:
                            eval_win_rate = float(fresh_eval_metrics.get("eval/win_rate_vs_value", 0.0))

                        if eval_win_rate > best_eval_win_rate:
                            best_eval_win_rate = eval_win_rate
                            os.makedirs(save_path, exist_ok=True)
                            best_path = os.path.join(save_path, "policy_best.pt")
                            torch.save(model.state_dict(), best_path)
                            if wandb.run is not None:
                                wandb.run.summary["best_eval_win_rate_vs_value"] = best_eval_win_rate
                            print(
                                f"  → Saved best policy (eval win rate vs value: {best_eval_win_rate:.3f})"
                            )

    envs.close()
    if do_eval and best_eval_win_rate > -float("inf"):
        print(
            f"\nTraining complete. {global_step:,} steps, {total_episodes} episodes. "
            f"Best eval win rate vs value: {best_eval_win_rate:.3f}"
        )
    else:
        print(
            f"\nTraining complete. {global_step:,} steps, {total_episodes} episodes. "
            f"Best avg training reward: {best_avg_reward:.3f}"
        )
    return model
