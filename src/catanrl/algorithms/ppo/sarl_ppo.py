#!/usr/bin/env python3
"""
Single Agent Reinforcement Learning training script for Catan Policy-Value Network.
Uses the Catanatron gym environment for training with PPO.
"""

import os
from collections import deque
from typing import List, Dict, Tuple, Literal

import numpy as np
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn as nn
import torch.nn.functional as F
import wandb
from tqdm import tqdm

from catanatron.players.value import ValueFunctionPlayer
from catanatron.models.player import RandomPlayer
from catanatron.gym.envs.catanatron_env import ACTION_SPACE_SIZE
from catanrl.players.nn_policy_player import NNPolicyPlayer
from .buffers import ExperienceBuffer
from ...envs.gym.single_env import create_opponents, make_vectorized_envs
from ...models import (
    policy_value_to_policy_only,
    build_flat_policy_value_network,
    build_hierarchical_policy_value_network,
)
from ...models.backbones import BackboneConfig, MLPBackboneConfig
from ...features.catanatron_utils import COLOR_ORDER
from .gae import compute_gae_batched
from ...eval.eval_nn_vs_catanatron import eval


ESTIMATED_STEPS_PER_EPISODE = 100


class SARLAgent:
    """Agent wrapper around a policy/value network for single-agent PPO training."""

    def __init__(
        self,
        model: nn.Module,
        model_type: str,
        device: str | torch.device,
    ):
        self.model = model
        self.model_type = model_type
        self.device = device

    def select_action(
        self,
        state: torch.Tensor,
        valid_actions: np.ndarray,
        deterministic: bool = False,
    ) -> Tuple[int, float, float]:
        """
        Select one action for a single environment step.

        Returns:
            action_idx: Index in ACTION_SPACE_SIZE
            log_prob: Log probability of selected action
            value: Value estimate
        """
        self.model.eval()
        with torch.no_grad():
            if self.model_type == "flat":
                policy_logits, value = self.model(state)
            elif self.model_type == "hierarchical":
                action_type_logits, param_logits, value = self.model(state)
                policy_logits = self.model.get_flat_action_logits(action_type_logits, param_logits)

            if torch.isnan(policy_logits).any() or torch.isnan(value).any():
                print("WARNING: NaN detected in model output!")
                action_idx = np.random.choice(valid_actions)
                log_prob = -np.log(len(valid_actions))
                return action_idx, log_prob, 0.0

            mask = torch.as_tensor(valid_actions, dtype=torch.bool, device=policy_logits.device)
            masked_logits = torch.clamp(
                torch.where(mask, policy_logits, torch.full_like(policy_logits, float("-inf"))),
                min=-100,
                max=100,
            )

            probs = F.softmax(masked_logits, dim=-1)
            log_probs_all = F.log_softmax(masked_logits, dim=-1)

            if deterministic:
                action_idx = torch.argmax(probs[0, valid_actions]).item()
                action_idx = valid_actions[action_idx]
            else:
                valid_probs = probs[0, valid_actions]
                valid_probs = valid_probs / valid_probs.sum()
                dist = torch.distributions.Categorical(probs=valid_probs)
                sampled_idx = dist.sample().item()
                action_idx = valid_actions[sampled_idx]

            log_prob = log_probs_all[0, action_idx].item()

        return int(action_idx), float(log_prob), float(value.item())

    def select_actions_batch(
        self,
        states: torch.Tensor,
        valid_action_masks: np.ndarray,
        deterministic: bool = False,
    ) -> Tuple[List[int], List[float], List[float]]:
        """Vectorized action selection for multiple parallel environments."""
        self.model.eval()
        actions: List[int] = []
        log_probs_out: List[float] = []
        values_out: List[float] = []

        with torch.no_grad():
            if self.model_type == "flat":
                policy_logits, values = self.model(states)
            elif self.model_type == "hierarchical":
                action_type_logits, param_logits, values = self.model(states)
                policy_logits = self.model.get_flat_action_logits(action_type_logits, param_logits)
            batch_size, num_actions = policy_logits.shape

            mask = torch.as_tensor(
                valid_action_masks, dtype=torch.bool, device=policy_logits.device
            )
            masked_logits = torch.clamp(
                torch.where(mask, policy_logits, torch.full_like(policy_logits, float("-inf"))),
                min=-100,
                max=100,
            )
            probs = F.softmax(masked_logits, dim=-1)
            log_probs_all = F.log_softmax(masked_logits, dim=-1)

            for i in range(batch_size):
                valid_actions = valid_action_masks[i].nonzero()[0]
                if deterministic:
                    local_idx = torch.argmax(probs[i, valid_actions]).item()
                    action_idx = int(valid_actions[local_idx])
                else:
                    valid_probs = probs[i, valid_actions]
                    valid_probs = valid_probs / (valid_probs.sum() + 1e-12)
                    dist = torch.distributions.Categorical(probs=valid_probs)
                    sampled_idx = dist.sample().item()
                    action_idx = int(valid_actions[sampled_idx])

                actions.append(action_idx)
                log_probs_out.append(float(log_probs_all[i, action_idx].item()))
                values_out.append(float(values[i].item()))

        return actions, log_probs_out, values_out

    def evaluate_actions(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        valid_action_masks: np.ndarray,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate actions for PPO updates."""
        if self.model_type == "flat":
            policy_logits, values = self.model(states)
        elif self.model_type == "hierarchical":
            action_type_logits, param_logits, values = self.model(states)
            policy_logits = self.model.get_flat_action_logits(action_type_logits, param_logits)

        mask = torch.as_tensor(valid_action_masks, dtype=torch.bool, device=policy_logits.device)
        masked_logits = torch.clamp(
            torch.where(mask, policy_logits, torch.full_like(policy_logits, float("-inf"))),
            min=-100,
            max=100,
        )
        log_probs_all = F.log_softmax(masked_logits, dim=-1)
        probs = F.softmax(masked_logits, dim=-1)

        log_probs = log_probs_all.gather(1, actions.unsqueeze(1)).squeeze(1)
        entropy = -(probs * log_probs_all).sum(dim=-1)

        return log_probs, values.squeeze(-1), entropy


def ppo_update(
    agent: SARLAgent,
    optimizer: torch.optim.Optimizer,
    buffer: ExperienceBuffer,
    clip_epsilon: float,
    value_coef: float,
    entropy_coef: float,
    n_epochs: int,
    batch_size: int,
    device: str | torch.device,
    last_states: np.ndarray,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    max_grad_norm: float = 0.5,
) -> Dict[str, float]:
    """
    Run PPO updates over the collected on-policy buffer for SARL training.
    """
    if len(buffer) == 0:
        return {"policy_loss": 0.0, "value_loss": 0.0, "entropy_loss": 0.0, "total_loss": 0.0}

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

    # Filter out experiences with only one available action (very common, e.g.
    # roll dice or end turn), to prevent biasing our model towards no-ops.
    multi_action_mask = np.array([va.sum() > 1 for va in valid_action_masks], dtype=bool)
    states = states[multi_action_mask]
    actions = actions[multi_action_mask]
    old_log_probs = old_log_probs[multi_action_mask]
    advantages = advantages[multi_action_mask]
    returns = returns[multi_action_mask]
    valid_action_masks = valid_action_masks[multi_action_mask]

    if len(advantages) == 0:
        print("No valid experiences found for PPO update. Skipping update.")
        return {"policy_loss": 0.0, "value_loss": 0.0, "entropy_loss": 0.0, "total_loss": 0.0}


    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    states_t = torch.from_numpy(states).float().to(device)
    actions_t = torch.from_numpy(actions).to(device)
    old_log_probs_t = torch.from_numpy(old_log_probs).float().to(device)
    advantages_t = torch.from_numpy(advantages).float().to(device)
    returns_t = torch.from_numpy(returns).float().to(device)

    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_entropy_loss = 0.0
    total_loss = 0.0
    n_updates = 0

    for _ in range(n_epochs):
        indices = np.random.permutation(len(states))
        for start in range(0, len(states), batch_size):
            end = start + batch_size
            batch_indices = indices[start:end]
            if len(batch_indices) <= 1:
                print(f"Skipping small batch: {len(batch_indices)}")
                continue

            batch_states = states_t[batch_indices]
            batch_actions = actions_t[batch_indices]
            batch_old_log_probs = old_log_probs_t[batch_indices]
            batch_advantages = advantages_t[batch_indices]
            batch_returns = returns_t[batch_indices]
            batch_valid_action_masks = valid_action_masks[batch_indices]

            log_probs, values, entropy = agent.evaluate_actions(
                batch_states, batch_actions, batch_valid_action_masks
            )
            
            ratio = torch.exp(log_probs - batch_old_log_probs)
            surr1 = ratio * batch_advantages
            surr2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * batch_advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = F.mse_loss(values, batch_returns)
            entropy_loss = -entropy.mean()

            loss = policy_loss + value_coef * value_loss + entropy_coef * entropy_loss
            
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

            total_policy_loss += float(policy_loss.item())
            total_value_loss += float(value_loss.item())
            total_entropy_loss += float(entropy_loss.item())
            total_loss += float(loss.item())
            n_updates += 1

    if n_updates == 0:
        print("  WARNING: n_updates was 0! Loop didn't run or all batches were too small.")
        return {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy_loss": 0.0,
            "total_loss": 0.0,
            "num_updates": 0,
        }

    return {
        "policy_loss": total_policy_loss / n_updates,
        "value_loss": total_value_loss / n_updates,
        "entropy_loss": total_entropy_loss / n_updates,
        "total_loss": total_loss / n_updates,
        "num_updates": n_updates,
    }


def train(
    input_dim: int,
    num_actions: int = ACTION_SPACE_SIZE,
    model_type: str = "flat",
    hidden_dims: List[int] = (512, 512),
    load_weights: str | None = None,
    n_episodes: int = 1000,
    rollout_steps: int = 4096,
    lr: float = 1e-4,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_epsilon: float = 0.2,
    value_coef: float = 0.5,
    entropy_coef: float = 0.01,
    n_epochs: int = 4,
    batch_size: int = 64,
    save_path: str = "weights/policy_value_rl_best.pt",
    save_freq: int = 100,
    device: str | torch.device = "cuda" if torch.cuda.is_available() else "cpu",
    wandb_config: dict | None = None,
    map_type: Literal["BASE", "TOURNAMENT", "MINI"] = "BASE",
    opponent_configs: List[str] | None = None,
    reward_function: str = "shaped",
    num_envs: int = 4,
    use_lr_scheduler: bool = False,
    lr_scheduler_kwargs: dict | None = None,
    metric_window: int = 200,
    num_validation_games: int = 250,
):
    """Train the shared policy/value network using single-agent PPO."""

    print(f"\n{'=' * 60}")
    print("Training Policy-Value Network with Single Agent Reinforcement Learning (PPO)")
    print(f"{'=' * 60}")
    print(f"Device: {device}")
    print(f"Map type: {map_type}")
    print(f"Episodes: {n_episodes}")
    print(f"Rollout steps: {rollout_steps}")
    print(f"Parallel environments: {num_envs}")

    opponent_configs = opponent_configs or ["random"]
    opponents = create_opponents(opponent_configs)
    num_players = len(opponents) + 1  # add the learning agent
    print(f"Number of players: {num_players}")
    print(f"Opponents: {[repr(o) for o in opponents]}")

    if wandb_config:
        wandb.init(**wandb_config)
        print(
            f"Initialized wandb: {wandb_config['project']}/"
            f"{wandb_config.get('name', wandb.run.name)}"
        )
    else:
        wandb.init(mode="disabled")

    backbone_config = BackboneConfig(
        architecture="mlp", args=MLPBackboneConfig(input_dim=input_dim, hidden_dims=hidden_dims)
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
        total_iters = scheduler_kwargs.get("total_iters", n_episodes)
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
        state_dim=input_dim,
        action_space_size=num_actions,
        num_envs=num_envs,
    )
    pending_scheduler_steps = 0
    episode_rewards = deque(maxlen=metric_window)
    episode_lengths = deque(maxlen=metric_window)
    best_avg_reward = float("-inf")
    global_step = 0
    total_episodes = 0
    last_checkpoint_episode = 0

    envs = make_vectorized_envs(
        reward_function=reward_function,
        map_type=map_type,
        opponent_configs=opponent_configs,
        num_envs=num_envs,
        representation="mixed",
    )

    def flatten_observations(observation_batch: Dict[str, np.ndarray]) -> np.ndarray:
        numeric_obs = observation_batch["numeric"].astype(np.float32)
        board_obs = observation_batch["board"].astype(np.float32)  # [N, C, H, W]
        board_obs_ch_last = np.transpose(board_obs, (0, 2, 3, 1))
        board_flat = board_obs_ch_last.reshape(board_obs_ch_last.shape[0], -1)
        return np.concatenate([numeric_obs, board_flat], axis=1)

    env_episode_rewards = np.zeros(num_envs)
    env_episode_lengths = np.zeros(num_envs, dtype=int)

    print("\nStarting training...")
    observations, infos = envs.reset()

    with tqdm(total=n_episodes, desc="Episodes") as pbar:
        while total_episodes < n_episodes:
            states = flatten_observations(observations)

            info_valid = infos.get("valid_actions")
            valid_action_masks = np.zeros((num_envs, ACTION_SPACE_SIZE), dtype=np.bool_)
            for i, v in enumerate(info_valid):
                valid_action_masks[i, v] = True
            states_t = torch.from_numpy(states).float().to(device)
            actions, log_probs, values = agent.select_actions_batch(states_t, valid_action_masks)

            next_observations, rewards, terminations, truncations, infos = envs.step(actions)

            dones = np.logical_or(terminations, truncations)
            env_episode_rewards += rewards
            env_episode_lengths += 1

            actions_arr = np.array(actions)
            values_arr = np.array(values)
            log_probs_arr = np.array(log_probs)

            buffer.add_batch(
                states,
                actions_arr,
                rewards,
                values_arr,
                log_probs_arr,
                valid_action_masks,
                dones,
            )
            global_step += num_envs

            done_indices = np.where(dones)[0]
            if len(done_indices) > 0:
                batch_completed_episodes_rewards = env_episode_rewards[done_indices]
                batch_completed_episodes_lengths = env_episode_lengths[done_indices]
                completed_episodes = len(done_indices)

                episode_rewards.extend(batch_completed_episodes_rewards.tolist())
                episode_lengths.extend(batch_completed_episodes_lengths.tolist())
                total_episodes += completed_episodes
                pbar.update(completed_episodes)

                avg_reward = float(np.mean(episode_rewards)) if episode_rewards else 0.0
                avg_length = float(np.mean(episode_lengths)) if episode_lengths else 0.0

                wandb.log(
                    {
                        "episode/batch_completed_episodes": completed_episodes,
                        "episode/batch_completed_episodes_reward_mean": float(np.mean(batch_completed_episodes_rewards)),
                        "episode/batch_completed_episodes_reward_min": float(np.min(batch_completed_episodes_rewards)),
                        "episode/batch_completed_episodes_reward_max": float(np.max(batch_completed_episodes_rewards)),
                        "episode/batch_completed_episodes_length_mean": float(np.mean(batch_completed_episodes_lengths)),
                        "episode/batch_completed_episodes_length_min": float(np.min(batch_completed_episodes_lengths)),
                        "episode/batch_completed_episodes_length_max": float(np.max(batch_completed_episodes_lengths)),
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

                if len(episode_rewards) >= metric_window and avg_reward > best_avg_reward:
                    best_avg_reward = avg_reward
                    torch.save(model.state_dict(), save_path)
                    print(f"  → Saved best model (avg_reward: {avg_reward:.2f})")
                    if wandb.run is not None:
                        wandb.run.summary["best_avg_reward"] = best_avg_reward

                if save_freq > 0:
                    while last_checkpoint_episode + save_freq <= total_episodes:
                        last_checkpoint_episode += save_freq
                        save_dir = os.path.dirname(save_path)
                        checkpoint_path = (
                            os.path.join(save_dir, f"checkpoint_ep{last_checkpoint_episode}.pt")
                            if save_dir
                            else f"checkpoint_ep{last_checkpoint_episode}.pt"
                        )
                        torch.save(model.state_dict(), checkpoint_path)
                        print(f"  → Saved checkpoint: {checkpoint_path}")

            observations = next_observations

            if len(buffer) >= rollout_steps:
                bootstrap_states = flatten_observations(observations)
                # quick lil winrate/vp eval against random and value function player
                wins, vps, total_vps, turns = eval(
                    NNPolicyPlayer(
                        COLOR_ORDER[0],
                        model_type=model_type,
                        model=policy_value_to_policy_only(model),
                    ),
                    [RandomPlayer(COLOR_ORDER[1], is_bot=True)],
                    map_type=map_type,
                    num_games=num_validation_games,
                    seed=42,
                )
                wandb.log(
                    {
                        "eval/win_rate_vs_random": wins / num_validation_games,
                        "eval/avg_vps_vs_random": sum(vps) / len(vps),
                        "eval/avg_total_vps_vs_random": sum(total_vps) / len(total_vps),
                        "eval/avg_turns_vs_random": sum(turns) / len(turns),
                    },
                    step=global_step,
                )

                wins, vps, total_vps, turns = eval(
                    NNPolicyPlayer(
                        COLOR_ORDER[0],
                        model_type=model_type,
                        model=policy_value_to_policy_only(model),
                    ),
                    [ValueFunctionPlayer(COLOR_ORDER[1])],
                    map_type=map_type,
                    num_games=num_validation_games,
                    seed=42,
                )
                wandb.log(
                    {
                        "eval/win_rate_vs_value": wins / num_validation_games,
                        "eval/avg_vps_vs_value": sum(vps) / len(vps),
                        "eval/avg_total_vps_vs_value": sum(total_vps) / len(total_vps),
                        "eval/avg_turns_vs_value": sum(turns) / len(turns),
                    },
                    step=global_step,
                )

                metrics = ppo_update(
                    agent=agent,
                    optimizer=optimizer,
                    buffer=buffer,
                    clip_epsilon=clip_epsilon,
                    value_coef=value_coef,
                    entropy_coef=entropy_coef,
                    n_epochs=n_epochs,
                    batch_size=batch_size,
                    device=device,
                    last_states=bootstrap_states,
                    gamma=gamma,
                    gae_lambda=gae_lambda,
                )
                buffer.clear()
                model.eval()

                # Step learning rate scheduler based on completed episodes since last update
                updates_performed = metrics.get("num_updates", 0)
                lr_before_scheduler = optimizer.param_groups[0]["lr"]
                lr_after_scheduler = lr_before_scheduler
                if (
                    updates_performed > 0
                    and scheduler is not None
                    and pending_scheduler_steps > 0
                ):
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
                    f"LR: {current_lr:.6f}"
                )

                log_dict = {
                    "train/policy_loss": metrics["policy_loss"],
                    "train/value_loss": metrics["value_loss"],
                    "train/entropy": -metrics["entropy_loss"],
                    "train/total_loss": metrics["total_loss"],
                    "train/learning_rate": current_lr,
                }
                wandb.log(log_dict, step=global_step)

    envs.close()
    print(f"\nBest average reward: {best_avg_reward:.2f}")
    return model
