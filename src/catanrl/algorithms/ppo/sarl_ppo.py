#!/usr/bin/env python3
"""
Single Agent Reinforcement Learning training script for Catan Policy-Value Network.
Uses the Catanatron gym environment for training with PPO.
"""

import os
from collections import deque
from typing import List, Dict

import numpy as np
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn.functional as F
import wandb
from tqdm import tqdm

from catanatron.gym.envs.catanatron_env import ACTION_SPACE_SIZE

from .buffers import OnPolicyBuffer
from ...agents import SARLAgent
from ...envs.single_env import create_opponents, make_vectorized_envs
from ...models.models import PolicyValueNetwork, HierarchicalPolicyValueNetwork
from .utils import compute_gae


ESTIMATED_STEPS_PER_EPISODE = 100

def ppo_update(
    agent: SARLAgent,
    optimizer: torch.optim.Optimizer,
    buffer: OnPolicyBuffer,
    clip_epsilon: float,
    value_coef: float,
    entropy_coef: float,
    n_epochs: int,
    batch_size: int,
    device: str,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    max_grad_norm: float = 0.5,
) -> Dict[str, float]:
    """
    Run PPO updates over the collected on-policy buffer for SARL training.
    """
    if len(buffer) == 0:
        return {"policy_loss": 0.0, "value_loss": 0.0, "entropy_loss": 0.0, "total_loss": 0.0}

    states, actions, rewards, old_values, old_log_probs, valid_actions, dones = buffer.get()

    agent.model.eval()
    with torch.no_grad():
        next_state = torch.from_numpy(states[-1:]).float().to(device)
        if agent.model_type == 'flat':
            _, next_value = agent.model(next_state)
        elif agent.model_type == 'hierarchical':
            _, _, next_value = agent.model(next_state)
        next_value = next_value.item()
    agent.model.train()

    advantages, returns = compute_gae(rewards, old_values, dones, next_value, gamma, gae_lambda)
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
                continue

            batch_states = states_t[batch_indices]
            batch_actions = actions_t[batch_indices]
            batch_old_log_probs = old_log_probs_t[batch_indices]
            batch_advantages = advantages_t[batch_indices]
            batch_returns = returns_t[batch_indices]
            batch_valid_actions = [valid_actions[i] for i in batch_indices]

            log_probs, values, entropy = agent.evaluate_actions(
                batch_states, batch_actions, batch_valid_actions
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
                optimizer.zero_grad()
                continue

            optimizer.step()

            total_policy_loss += float(policy_loss.item())
            total_value_loss += float(value_loss.item())
            total_entropy_loss += float(entropy_loss.item())
            total_loss += float(loss.item())
            n_updates += 1

    if n_updates == 0:
        return {"policy_loss": 0.0, "value_loss": 0.0, "entropy_loss": 0.0, "total_loss": 0.0}

    return {
        "policy_loss": total_policy_loss / n_updates,
        "value_loss": total_value_loss / n_updates,
        "entropy_loss": total_entropy_loss / n_updates,
        "total_loss": total_loss / n_updates,
    }


def train(
    input_dim: int,
    num_actions: int = ACTION_SPACE_SIZE,
    model_type: str = 'flat',
    hidden_dims: List[int] = (512, 512),
    load_weights: str | None = None,
    n_episodes: int = 1000,
    update_freq: int = 32,
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
    map_type: str = "BASE",
    opponent_configs: List[str] | None = None,
    num_envs: int = 4,
    use_lr_scheduler: bool = False,
    lr_scheduler_kwargs: dict | None = None,
):
    """Train PolicyValueNetwork using single-agent PPO."""

    print(f"\n{'=' * 60}")
    print("Training Policy-Value Network with Single Agent Reinforcement Learning (PPO)")
    print(f"{'=' * 60}")
    print(f"Device: {device}")
    print(f"Map type: {map_type}")
    print(f"Episodes: {n_episodes}")
    print(f"Update frequency: {update_freq} episodes")
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

    if model_type == 'flat':
        model = PolicyValueNetwork(input_dim, num_actions, list(hidden_dims)).to(device)
    elif model_type == 'hierarchical':
        model = HierarchicalPolicyValueNetwork(input_dim, list(hidden_dims)).to(device)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    if load_weights and os.path.exists(load_weights):
        print(f"Loading weights from: {load_weights}")
        state_dict = torch.load(load_weights, map_location=device)
        model.load_state_dict(state_dict)
        print("  ✓ Weights loaded successfully")

    agent = SARLAgent(model, model_type, device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Create learning rate scheduler
    scheduler = None
    if use_lr_scheduler:
        scheduler_kwargs = lr_scheduler_kwargs or {}
        start_factor = scheduler_kwargs.get("start_factor", 1.0)
        end_factor = scheduler_kwargs.get("end_factor", 0.0)
        total_iters = scheduler_kwargs.get("total_iters", n_episodes)
        scheduler = lr_scheduler.LinearLR(
            optimizer, start_factor=start_factor, end_factor=end_factor, total_iters=total_iters
        )
        print(f"LR Scheduler: LinearLR (start_factor={start_factor}, end_factor={end_factor}, total_iters={total_iters})")
    else:
        print("LR Scheduler: None (constant learning rate)")
    
    model.eval()

    buffer = OnPolicyBuffer()
    episode_rewards = deque(maxlen=100)
    episode_lengths = deque(maxlen=100)
    best_avg_reward = float("-inf")
    global_step = 0
    total_episodes = 0

    envs = make_vectorized_envs(
        map_type=map_type,
        opponent_configs=opponent_configs,
        num_envs=num_envs,
        representation="mixed",
    )

    env_episode_rewards = np.zeros(num_envs)
    env_episode_lengths = np.zeros(num_envs, dtype=int)
    episode_start = np.zeros(num_envs, dtype=bool)

    print("\nStarting training...")
    observations, infos = envs.reset()

    with tqdm(total=n_episodes, desc="Episodes") as pbar:
        while total_episodes < n_episodes:
            numeric_obs = observations["numeric"].astype(np.float32)
            board_obs = observations["board"].astype(np.float32)  # [N, C, H, W]
            board_obs_ch_last = np.transpose(board_obs, (0, 2, 3, 1))
            board_flat = board_obs_ch_last.reshape(board_obs_ch_last.shape[0], -1)
            states = np.concatenate([numeric_obs, board_flat], axis=1)

            info_valid = infos.get("valid_actions")
            if info_valid is None:
                valid_actions_list = [
                    np.arange(ACTION_SPACE_SIZE, dtype=np.int64) for _ in range(num_envs)
                ]
            else:
                valid_actions_list = []
                for v in info_valid:
                    arr = np.array(v, dtype=np.int64)
                    if arr.size == 0:
                        arr = np.arange(ACTION_SPACE_SIZE, dtype=np.int64)
                    valid_actions_list.append(arr)

            states_t = torch.from_numpy(states).float().to(device)
            actions, log_probs, values = agent.select_actions_batch(
                states_t, valid_actions_list
            )

            next_observations, rewards, terminations, truncations, infos = envs.step(actions)

            for env_idx in range(num_envs):
                if not episode_start[env_idx]:
                    buffer.add(
                        states[env_idx],
                        actions[env_idx],
                        rewards[env_idx],
                        values[env_idx],
                        log_probs[env_idx],
                        valid_actions_list[env_idx],
                        bool(terminations[env_idx] or truncations[env_idx]),
                    )
                    env_episode_rewards[env_idx] += rewards[env_idx]
                    env_episode_lengths[env_idx] += 1
                    global_step += 1

                if terminations[env_idx] or truncations[env_idx]:
                    episode_rewards.append(env_episode_rewards[env_idx])
                    episode_lengths.append(env_episode_lengths[env_idx])
                    total_episodes += 1
                    pbar.update(1)

                    avg_reward = np.mean(episode_rewards)
                    avg_length = np.mean(episode_lengths)

                    if total_episodes % 10 == 0:
                        print(
                            f"\nEpisode {total_episodes}/{n_episodes} | "
                            f"Reward: {env_episode_rewards[env_idx]:.2f} | "
                            f"Avg(100): {avg_reward:.2f} | "
                            f"Length: {env_episode_lengths[env_idx]}"
                        )

                    wandb.log(
                        {
                            "episode/reward": env_episode_rewards[env_idx],
                            "episode/length": env_episode_lengths[env_idx],
                            "episode/avg_reward": avg_reward,
                            "episode/avg_length": avg_length,
                            "episode": total_episodes,
                        },
                        step=global_step,
                    )

                    env_episode_rewards[env_idx] = 0
                    env_episode_lengths[env_idx] = 0
                    episode_start[env_idx] = True

                    min_episodes_for_best = min(update_freq, 100)
                    if len(episode_rewards) >= min_episodes_for_best and avg_reward > best_avg_reward:
                        best_avg_reward = avg_reward
                        torch.save(model.state_dict(), save_path)
                        print(f"  → Saved best model (avg_reward: {avg_reward:.2f})")
                        if wandb.run is not None:
                            wandb.run.summary["best_avg_reward"] = best_avg_reward

                    if total_episodes % save_freq == 0:
                        save_dir = os.path.dirname(save_path)
                        checkpoint_path = (
                            os.path.join(save_dir, f"checkpoint_ep{total_episodes}.pt")
                            if save_dir
                            else f"checkpoint_ep{total_episodes}.pt"
                        )
                        torch.save(model.state_dict(), checkpoint_path)
                        print(f"  → Saved checkpoint: {checkpoint_path}")
                else:
                    episode_start[env_idx] = False

            observations = next_observations

            if len(buffer) >= update_freq * ESTIMATED_STEPS_PER_EPISODE:
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
                    gamma=gamma,
                    gae_lambda=gae_lambda,
                )
                buffer.clear()
                model.eval()

                # Step learning rate scheduler
                current_lr = optimizer.param_groups[0]["lr"]
                if scheduler is not None:
                    scheduler.step()
                    new_lr = optimizer.param_groups[0]["lr"]
                    if new_lr != current_lr:
                        print(f"  → LR updated: {current_lr:.6f} → {new_lr:.6f}")

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

