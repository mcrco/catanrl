#!/usr/bin/env python3
"""
Multi-agent self-play training script using the PettingZoo AEC environment.

The script mirrors the structure of rl/train_sl.py but replaces supervised learning
with PPO-based reinforcement learning where all agents share the same policy/value
network and learn via self-play in the CatanatronAECEnv.
"""

import os
import random
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb
from tqdm import tqdm

from catanatron.gym.envs.catanatron_env import ACTION_SPACE_SIZE

from .buffers import MultiAgentExperienceBuffer
from ...envs.multi_env import (
    build_marl_env,
    compute_multiagent_input_dim,
    flatten_marl_observation,
    get_valid_actions,
)
from ...models.models import (
    build_flat_policy_value_network,
    build_hierarchical_policy_value_network,
)
from ...models.backbones import BackboneConfig, MLPBackboneConfig
from .utils import compute_gae


class MARLAgent:
    """Shared policy/value agent for multi-agent self-play training."""

    def __init__(
        self,
        model: nn.Module,
        model_type: str,
        device: str,
    ):
        self.model = model
        self.model_type = model_type
        self.device = device

    def select_action(
        self,
        state_vec: np.ndarray,
        valid_actions: np.ndarray,
        deterministic: bool = False,
    ) -> Tuple[int, float, float]:
        """Select an action for the current PettingZoo agent."""
        if valid_actions.size == 0:
            valid_actions = np.arange(ACTION_SPACE_SIZE, dtype=np.int64)

        state_tensor = torch.from_numpy(state_vec).float().unsqueeze(0).to(self.device)
        self.model.eval()
        with torch.no_grad():
            if self.model_type == "flat":
                policy_logits, value = self.model(state_tensor)
            elif self.model_type == "hierarchical":
                action_type_logits, param_logits, value = self.model(state_tensor)
                policy_logits = self.model.get_flat_action_logits(action_type_logits, param_logits)

            mask = torch.full_like(policy_logits, float("-inf"))
            mask[0, valid_actions] = 0.0
            masked_logits = torch.clamp(policy_logits + mask, min=-100, max=100)

            probs = torch.softmax(masked_logits, dim=-1)
            log_probs_all = torch.log_softmax(masked_logits, dim=-1)

            if deterministic:
                action_idx = torch.argmax(probs[0, valid_actions]).item()
                action = int(valid_actions[action_idx])
            else:
                valid_probs = probs[0, valid_actions]
                valid_probs = valid_probs / (valid_probs.sum() + 1e-12)
                dist = torch.distributions.Categorical(valid_probs)
                action = int(valid_actions[dist.sample().item()])

            log_prob = float(log_probs_all[0, action].item())
            value_out = float(value.item())

        return action, log_prob, value_out

    def evaluate_actions(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        valid_actions_batch: List[np.ndarray],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate log-probs, values, and entropy for PPO updates."""
        if self.model_type == "flat":
            policy_logits, values = self.model(states)
        elif self.model_type == "hierarchical":
            action_type_logits, param_logits, values = self.model(states)
            policy_logits = self.model.get_flat_action_logits(action_type_logits, param_logits)
        batch_size, num_actions = policy_logits.shape

        masks = torch.full((batch_size, num_actions), float("-inf"), device=states.device)
        for idx, valid in enumerate(valid_actions_batch):
            if valid.size == 0:
                masks[idx] = 0.0
            else:
                masks[idx, valid] = 0.0

        masked_logits = torch.clamp(policy_logits + masks, min=-100, max=100)
        log_probs_all = torch.log_softmax(masked_logits, dim=-1)
        probs = torch.softmax(masked_logits, dim=-1)

        log_probs = log_probs_all.gather(1, actions.unsqueeze(1)).squeeze(1)
        entropy = -(probs * log_probs_all).sum(dim=-1)
        return log_probs, values.squeeze(-1), entropy


@dataclass
class PendingExperience:
    state: np.ndarray
    action: int
    value: float
    log_prob: float
    valid_actions: np.ndarray


def set_global_seeds(seed: Optional[int]):
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ppo_update(
    agent: MARLAgent,
    optimizer: torch.optim.Optimizer,
    buffer: MultiAgentExperienceBuffer,
    clip_epsilon: float,
    value_coef: float,
    entropy_coef: float,
    n_epochs: int,
    batch_size: int,
    device: str,
    gamma: float,
    gae_lambda: float,
    max_grad_norm: float,
) -> Dict[str, float]:
    """Perform PPO update for multi-agent experiences."""
    if len(buffer) == 0:
        return {"policy_loss": 0.0, "value_loss": 0.0, "entropy_loss": 0.0, "total_loss": 0.0}

    states, actions, rewards, old_values, old_log_probs, valid_actions, dones = buffer.get()

    agent.model.eval()
    with torch.no_grad():
        last_state = torch.from_numpy(states[-1:]).float().to(device)
        if agent.model_type == "flat":
            _, next_value_tensor = agent.model(last_state)
        elif agent.model_type == "hierarchical":
            action_type_logits, param_logits, next_value_tensor = agent.model(last_state)
            next_value_tensor = agent.model.get_flat_action_logits(action_type_logits, param_logits)
        next_value = float(next_value_tensor.item())
    agent.model.train()

    advantages, returns = compute_gae(rewards, old_values, dones, next_value, gamma, gae_lambda)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    states_t = torch.from_numpy(states).float().to(device)
    actions_t = torch.from_numpy(actions).long().to(device)
    old_log_probs_t = torch.from_numpy(old_log_probs).float().to(device)
    advantages_t = torch.from_numpy(advantages).float().to(device)
    returns_t = torch.from_numpy(returns).float().to(device)

    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_entropy = 0.0
    total_loss = 0.0
    n_updates = 0

    for _ in range(n_epochs):
        indices = np.random.permutation(len(states))
        for start in range(0, len(states), batch_size):
            end = start + batch_size
            batch_idx = indices[start:end]
            if len(batch_idx) < 2:
                continue

            batch_states = states_t[batch_idx]
            batch_actions = actions_t[batch_idx]
            batch_old_log_probs = old_log_probs_t[batch_idx]
            batch_advantages = advantages_t[batch_idx]
            batch_returns = returns_t[batch_idx]
            batch_valid_actions: List[np.ndarray] = [valid_actions[i] for i in batch_idx]

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
                torch.isnan(param.grad).any()
                for param in agent.model.parameters()
                if param.grad is not None
            )
            if has_nan_grad:
                optimizer.zero_grad()
                continue

            optimizer.step()

            total_policy_loss += float(policy_loss.item())
            total_value_loss += float(value_loss.item())
            total_entropy += float(entropy_loss.item())
            total_loss += float(loss.item())
            n_updates += 1

    if n_updates == 0:
        return {"policy_loss": 0.0, "value_loss": 0.0, "entropy_loss": 0.0, "total_loss": 0.0}

    return {
        "policy_loss": total_policy_loss / n_updates,
        "value_loss": total_value_loss / n_updates,
        "entropy_loss": total_entropy / n_updates,
        "total_loss": total_loss / n_updates,
    }


def train(
    num_players: int = 2,
    map_type: str = "BASE",
    model_type: str = "flat",
    vps_to_win: int = 10,
    representation: str = "vector",
    episodes: int = 500,
    rollout_steps: int = 4096,
    lr: float = 3e-4,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_epsilon: float = 0.2,
    value_coef: float = 0.5,
    entropy_coef: float = 0.01,
    ppo_epochs: int = 4,
    batch_size: int = 256,
    hidden_dims: List[int] = (512, 512),
    save_path: str = "weights/policy_value_marl.pt",
    save_freq: int = 50,
    load_weights: Optional[str] = None,
    wandb_config: Optional[Dict] = None,
    invalid_action_reward: float = -1.0,
    max_invalid_actions: int = 10,
    seed: Optional[int] = 42,
    max_grad_norm: float = 0.5,
    deterministic_policy: bool = False,
) -> nn.Module:
    """Run PPO-based self-play training."""

    assert 2 <= num_players <= 4, "num_players must be between 2 and 4"
    assert representation in ("vector", "mixed"), "representation must be vector or mixed"

    set_global_seeds(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    input_dim, board_shape, numeric_dim = compute_multiagent_input_dim(
        num_players, map_type, representation
    )
    print(f"\n{'=' * 60}")
    print("Multi-Agent Self-Play Training (PPO)")
    print(f"{'=' * 60}")
    print(f"Device: {device}")
    print(f"Map type: {map_type} | Players: {num_players} | Representation: {representation}")
    print(f"Input dimension: {input_dim}")
    if representation == "mixed" and board_shape:
        print(f"  Numeric features: {numeric_dim}")
        print(f"  Board tensor shape: {board_shape}")
    print(f"Episodes: {episodes} | Rollout steps: {rollout_steps}")

    env_config = {
        "num_players": num_players,
        "map_type": map_type,
        "vps_to_win": vps_to_win,
        "representation": representation,
        "invalid_action_reward": invalid_action_reward,
        "max_invalid_actions": max_invalid_actions,
    }
    env = build_marl_env(env_config)

    if wandb_config:
        wandb.init(**wandb_config)
        print(
            f"Initialized wandb: {wandb_config['project']}/{wandb_config.get('name', wandb.run.name)}"
        )
    else:
        wandb.init(mode="disabled")

    backbone_config = BackboneConfig(
        architecture="mlp", args=MLPBackboneConfig(input_dim=input_dim, hidden_dims=hidden_dims)
    )
    if model_type == "flat":
        model = build_flat_policy_value_network(
            backbone_config=backbone_config, num_actions=ACTION_SPACE_SIZE
        ).to(device)
    elif model_type == "hierarchical":
        model = build_hierarchical_policy_value_network(backbone_config=backbone_config).to(device)
    if load_weights:
        if os.path.exists(load_weights):
            print(f"Loading weights from {load_weights}")
            state_dict = torch.load(load_weights, map_location=device)
            model.load_state_dict(state_dict)
        else:
            print(f"Warning: weights file {load_weights} not found, starting from scratch.")

    agent = MARLAgent(model, model_type, device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    buffer = MultiAgentExperienceBuffer()
    recent_rewards = deque(maxlen=100)
    recent_wins = deque(maxlen=100)
    best_win_rate = -float("inf")
    global_step = 0
    primary_agent = env.possible_agents[0]

    try:
        for episode in tqdm(range(1, episodes + 1), desc="Episodes"):
            env.reset(seed=None if seed is None else seed + episode)
            pending: Dict[str, Optional[PendingExperience]] = {
                agent_name: None for agent_name in env.possible_agents
            }
            episode_returns = defaultdict(float)
            episode_steps = 0

            for agent_name in env.agent_iter():
                observation, reward, termination, truncation, info = env.last()
                done = termination or truncation

                if pending[agent_name] is not None:
                    exp = pending[agent_name]
                    buffer.add(
                        exp.state,
                        exp.action,
                        reward,
                        exp.value,
                        exp.log_prob,
                        exp.valid_actions,
                        done,
                    )
                    episode_returns[agent_name] += reward
                    pending[agent_name] = None
                    global_step += 1
                    episode_steps += 1

                if done:
                    env.step(None)
                    continue

                obs_vector = flatten_marl_observation(observation, representation)
                valid_actions = get_valid_actions(info)
                action, log_prob, value = agent.select_action(
                    obs_vector, valid_actions, deterministic=deterministic_policy
                )
                pending[agent_name] = PendingExperience(
                    state=obs_vector.copy(),
                    action=action,
                    value=value,
                    log_prob=log_prob,
                    valid_actions=valid_actions.copy(),
                )
                env.step(action)

            winner_color = env.game.winning_color() if env.game else None
            winner_agent = (
                env.color_to_agent_name[winner_color] if winner_color is not None else None
            )
            win_flag = 1 if winner_agent == primary_agent else 0
            recent_wins.append(win_flag)
            primary_return = episode_returns.get(primary_agent, 0.0)
            recent_rewards.append(primary_return)

            avg_reward_window = np.mean(recent_rewards) if recent_rewards else 0.0
            win_rate_window = np.mean(recent_wins) if recent_wins else 0.0

            wandb.log(
                {
                    "episode": episode,
                    "episode/steps": episode_steps,
                    f"episode/return/{primary_agent}": primary_return,
                    "episode/return_mean": np.mean(list(episode_returns.values()))
                    if episode_returns
                    else 0.0,
                    "episode/win_flag": win_flag,
                    "episode/avg_reward_window": avg_reward_window,
                    "episode/win_rate_window": win_rate_window,
                    "global_step": global_step,
                },
                step=global_step,
            )

            if len(buffer) >= max(batch_size * 2, rollout_steps):
                metrics = ppo_update(
                    agent=agent,
                    optimizer=optimizer,
                    buffer=buffer,
                    clip_epsilon=clip_epsilon,
                    value_coef=value_coef,
                    entropy_coef=entropy_coef,
                    n_epochs=ppo_epochs,
                    batch_size=batch_size,
                    device=device,
                    gamma=gamma,
                    gae_lambda=gae_lambda,
                    max_grad_norm=max_grad_norm,
                )
                buffer.clear()
                agent.model.eval()

                wandb.log(
                    {
                        "train/policy_loss": metrics["policy_loss"],
                        "train/value_loss": metrics["value_loss"],
                        "train/entropy": -metrics["entropy_loss"],
                        "train/total_loss": metrics["total_loss"],
                    },
                    step=global_step,
                )

            if (
                len(recent_wins) == recent_wins.maxlen
                and win_rate_window > best_win_rate
                and save_path
            ):
                best_win_rate = win_rate_window
                torch.save(model.state_dict(), save_path)
                print(
                    f"  → Saved best model (window win rate: {win_rate_window:.2%}, avg reward: {avg_reward_window:.2f})"
                )
                wandb.run.summary["best_win_rate"] = best_win_rate

            if save_path and episode % save_freq == 0:
                torch.save(model.state_dict(), save_path)
                print(f"  → Checkpoint saved at {save_path}")

    finally:
        env.close()

    print(f"\nTraining complete. Best window win rate: {best_win_rate:.2%}")
    return model
