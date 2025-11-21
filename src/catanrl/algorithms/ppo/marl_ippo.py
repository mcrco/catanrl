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
import torch.optim as optim
import wandb
from tqdm import tqdm

from catanatron.gym.envs.catanatron_env import ACTION_SPACE_SIZE

from ...agents import MultiAgentRLAgent
from .buffers import MultiAgentExperienceBuffer
from .trainer_marl import ppo_update
from ...envs.multi_env import (
    build_marl_env,
    compute_multiagent_input_dim,
    flatten_marl_observation,
    get_valid_actions,
)
from ...models.models import PolicyValueNetwork


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


def train(
    num_players: int = 2,
    map_type: str = "BASE",
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
) -> PolicyValueNetwork:
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

    model = PolicyValueNetwork(input_dim, ACTION_SPACE_SIZE, list(hidden_dims)).to(device)
    if load_weights:
        if os.path.exists(load_weights):
            print(f"Loading weights from {load_weights}")
            state_dict = torch.load(load_weights, map_location=device)
            model.load_state_dict(state_dict)
        else:
            print(f"Warning: weights file {load_weights} not found, starting from scratch.")

    agent = MultiAgentRLAgent(model, device)
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

