#!/usr/bin/env python3
"""
Dataset Aggregation (DAgger) imitation-learning trainer for the Catan RL agent.

This module follows the same conventions as the PPO and supervised trainers:
  * Builds Policy/Value networks from shared backbone configs
  * Exposes a single `train(...)` entrypoint consumed by experiment scripts
  * Supports both flat and hierarchical policy heads
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import wandb

from catanatron.gym.envs.catanatron_env import ACTION_SPACE_SIZE, to_action_space
from catanatron.models.player import Color, Player, RandomPlayer
from catanatron.players.mcts import MCTSPlayer
from catanatron.players.minimax import AlphaBetaPlayer, SameTurnAlphaBetaPlayer
from catanatron.players.playouts import GreedyPlayoutsPlayer
from catanatron.players.search import VictoryPointPlayer
from catanatron.players.value import ValueFunctionPlayer
from catanatron.players.weighted_random import WeightedRandomPlayer

from ...envs.gym.single_env import create_opponents
from ...models.backbones import BackboneConfig, MLPBackboneConfig
from ...models.models import (
    build_flat_policy_value_network,
    build_hierarchical_policy_value_network,
)
from ...algorithms.supervised.loss_utils import create_loss_computer


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


def _select_policy_action(
    model: nn.Module,
    state_vec: np.ndarray,
    valid_actions: np.ndarray,
    model_type: str,
    device: torch.device,
    deterministic: bool = False,
) -> int:
    """Sample or pick the highest-probability action under the current policy."""
    state_t = torch.from_numpy(state_vec).float().unsqueeze(0).to(device)
    was_training = model.training
    model.eval()

    with torch.no_grad():
        if model_type == "flat":
            policy_logits, _ = model(state_t)
        elif model_type == "hierarchical":
            action_type_logits, param_logits, _ = model(state_t)
            policy_logits = model.get_flat_action_logits(action_type_logits, param_logits)
        else:  # pragma: no cover - defensive
            raise ValueError(f"Unknown model_type '{model_type}'")

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
        model.train()
    return action_idx


def _query_expert_action(env: Any, expert: Player) -> int:
    """Ask the expert policy for the optimal action in the current game state."""
    playable_actions = list(getattr(env.game.state, "playable_actions", []))
    if not playable_actions:
        return 0

    try:
        expert_action = expert.decide(env.game, playable_actions)
    except Exception:
        expert_action = playable_actions[0]

    try:
        expert_idx = to_action_space(expert_action)
    except ValueError:
        expert_idx = to_action_space(playable_actions[0])
    return expert_idx


def _evaluate_policy(
    map_type: str,
    opponent_configs: Sequence[str],
    model: nn.Module,
    model_type: str,
    device: torch.device,
    episodes: int,
    seed: int,
) -> Dict[str, float]:
    """Roll deterministic episodes to track win-rate and reward."""
    opponents = create_opponents(list(opponent_configs))
    eval_env = gym.make(
        "catanatron/Catanatron-v0",
        config={
            "map_type": map_type,
            "enemies": opponents,
            "representation": "mixed",
        },
    )

    episode_rewards: List[float] = []
    episode_lengths: List[int] = []
    wins = 0

    for ep in range(episodes):
        obs, info = eval_env.reset(seed=seed + ep)
        done = False
        ep_reward = 0.0
        ep_length = 0

        while not done:
            state_vec = _observation_to_state_vector(obs)
            valid_actions = _extract_valid_actions(info)
            action = _select_policy_action(
                model, state_vec, valid_actions, model_type, device, deterministic=True
            )
            obs, reward, terminated, truncated, info = eval_env.step(action)
            ep_reward += float(reward)
            ep_length += 1
            done = bool(terminated or truncated)

        episode_rewards.append(ep_reward)
        episode_lengths.append(ep_length)
        if getattr(eval_env, "game", None) and eval_env.game.winning_color() == Color.BLUE:
            wins += 1

    eval_env.close()

    return {
        "avg_reward": float(np.mean(episode_rewards)) if episode_rewards else 0.0,
        "avg_length": float(np.mean(episode_lengths)) if episode_lengths else 0.0,
        "win_rate": wins / episodes if episodes > 0 else 0.0,
    }


class AggregatedDataset(Dataset):
    """Simple in-memory dataset for aggregated DAgger trajectories."""

    def __init__(self, input_dim: int, max_size: int | None = None):
        self.input_dim = input_dim
        self.max_size = max_size
        self.states: List[np.ndarray] = []
        self.actions: List[int] = []
        self.returns: List[float] = []

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            torch.from_numpy(self.states[idx]),
            torch.tensor(self.actions[idx], dtype=torch.long),
            torch.tensor(self.returns[idx], dtype=torch.float32),
        )

    def add_episode(
        self,
        states: List[np.ndarray],
        expert_actions: List[int],
        rewards: List[float],
        gamma: float,
    ) -> None:
        returns = _discount_rewards(rewards, gamma)
        for state, action, ret in zip(states, expert_actions, returns):
            self._add_sample(state, action, ret)

    def _add_sample(self, state: np.ndarray, action: int, ret: float) -> None:
        vec = np.asarray(state, dtype=np.float32).reshape(-1)
        if vec.shape[0] != self.input_dim:
            raise ValueError(f"State dim mismatch: expected {self.input_dim}, got {vec.shape[0]}")
        self.states.append(vec)
        self.actions.append(int(action))
        self.returns.append(float(ret))
        self._truncate_if_needed()

    def _truncate_if_needed(self) -> None:
        if self.max_size is None:
            return
        overflow = len(self.states) - self.max_size
        if overflow > 0:
            self.states = self.states[overflow:]
            self.actions = self.actions[overflow:]
            self.returns = self.returns[overflow:]


def _create_expert_player(config: str) -> Player:
    """Parse CLI-like config string into a Player that labels actions for BLUE."""
    color = Color.BLUE
    parts = config.split(":")
    player_type = parts[0].lower()
    params = parts[1:]

    try:
        if player_type in {"random", "r"}:
            return RandomPlayer(color)
        if player_type in {"weighted", "w"}:
            return WeightedRandomPlayer(color)
        if player_type in {"value", "f"}:
            return ValueFunctionPlayer(color)
        if player_type in {"victorypoint", "vp"}:
            return VictoryPointPlayer(color)
        if player_type in {"alphabeta", "ab"}:
            depth = int(params[0]) if params else 2
            pruning = params[1].lower() != "false" if len(params) > 1 else True
            return AlphaBetaPlayer(color, depth=depth, prunning=pruning)
        if player_type in {"sameturnalphabeta", "sab"}:
            return SameTurnAlphaBetaPlayer(color)
        if player_type in {"mcts", "m"}:
            simulations = int(params[0]) if params else 200
            return MCTSPlayer(color, simulations)
        if player_type in {"playouts", "greedyplayouts", "g"}:
            num_playouts = int(params[0]) if params else 25
            return GreedyPlayoutsPlayer(color, num_playouts)
    except Exception as exc:  # pragma: no cover - defensive
        print(f"[DAgger] Failed to build expert '{config}': {exc}. Falling back to RandomPlayer.")

    print(f"[DAgger] Unknown expert '{config}', defaulting to RandomPlayer.")
    return RandomPlayer(color)


@dataclass
class DAggerCollectStats:
    episodes: int
    steps: int
    mean_reward: float
    expert_fraction: float
    dataset_size: int


def _collect_dagger_rollouts(
    env: gym.Env,
    model: nn.Module,
    dataset: AggregatedDataset,
    expert: Player,
    episodes: int,
    beta: float,
    gamma: float,
    model_type: str,
    device: torch.device,
    seed: int,
) -> DAggerCollectStats:
    """Run policy/environment interaction and aggregate expert-labeled data."""
    episode_rewards: List[float] = []
    steps_collected = 0
    expert_actions_used = 0
    total_actions = 0

    for ep in range(episodes):
        expert.reset_state()
        obs, info = env.reset(seed=seed + ep)
        done = False
        ep_states: List[np.ndarray] = []
        ep_labels: List[int] = []
        ep_rewards: List[float] = []

        while not done:
            state_vec = _observation_to_state_vector(obs)
            valid_actions = _extract_valid_actions(info)
            expert_action = _query_expert_action(env, expert)
            policy_action = _select_policy_action(
                model, state_vec, valid_actions, model_type, device, deterministic=False
            )

            if random.random() < beta:
                action_to_play = expert_action
                expert_actions_used += 1
            else:
                action_to_play = policy_action

            total_actions += 1
            ep_states.append(state_vec)
            ep_labels.append(expert_action)

            obs, reward, terminated, truncated, info = env.step(int(action_to_play))
            ep_rewards.append(float(reward))
            steps_collected += 1
            done = bool(terminated or truncated)

        dataset.add_episode(ep_states, ep_labels, ep_rewards, gamma)
        episode_rewards.append(float(np.sum(ep_rewards)))

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
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_computer,
    epochs: int,
    batch_size: int,
    model_type: str,
    device: torch.device,
) -> Dict[str, float]:
    """Run supervised updates on the aggregated dataset."""
    if len(dataset) == 0 or epochs <= 0:
        return {"total_loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0, "accuracy": 0.0}

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    model.train()

    total_loss = 0.0
    total_policy_loss = 0.0
    total_value_loss = 0.0
    correct = 0
    total_samples = 0

    for _ in range(epochs):
        for features, actions, returns in loader:
            features = features.to(device)
            actions = actions.to(device)
            returns = returns.to(device)

            optimizer.zero_grad()

            if model_type == "flat":
                policy_logits, value_pred = model(features)
                batch_total_loss, policy_loss, value_loss = loss_computer.compute_flat_loss(
                    policy_logits, value_pred.squeeze(-1), actions, returns
                )
                logits_for_metrics = policy_logits
            elif model_type == "hierarchical":
                action_type_logits, param_logits, value_pred = model(features)
                (
                    batch_total_loss,
                    policy_loss,
                    _,
                    value_loss,
                ) = loss_computer.compute_hierarchical_loss(
                    action_type_logits,
                    param_logits,
                    value_pred.squeeze(-1),
                    actions,
                    returns,
                    model.flat_to_hierarchical,
                )
                logits_for_metrics = model.get_flat_action_logits(action_type_logits, param_logits)
            else:  # pragma: no cover
                raise ValueError(f"Unknown model_type '{model_type}'")

            batch_total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            total_loss += float(batch_total_loss.item())
            total_policy_loss += float(policy_loss.item())
            total_value_loss += float(value_loss.item())

            predictions = torch.argmax(logits_for_metrics, dim=-1)
            correct += int((predictions == actions).sum().item())
            total_samples += actions.size(0)

    n_updates = max(len(loader) * epochs, 1)
    accuracy = correct / total_samples if total_samples else 0.0
    return {
        "total_loss": total_loss / n_updates,
        "policy_loss": total_policy_loss / n_updates,
        "value_loss": total_value_loss / n_updates,
        "accuracy": accuracy,
    }


def train(
    input_dim: int,
    num_actions: int = ACTION_SPACE_SIZE,
    model_type: str = "flat",
    hidden_dims: Sequence[int] = (512, 512),
    load_weights: str | None = None,
    n_iterations: int = 10,
    episodes_per_iteration: int = 5,
    train_epochs: int = 1,
    batch_size: int = 256,
    lr: float = 1e-4,
    gamma: float = 0.99,
    expert_config: str = "AB:2",
    opponent_configs: Sequence[str] | None = None,
    map_type: str = "BASE",
    beta_init: float = 1.0,
    beta_decay: float = 0.9,
    beta_min: float = 0.05,
    max_dataset_size: int | None = None,
    save_path: str | None = None,
    device: str | torch.device | None = None,
    wandb_config: Dict[str, Any] | None = None,
    eval_episodes: int = 5,
    seed: int = 42,
) -> nn.Module:
    """
    Train the policy-value network with DAgger.

    Args mirror other training utilities for consistency.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        device = torch.device(device)

    opponent_configs = list(opponent_configs) if opponent_configs else ["random"]
    save_path = save_path or "weights/policy_value_dagger.pt"

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if wandb_config:
        wandb.init(**wandb_config)
    else:
        wandb.init(mode="disabled")

    backbone_config = BackboneConfig(
        architecture="mlp",
        args=MLPBackboneConfig(input_dim=input_dim, hidden_dims=list(hidden_dims)),
    )

    if model_type == "flat":
        model = build_flat_policy_value_network(
            backbone_config=backbone_config, num_actions=num_actions
        ).to(device)
    elif model_type == "hierarchical":
        model = build_hierarchical_policy_value_network(backbone_config=backbone_config).to(device)
    else:
        raise ValueError(f"Unknown model_type '{model_type}'")

    if load_weights:
        if os.path.exists(load_weights):
            state = torch.load(load_weights, map_location=device)
            model.load_state_dict(state)
            print(f"[DAgger] Loaded weights from {load_weights}")
        else:
            print(f"[DAgger] Warning: weights file '{load_weights}' not found.")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_computer = create_loss_computer(
        model=model,
        train_actions=None,
        policy_weight=1.0,
        value_weight=1.0,
        action_type_weight=1.0,
        param_weight=1.0,
        use_class_weights=False,
        device=str(device),
    )

    env = gym.make(
        "catanatron/Catanatron-v0",
        config={
            "map_type": map_type,
            "enemies": create_opponents(opponent_configs),
            "representation": "mixed",
        },
    )
    expert = _create_expert_player(expert_config)
    dataset = AggregatedDataset(input_dim=input_dim, max_size=max_dataset_size)

    beta = beta_init
    best_eval_reward = float("-inf")

    for iteration in range(1, n_iterations + 1):
        print(f"\n[DAgger] Iteration {iteration}/{n_iterations} | beta={beta:.3f}")
        collect_stats = _collect_dagger_rollouts(
            env=env,
            model=model,
            dataset=dataset,
            expert=expert,
            episodes=episodes_per_iteration,
            beta=beta,
            gamma=gamma,
            model_type=model_type,
            device=device,
            seed=seed + iteration * 100,
        )
        print(
            f"  Collected {collect_stats.steps} steps "
            f"({collect_stats.episodes} episodes, mean reward {collect_stats.mean_reward:.3f})"
        )
        print(
            f"  Expert executed {collect_stats.expert_fraction * 100:.1f}% of actions | "
            f"Dataset size: {collect_stats.dataset_size}"
        )

        train_stats = _train_on_dataset(
            dataset=dataset,
            model=model,
            optimizer=optimizer,
            loss_computer=loss_computer,
            epochs=train_epochs,
            batch_size=batch_size,
            model_type=model_type,
            device=device,
        )
        print(
            f"  Train loss {train_stats['total_loss']:.4f} | "
            f"Policy {train_stats['policy_loss']:.4f} | "
            f"Value {train_stats['value_loss']:.4f} | "
            f"Accuracy {train_stats['accuracy'] * 100:.2f}%"
        )

        eval_stats = _evaluate_policy(
            map_type=map_type,
            opponent_configs=opponent_configs,
            model=model,
            model_type=model_type,
            device=device,
            episodes=eval_episodes,
            seed=seed + iteration * 1000,
        )
        print(
            f"  Eval reward {eval_stats['avg_reward']:.3f} | "
            f"Win-rate {eval_stats['win_rate'] * 100:.1f}% | "
            f"Avg length {eval_stats['avg_length']:.1f}"
        )

        if eval_stats["avg_reward"] > best_eval_reward:
            best_eval_reward = eval_stats["avg_reward"]
            save_dir = os.path.dirname(save_path)
            if save_dir and not os.path.exists(save_dir):
                os.makedirs(save_dir, exist_ok=True)
            torch.save(model.state_dict(), save_path)
            print(f"  → Saved improved checkpoint to {save_path}")

        wandb.log(
            {
                "dagger/dataset_size": len(dataset),
                "dagger/collect_reward": collect_stats.mean_reward,
                "dagger/expert_fraction": collect_stats.expert_fraction,
                "dagger/train_total_loss": train_stats["total_loss"],
                "dagger/train_policy_loss": train_stats["policy_loss"],
                "dagger/train_value_loss": train_stats["value_loss"],
                "dagger/train_accuracy": train_stats["accuracy"],
                "dagger/eval_reward": eval_stats["avg_reward"],
                "dagger/eval_win_rate": eval_stats["win_rate"],
            },
            step=iteration,
        )

        beta = max(beta * beta_decay, beta_min)

    env.close()
    if wandb.run is not None:
        wandb.run.summary["best_eval_reward"] = best_eval_reward

    return model


__all__ = ["train"]
