#!/usr/bin/env python3
"""
Single Agent Reinforcement Learning training script for Catan Policy-Value Network.
Uses the Catanatron gym environment for training with PPO algorithm.
Run: python train_sarl.py --load-weights <path_to_pretrained_model>
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
from tqdm import tqdm
import wandb
from typing import List, Dict, Tuple

import gymnasium
from catanatron.models.player import RandomPlayer, Color
from catanatron.players.weighted_random import WeightedRandomPlayer
from catanatron.players.value import ValueFunctionPlayer
from catanatron.players.minimax import AlphaBetaPlayer, SameTurnAlphaBetaPlayer
from catanatron.players.search import VictoryPointPlayer
from catanatron.players.mcts import MCTSPlayer
from catanatron.players.playouts import GreedyPlayoutsPlayer
from catanatron.features import create_sample_vector, get_feature_ordering
from catanatron.gym.board_tensor_features import create_board_tensor, is_graph_feature
from catanatron.gym.envs.catanatron_env import ACTION_SPACE_SIZE, ACTIONS_ARRAY
from catanatron.game import Game
from catanatron.models.map import build_map
from ..models.models import PolicyValueNetwork
from catanrl.agents.gym_agent import GymAgent


def game_to_features(game, color: Color) -> np.ndarray:
    """Extract features from game state."""
    feature_vector = create_sample_vector(game, color)
    board_tensor = create_board_tensor(game, color)
    return np.concatenate([feature_vector, board_tensor.flatten()], axis=0)


def create_opponents(opponent_configs: List[str]) -> List:
    """
    Create opponent players based on configuration strings.

    Args:
        opponent_configs: List of opponent specifications (1-3 opponents). Examples:

    Returns:
        List of Player objects
    """
    colors = [Color.RED, Color.WHITE, Color.ORANGE]
    opponents = []

    for i, config in enumerate(opponent_configs):
        color = colors[i]
        parts = config.split(":")
        player_type = parts[0].lower()
        params = parts[1:] if len(parts) > 1 else []

        try:
            # Random Player
            if player_type in ["random", "r"]:
                opponents.append(RandomPlayer(color))

            # Weighted Random Player
            elif player_type in ["weighted", "w"]:
                opponents.append(WeightedRandomPlayer(color))

            # Value Function Player
            elif player_type in ["value", "valuefunction", "f"]:
                opponents.append(ValueFunctionPlayer(color))

            # Victory Point Player
            elif player_type in ["victorypoint", "vp"]:
                opponents.append(VictoryPointPlayer(color))

            # Alpha-Beta Player
            elif player_type in ["alphabeta", "ab"]:
                depth = int(params[0]) if len(params) > 0 else 2
                prunning = params[1].lower() != "false" if len(params) > 1 else True
                opponents.append(AlphaBetaPlayer(color, depth=depth, prunning=prunning))

            # Same Turn Alpha-Beta Player
            elif player_type in ["sameturnalphabeta", "sab"]:
                opponents.append(SameTurnAlphaBetaPlayer(color))

            # MCTS Player
            elif player_type in ["mcts", "m"]:
                num_simulations = int(params[0]) if len(params) > 0 else 100
                opponents.append(MCTSPlayer(color, num_simulations))

            # Greedy Playouts Player
            elif player_type in ["playouts", "greedyplayouts", "g"]:
                num_playouts = int(params[0]) if len(params) > 0 else 25
                opponents.append(GreedyPlayoutsPlayer(color, num_playouts))

            else:
                print(f"Warning: Unknown opponent type '{player_type}', using RandomPlayer")
                opponents.append(RandomPlayer(color))

        except Exception as e:
            print(f"Warning: Error creating opponent '{config}': {e}. Using RandomPlayer instead.")
            opponents.append(RandomPlayer(color))

    return opponents

class ExperienceBuffer:
    """Buffer for storing and sampling experiences."""

    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.valid_actions = []
        self.dones = []

    def add(self, state, action, reward, value, log_prob, valid_actions, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.valid_actions.append(valid_actions)
        self.dones.append(done)

    def get(self):
        return (
            np.array(self.states, dtype=np.float32),
            np.array(self.actions, dtype=np.int64),
            np.array(self.rewards, dtype=np.float32),
            np.array(self.values, dtype=np.float32),
            np.array(self.log_probs, dtype=np.float32),
            self.valid_actions,
            np.array(self.dones, dtype=np.bool_),
        )

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()
        self.valid_actions.clear()
        self.dones.clear()

    def __len__(self):
        return len(self.states)


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    next_value: float,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Generalized Advantage Estimation (GAE).

    Returns:
        advantages: GAE advantages
        returns: Target values (advantages + values)
    """
    advantages = np.zeros_like(rewards)
    last_gae = 0

    # Work backwards through the episode
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_value_t = next_value if not dones[t] else 0
        else:
            next_value_t = values[t + 1]

        delta = rewards[t] + gamma * next_value_t - values[t]
        last_gae = delta + gamma * gae_lambda * (1 - dones[t]) * last_gae
        advantages[t] = last_gae

    returns = advantages + values
    return advantages, returns


def ppo_update(
    agent: GymAgent,
    optimizer: optim.Optimizer,
    buffer: ExperienceBuffer,
    clip_epsilon: float,
    value_coef: float,
    entropy_coef: float,
    n_epochs: int,
    batch_size: int,
    device: str,
) -> Dict[str, float]:
    """
    Perform PPO update on collected experiences.

    Returns:
        Dictionary of training metrics
    """
    states, actions, rewards, old_values, old_log_probs, valid_actions, dones = buffer.get()

    # Compute GAE
    agent.model.eval()  # Set to eval mode for BatchNorm
    with torch.no_grad():
        next_state = torch.from_numpy(states[-1:]).float().to(device)
        _, next_value = agent.model(next_state)
        next_value = next_value.item()
    agent.model.train()  # Set back to train mode

    advantages, returns = compute_gae(rewards, old_values, dones, next_value)

    # Normalize advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # Convert to tensors (ensure float32)
    states_t = torch.from_numpy(states).float().to(device)
    actions_t = torch.from_numpy(actions).to(device)
    old_log_probs_t = torch.from_numpy(old_log_probs).float().to(device)
    advantages_t = torch.from_numpy(advantages).float().to(device)
    returns_t = torch.from_numpy(returns).float().to(device)

    # Training metrics
    total_policy_loss = 0
    total_value_loss = 0
    total_entropy_loss = 0
    total_loss = 0
    n_updates = 0

    # Multiple epochs of updates
    for epoch in range(n_epochs):
        # Mini-batch updates
        indices = np.random.permutation(len(states))
        for start in range(0, len(states), batch_size):
            end = start + batch_size
            batch_indices = indices[start:end]

            # Skip batches with size 1 (BatchNorm requires > 1)
            if len(batch_indices) == 1:
                continue

            # Get batch
            batch_states = states_t[batch_indices]
            batch_actions = actions_t[batch_indices]
            batch_old_log_probs = old_log_probs_t[batch_indices]
            batch_advantages = advantages_t[batch_indices]
            batch_returns = returns_t[batch_indices]
            batch_valid_actions = [valid_actions[i] for i in batch_indices]

            # Evaluate actions
            log_probs, values, entropy = agent.evaluate_actions(
                batch_states, batch_actions, batch_valid_actions
            )

            # PPO policy loss with clipping
            ratio = torch.exp(log_probs - batch_old_log_probs)
            surr1 = ratio * batch_advantages
            surr2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * batch_advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value loss
            value_loss = F.mse_loss(values, batch_returns)

            # Entropy bonus (for exploration)
            entropy_loss = -entropy.mean()

            # Total loss
            loss = policy_loss + value_coef * value_loss + entropy_coef * entropy_loss

            # Backprop and optimize
            optimizer.zero_grad()
            loss.backward()

            # Clip gradients to prevent explosion
            torch.nn.utils.clip_grad_norm_(agent.model.parameters(), max_norm=0.5)

            # Check for NaN in gradients
            has_nan_grad = False
            for param in agent.model.parameters():
                if param.grad is not None and torch.isnan(param.grad).any():
                    has_nan_grad = True
                    break

            if has_nan_grad:
                print("WARNING: NaN gradient detected, skipping update")
                optimizer.zero_grad()
            else:
                optimizer.step()

            # Track metrics
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy_loss += entropy_loss.item()
            total_loss += loss.item()
            n_updates += 1

    # Return metrics (handle case where no updates occurred)
    if n_updates == 0:
        return {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy_loss": 0.0,
            "total_loss": 0.0,
        }

    return {
        "policy_loss": total_policy_loss / n_updates,
        "value_loss": total_value_loss / n_updates,
        "entropy_loss": total_entropy_loss / n_updates,
        "total_loss": total_loss / n_updates,
    }


def train(
    input_dim: int,
    num_actions: int = ACTION_SPACE_SIZE,
    hidden_dims: List[int] = [512, 512],
    load_weights: str = None,
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
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    wandb_config: dict = None,
    map_type: str = "BASE",
    opponent_configs: List[str] = None,
    num_envs: int = 4,
):
    """Train PolicyValueNetwork using single agent reinforcement learning with PPO."""

    print(f"\n{'=' * 60}")
    print("Training Policy-Value Network with Single Agent Reinforcement Learning (PPO)")
    print(f"{'=' * 60}")
    print(f"Device: {device}")
    print(f"Map type: {map_type}")
    print(f"Episodes: {n_episodes}")
    print(f"Update frequency: {update_freq} episodes")
    print(f"Parallel environments: {num_envs}")

    # Create opponents
    if opponent_configs is None:
        opponent_configs = ["random"]
    opponents = create_opponents(opponent_configs)
    num_players = len(opponents) + 1  # +1 for the RL agent
    print(f"Number of players: {num_players}")
    print(f"Opponents: {[repr(o) for o in opponents]}")

    # Initialize wandb
    if wandb_config:
        wandb.init(**wandb_config)
        print(
            f"Initialized wandb: {wandb_config['project']}/{wandb_config.get('name', wandb.run.name)}"
        )
    else:
        wandb.init(mode="disabled")

    # Create model
    model = PolicyValueNetwork(input_dim, num_actions, hidden_dims).to(device)

    # Load pre-trained weights if provided
    if load_weights and os.path.exists(load_weights):
        print(f"Loading weights from: {load_weights}")
        state_dict = torch.load(load_weights, map_location=device)
        model.load_state_dict(state_dict)
        print("  ✓ Weights loaded successfully")

    # Create agent and optimizer
    agent = GymAgent(model, device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.eval()  # Start in eval mode for episode collection

    # Training tracking
    buffer = ExperienceBuffer()
    episode_rewards = deque(maxlen=100)
    episode_lengths = deque(maxlen=100)
    best_avg_reward = float("-inf")
    global_step = 0
    total_episodes = 0

    # Create vectorized environments
    def make_env():
        def _init():
            env = gymnasium.make(
                "catanatron/Catanatron-v0",
                config={
                    "map_type": map_type,
                    "enemies": create_opponents(opponent_configs),  # Create fresh opponents for each env
                    "representation": "mixed",  # return {'numeric', 'board'} for stable vectorization
                },
            )
            return env
        return _init

    envs = gymnasium.vector.AsyncVectorEnv([make_env() for _ in range(num_envs)])
    
    # Per-environment episode tracking
    env_episode_rewards = np.zeros(num_envs)
    env_episode_lengths = np.zeros(num_envs, dtype=int)
    episode_start = np.zeros(num_envs, dtype=bool)

    print("\nStarting training...")

    observations, infos = envs.reset()
    
    # Training loop - continue until we've completed enough total episodes
    with tqdm(total=n_episodes, desc="Episodes") as pbar:
        while total_episodes < n_episodes:
            # Build states from observations (mixed: dict of arrays)
            # numeric: [N, numeric_len], board: [N, C, H, W] (channels_first=True)
            numeric_obs = observations["numeric"].astype(np.float32)
            board_obs = observations["board"].astype(np.float32)  # [N, C, H, W]
            # transpose to channels_last like training used previously, then flatten
            board_obs_ch_last = np.transpose(board_obs, (0, 2, 3, 1))  # [N, H, W, C]
            board_flat = board_obs_ch_last.reshape(board_obs_ch_last.shape[0], -1)
            states = np.concatenate([numeric_obs, board_flat], axis=1).astype(np.float32)  # [N, D]
            valid_actions_list = [np.array(v, dtype=np.int64) for v in infos["valid_actions"]]

            states_t = torch.from_numpy(states).float().to(device)

            # Batched action selection
            actions, log_probs, values = agent.select_actions_batch(states_t, valid_actions_list)
            
            # Step all environments
            next_observations, rewards, terminations, truncations, infos = envs.step(actions)
            
            # Process results for each environment
            for env_idx in range(num_envs):
                if not episode_start[env_idx]:
                    # Add experience to buffer
                    buffer.add(
                        states[env_idx],
                        actions[env_idx],
                        rewards[env_idx],
                        values[env_idx],
                        log_probs[env_idx],
                        valid_actions_list[env_idx],
                        terminations[env_idx] or truncations[env_idx],
                    )
                    
                    env_episode_rewards[env_idx] += rewards[env_idx]
                    env_episode_lengths[env_idx] += 1
                    global_step += 1
                
                # Check if episode ended
                if terminations[env_idx] or truncations[env_idx]:
                    # Log completed episode
                    episode_rewards.append(env_episode_rewards[env_idx])
                    episode_lengths.append(env_episode_lengths[env_idx])
                    total_episodes += 1
                    pbar.update(1)
                    
                    # Log to wandb
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
                    
                    # Reset episode tracking for this env
                    env_episode_rewards[env_idx] = 0
                    env_episode_lengths[env_idx] = 0
                    episode_start[env_idx] = True
                    
                    # Save best model
                    min_episodes_for_best = min(update_freq, 100)
                    if len(episode_rewards) >= min_episodes_for_best and avg_reward > best_avg_reward:
                        best_avg_reward = avg_reward
                        torch.save(model.state_dict(), save_path)
                        print(f"  → Saved best model (avg_reward: {avg_reward:.2f})")
                        wandb.run.summary["best_avg_reward"] = best_avg_reward
                    
                    # Periodic checkpoint
                    if total_episodes % save_freq == 0:
                        save_dir = os.path.dirname(save_path)
                        if save_dir:
                            checkpoint_path = os.path.join(save_dir, f"checkpoint_ep{total_episodes}.pt")
                        else:
                            checkpoint_path = f"checkpoint_ep{total_episodes}.pt"
                        torch.save(model.state_dict(), checkpoint_path)
                        print(f"  → Saved checkpoint: {checkpoint_path}")
                else:
                    episode_start[env_idx] = False
            
            # Update observations
            observations = next_observations
            
            # Perform PPO update when buffer is large enough
            min_samples = max(batch_size * 2, 4)
            if len(buffer) >= update_freq * 50 and len(buffer) >= min_samples:  # Update more frequently with vectorized envs
                metrics = ppo_update(
                    agent,
                    optimizer,
                    buffer,
                    clip_epsilon,
                    value_coef,
                    entropy_coef,
                    n_epochs,
                    batch_size,
                    device,
                )
                buffer.clear()
                model.eval()  # Set back to eval mode
                
                print(
                    f"\n  → PPO Update | "
                    f"Policy Loss: {metrics['policy_loss']:.4f} | "
                    f"Value Loss: {metrics['value_loss']:.4f} | "
                    f"Entropy: {-metrics['entropy_loss']:.4f}"
                )
                
                wandb.log(
                    {
                        "train/policy_loss": metrics["policy_loss"],
                        "train/value_loss": metrics["value_loss"],
                        "train/entropy": -metrics["entropy_loss"],
                        "train/total_loss": metrics["total_loss"],
                    },
                    step=global_step,
                )

    # Close environments
    envs.close()
    
    print(f"\nBest average reward: {best_avg_reward:.2f}")
    return model