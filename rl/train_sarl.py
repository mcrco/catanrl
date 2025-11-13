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
import catanatron.gym
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
from models import PolicyValueNetwork


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


class RLAgent:
    """Agent wrapper for PolicyValueNetwork with action selection."""

    def __init__(self, model: PolicyValueNetwork, device: str):
        self.model = model
        self.device = device

    def select_action(
        self, state: torch.Tensor, valid_actions: np.ndarray, deterministic: bool = False
    ) -> Tuple[int, float, float]:
        """
        Select action using the policy network.

        Returns:
            action_idx: Index in ACTION_SPACE_SIZE
            log_prob: Log probability of selected action
            value: Value estimate
        """
        self.model.eval()  # Ensure eval mode for BatchNorm
        with torch.no_grad():
            policy_logits, value = self.model(state)  # [1, num_actions], [1]

            # Check for NaN in model outputs
            if torch.isnan(policy_logits).any() or torch.isnan(value).any():
                print("WARNING: NaN detected in model output!")
                # Fallback to random action
                action_idx = np.random.choice(valid_actions)
                log_prob = -np.log(len(valid_actions))
                return action_idx, log_prob, 0.0

            # Mask invalid actions
            mask = torch.full_like(policy_logits, float("-inf"))
            mask[0, valid_actions] = 0.0
            masked_logits = policy_logits + mask

            # Clamp logits to prevent overflow in softmax
            masked_logits = torch.clamp(masked_logits, min=-100, max=100)

            # Get action probabilities
            probs = F.softmax(masked_logits, dim=-1)  # [1, num_actions]

            if deterministic:
                action_idx = torch.argmax(probs[0, valid_actions]).item()
                action_idx = valid_actions[action_idx]
            else:
                # Sample from valid actions only
                valid_probs = probs[0, valid_actions]
                valid_probs = valid_probs / valid_probs.sum()  # Renormalize
                dist = torch.distributions.Categorical(probs=valid_probs)
                sampled_idx = dist.sample().item()
                action_idx = valid_actions[sampled_idx]

            # Calculate log probability for the selected action
            log_probs_all = F.log_softmax(masked_logits, dim=-1)  # [1, num_actions]
            log_prob = log_probs_all[0, action_idx].item()

        return action_idx, log_prob, value.item()

    def select_actions_batch(
        self, states: torch.Tensor, valid_actions_batch: List[np.ndarray], deterministic: bool = False
    ) -> Tuple[List[int], List[float], List[float]]:
        """
        Batched action selection for vectorized environments.
        Args:
            states: [batch, input_dim]
            valid_actions_batch: list of np.ndarray of valid action indices per env
        Returns:
            actions: list[int] length batch
            log_probs: list[float] length batch
            values: list[float] length batch
        """
        self.model.eval()
        actions: List[int] = []
        log_probs_out: List[float] = []
        values_out: List[float] = []
        with torch.no_grad():
            policy_logits, values = self.model(states)  # [B, A], [B]
            batch_size, num_actions = policy_logits.shape

            # Build mask
            masks = torch.full((batch_size, num_actions), float("-inf"), device=self.device)
            for i, valid_acts in enumerate(valid_actions_batch):
                masks[i, valid_acts] = 0.0

            masked_logits = policy_logits + masks
            masked_logits = torch.clamp(masked_logits, min=-100, max=100)

            probs = F.softmax(masked_logits, dim=-1)  # [B, A]
            log_probs_all = F.log_softmax(masked_logits, dim=-1)  # [B, A]

            for i in range(batch_size):
                valid_actions = valid_actions_batch[i]
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
        self, states: torch.Tensor, actions: torch.Tensor, valid_actions_batch: List[np.ndarray]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions for PPO update.

        Returns:
            log_probs: Log probabilities of actions [batch_size]
            values: Value estimates [batch_size]
            entropy: Policy entropy [batch_size]
        """
        policy_logits, values = self.model(states)  # [batch_size, num_actions], [batch_size]

        # Create masks for each sample in batch
        batch_size = states.shape[0]
        num_actions = policy_logits.shape[1]
        masks = torch.full((batch_size, num_actions), float("-inf"), device=self.device)
        for i, valid_acts in enumerate(valid_actions_batch):
            masks[i, valid_acts] = 0.0

        masked_logits = policy_logits + masks

        # Clamp for numerical stability
        masked_logits = torch.clamp(masked_logits, min=-100, max=100)

        log_probs_all = F.log_softmax(masked_logits, dim=-1)  # [batch_size, num_actions]

        # Get log probs for taken actions
        log_probs = log_probs_all.gather(1, actions.unsqueeze(1)).squeeze(1)  # [batch_size]

        # Calculate entropy
        probs = F.softmax(masked_logits, dim=-1)  # [batch_size, num_actions]
        entropy = -(probs * log_probs_all).sum(dim=-1)  # [batch_size]

        return log_probs, values, entropy


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
    agent: RLAgent,
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
    save_path: str = "rl/weights/policy_value_rl_best.pt",
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
    agent = RLAgent(model, device)
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


def main():
    parser = argparse.ArgumentParser(
        description="Train Catan Policy-Value Network with Self-Play RL (PPO)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--load-weights",
        type=str,
        default=None,
        help="Path to pre-trained weights from supervised learning",
    )
    parser.add_argument(
        "--episodes", type=int, default=1000, help="Number of episodes to train (default: 1000)"
    )
    parser.add_argument(
        "--update-freq", type=int, default=32, help="Update model every N episodes (default: 32)"
    )
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate (default: 1e-4)")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor (default: 0.99)")
    parser.add_argument(
        "--gae-lambda", type=float, default=0.95, help="GAE lambda parameter (default: 0.95)"
    )
    parser.add_argument(
        "--clip-epsilon", type=float, default=0.2, help="PPO clipping parameter (default: 0.2)"
    )
    parser.add_argument(
        "--value-coef", type=float, default=0.5, help="Value loss coefficient (default: 0.5)"
    )
    parser.add_argument(
        "--entropy-coef", type=float, default=0.01, help="Entropy coefficient (default: 0.01)"
    )
    parser.add_argument(
        "--n-epochs", type=int, default=4, help="Number of PPO epochs per update (default: 4)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=64, help="Mini-batch size for PPO update (default: 64)"
    )
    parser.add_argument(
        "--hidden-dims", type=str, default="512,512", help="Hidden dimensions (default: 512,512)"
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default=None,
        help="Path to save model (default: rl/weights/{wandb-run-name}/best.pt if using wandb, else rl/weights/policy_value_rl_best.pt)",
    )
    parser.add_argument(
        "--save-freq", type=int, default=100, help="Save checkpoint every N episodes (default: 100)"
    )
    parser.add_argument(
        "--map-type",
        type=str,
        default="BASE",
        choices=["BASE", "MINI", "TOURNAMENT"],
        help="Map type to use (default: BASE)",
    )
    parser.add_argument(
        "--opponents",
        type=str,
        nargs="+",
        default=["random"],
        help="""Opponent bot types (1-3 opponents for 2-4 player game). Options:
                - random, R: RandomPlayer
                - weighted, W: WeightedRandomPlayer  
                - value, F: ValueFunctionPlayer
                - victorypoint, VP: VictoryPointPlayer
                - alphabeta, AB: AlphaBetaPlayer (use AB:depth:prunning for custom params)
                - sameturnalphabeta, SAB: SameTurnAlphaBetaPlayer
                - mcts, M: MCTSPlayer (use M:num_sims for custom simulations)
                - playouts, G: GreedyPlayoutsPlayer (use G:num_playouts for custom playouts)
                Examples: --opponents random random random
                          --opponents F W AB
                          --opponents AB:3:False M:500
                (default: random)""",
    )
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="catan-rl",
        help="Wandb project name (default: catan-rl)",
    )
    parser.add_argument(
        "--wandb-run-name", type=str, default=None, help="Wandb run name (default: auto-generated)"
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=4,
        help="Number of parallel environments for vectorized training (default: 4)",
    )

    args = parser.parse_args()

    # Check if weights file exists
    if args.load_weights and not os.path.exists(args.load_weights):
        print(f"Error: Weights file '{args.load_weights}' not found!")
        print("Please train a model with train_joint.py first")
        return

    # Set default save path
    if args.save_path is None:
        if args.wandb and args.wandb_run_name:
            # Use wandb run name as directory
            args.save_path = f"rl/weights/{args.wandb_run_name}/best.pt"
        else:
            # Default fallback
            args.save_path = "rl/weights/policy_value_rl_best.pt"

    # Create save directory
    save_dir = os.path.dirname(args.save_path)
    if save_dir and not os.path.exists(save_dir):
        print(f"Creating directory '{save_dir}'")
        os.makedirs(save_dir, exist_ok=True)

    # Create opponents first to determine number of players
    if not args.opponents:
        args.opponents = ["random"]
    temp_opponents = create_opponents(args.opponents)
    num_players = len(temp_opponents) + 1  # +1 for the RL agent (BLUE)

    # Calculate input dimension based on actual number of players
    from catanatron.game import Game
    from catanatron.models.map import build_map

    # Create dummy players matching the actual game setup
    colors_list = [Color.RED, Color.BLUE, Color.WHITE, Color.ORANGE]
    dummy_players = [RandomPlayer(colors_list[i]) for i in range(num_players)]
    dummy_game = Game(dummy_players, catan_map=build_map(args.map_type))
    # For vectorized training we use 'mixed' observation: numeric (non-graph) + board tensor
    all_features = get_feature_ordering(num_players, args.map_type)
    numeric_len = len([f for f in all_features if not is_graph_feature(f)])
    board_tensor = create_board_tensor(dummy_game, dummy_game.state.colors[0])
    input_dim = numeric_len + board_tensor.size
    print(f"Number of players: {num_players}")
    print(f"Input dimension: {input_dim}")

    # Parse hidden dimensions
    hidden_dims = [int(dim) for dim in args.hidden_dims.split(",")]

    # Prepare wandb config
    wandb_config = None
    if args.wandb:
        wandb_config = {
            "project": args.wandb_project,
            "name": args.wandb_run_name,
            "config": {
                "algorithm": "PPO",
                "episodes": args.episodes,
                "update_freq": args.update_freq,
                "lr": args.lr,
                "gamma": args.gamma,
                "gae_lambda": args.gae_lambda,
                "clip_epsilon": args.clip_epsilon,
                "value_coef": args.value_coef,
                "entropy_coef": args.entropy_coef,
                "n_epochs": args.n_epochs,
                "batch_size": args.batch_size,
                "hidden_dims": hidden_dims,
                "map_type": args.map_type,
                "load_weights": args.load_weights,
                "opponents": args.opponents,
                "num_envs": args.num_envs,
            },
        }

    # Train model
    train(
        input_dim=input_dim,
        num_actions=ACTION_SPACE_SIZE,
        hidden_dims=hidden_dims,
        load_weights=args.load_weights,
        n_episodes=args.episodes,
        update_freq=args.update_freq,
        lr=args.lr,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_epsilon=args.clip_epsilon,
        value_coef=args.value_coef,
        entropy_coef=args.entropy_coef,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        save_path=args.save_path,
        save_freq=args.save_freq,
        wandb_config=wandb_config,
        map_type=args.map_type,
        opponent_configs=args.opponents,
        num_envs=args.num_envs,
    )

    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Model saved: {args.save_path}")

    # Finish wandb
    if args.wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
