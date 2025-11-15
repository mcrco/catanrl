#!/usr/bin/env python3
"""
Multi-agent self-play training script using the PettingZoo AEC environment.

The script mirrors the structure of rl/train_sl.py but replaces supervised learning
with PPO-based reinforcement learning where all agents share the same policy/value
network and learn via self-play in the CatanatronAECEnv.
"""

import argparse
import os
import random
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import wandb

from catanatron.features import get_feature_ordering
from catanatron.gym.board_tensor_features import get_channels, is_graph_feature
from catanatron.gym.envs.catanatron_env import ACTION_SPACE_SIZE
from catanatron.zoo.env.catanatron_aec_env import CatanatronAECEnv, aec_env
from models import PolicyValueNetwork


@dataclass
class PendingExperience:
    state: np.ndarray
    action: int
    value: float
    log_prob: float
    valid_actions: np.ndarray


class MultiAgentRLAgent:
    """Shared policy/value agent for all players."""

    def __init__(self, model: PolicyValueNetwork, device: str):
        self.model = model
        self.device = device

    def select_action(
        self, state_vec: np.ndarray, valid_actions: np.ndarray, deterministic: bool = False
    ) -> Tuple[int, float, float]:
        """Select an action for the current agent with action masking."""
        if valid_actions.size == 0:
            valid_actions = np.arange(ACTION_SPACE_SIZE, dtype=np.int64)

        state_tensor = torch.from_numpy(state_vec).float().unsqueeze(0).to(self.device)
        self.model.eval()
        with torch.no_grad():
            policy_logits, value = self.model(state_tensor)

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
        """Evaluate log-probs, values, and entropy with action masking."""
        policy_logits, values = self.model(states)
        batch_size, num_actions = policy_logits.shape

        masks = torch.full(
            (batch_size, num_actions), float("-inf"), device=states.device
        )
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
        return log_probs, values, entropy


class ExperienceBuffer:
    """Simple on-policy buffer collecting transitions across agents."""

    def __init__(self):
        self.states: List[np.ndarray] = []
        self.actions: List[int] = []
        self.rewards: List[float] = []
        self.values: List[float] = []
        self.log_probs: List[float] = []
        self.valid_actions: List[np.ndarray] = []
        self.dones: List[bool] = []

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        value: float,
        log_prob: float,
        valid_actions: np.ndarray,
        done: bool,
    ):
        self.states.append(state.astype(np.float32, copy=False))
        self.actions.append(int(action))
        self.rewards.append(float(reward))
        self.values.append(float(value))
        self.log_probs.append(float(log_prob))
        self.valid_actions.append(valid_actions.astype(np.int64, copy=False))
        self.dones.append(bool(done))

    def get(self):
        return (
            np.stack(self.states).astype(np.float32),
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

    def __len__(self) -> int:
        return len(self.states)


def set_global_seeds(seed: Optional[int]):
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_input_dim(
    num_players: int, map_type: str, representation: str
) -> Tuple[int, Optional[Tuple[int, int, int]], int]:
    features = get_feature_ordering(num_players, map_type)
    if representation == "mixed":
        numeric_features = [f for f in features if not is_graph_feature(f)]
        channels = get_channels(num_players)
        board_shape = (channels, 21, 11)
        board_dim = int(np.prod(board_shape))
        numeric_dim = len(numeric_features)
        return numeric_dim + board_dim, board_shape, numeric_dim
    return len(features), None, len(features)


def flatten_observation(
    observation: np.ndarray, representation: str
) -> np.ndarray:
    if representation == "mixed":
        numeric = observation["numeric"].astype(np.float32, copy=False)
        board = observation["board"].astype(np.float32, copy=False)
        board_ch_last = np.transpose(board, (1, 2, 0))
        board_flat = board_ch_last.reshape(-1).astype(np.float32, copy=False)
        return np.concatenate([numeric, board_flat], axis=0)
    return observation.astype(np.float32, copy=False)


def get_valid_actions(info: Dict) -> np.ndarray:
    actions = info.get("valid_actions")
    if not actions:
        return np.arange(ACTION_SPACE_SIZE, dtype=np.int64)
    return np.array(actions, dtype=np.int64)


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    next_value: float,
    gamma: float,
    gae_lambda: float,
) -> Tuple[np.ndarray, np.ndarray]:
    advantages = np.zeros_like(rewards)
    last_gae = 0.0

    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_value_t = 0.0 if dones[t] else next_value
        else:
            next_value_t = values[t + 1]

        delta = rewards[t] + gamma * next_value_t * (1 - dones[t]) - values[t]
        last_gae = delta + gamma * gae_lambda * (1 - dones[t]) * last_gae
        advantages[t] = last_gae

    returns = advantages + values
    return advantages, returns


def ppo_update(
    agent: MultiAgentRLAgent,
    optimizer: optim.Optimizer,
    buffer: ExperienceBuffer,
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
    states, actions, rewards, old_values, old_log_probs, valid_actions, dones = buffer.get()

    agent.model.eval()
    with torch.no_grad():
        last_state = torch.from_numpy(states[-1:]).float().to(device)
        _, next_value_tensor = agent.model(last_state)
        next_value = float(next_value_tensor.item())
    agent.model.train()

    advantages, returns = compute_gae(
        rewards, old_values, dones, next_value, gamma, gae_lambda
    )
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
            batch_valid_actions = [valid_actions[i] for i in batch_idx]

            log_probs, values, entropy = agent.evaluate_actions(
                batch_states, batch_actions, batch_valid_actions
            )

            ratio = torch.exp(log_probs - batch_old_log_probs)
            surr1 = ratio * batch_advantages
            surr2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * batch_advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = F.mse_loss(values.squeeze(-1), batch_returns)
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
        return {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy_loss": 0.0,
            "total_loss": 0.0,
        }

    return {
        "policy_loss": total_policy_loss / n_updates,
        "value_loss": total_value_loss / n_updates,
        "entropy_loss": total_entropy / n_updates,
        "total_loss": total_loss / n_updates,
    }


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
    save_path: str = "rl/weights/policy_value_marl.pt",
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

    input_dim, board_shape, numeric_dim = compute_input_dim(num_players, map_type, representation)
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
    env: CatanatronAECEnv = aec_env(env_config)

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

    buffer = ExperienceBuffer()
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

                obs_vector = flatten_observation(observation, representation)
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
                    agent,
                    optimizer,
                    buffer,
                    clip_epsilon,
                    value_coef,
                    entropy_coef,
                    ppo_epochs,
                    batch_size,
                    device,
                    gamma,
                    gae_lambda,
                    max_grad_norm,
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


def parse_hidden_dims(hidden_dims: str) -> List[int]:
    return [int(dim) for dim in hidden_dims.split(",") if dim.strip()]


def main():
    parser = argparse.ArgumentParser(
        description="Train a multi-agent self-play PPO policy using the PettingZoo Catanatron env",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--num-players", type=int, default=2, choices=[2, 3, 4])
    parser.add_argument("--map-type", type=str, default="BASE", choices=["BASE", "MINI"])
    parser.add_argument("--vps-to-win", type=int, default=10)
    parser.add_argument("--representation", type=str, default="vector", choices=["vector", "mixed"])
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--rollout-steps", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-epsilon", type=float, default=0.2)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-dims", type=str, default="512,512")
    parser.add_argument("--save-path", type=str, default="rl/weights/policy_value_marl.pt")
    parser.add_argument("--save-freq", type=int, default=50)
    parser.add_argument("--load-weights", type=str, default=None)
    parser.add_argument("--invalid-action-reward", type=float, default=-1.0)
    parser.add_argument("--max-invalid-actions", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb-project", type=str, default="catan-marl")
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument(
        "--deterministic-policy",
        action="store_true",
        help="Use greedy action selection during data collection (default: sample)",
    )

    args = parser.parse_args()

    if args.save_path:
        save_dir = os.path.dirname(args.save_path)
        if save_dir and not os.path.exists(save_dir):
            print(f"Creating directory {save_dir}")
            os.makedirs(save_dir, exist_ok=True)

    hidden_dims = parse_hidden_dims(args.hidden_dims)

    wandb_config = None
    if args.wandb:
        config_dict = {
            "episodes": args.episodes,
            "rollout_steps": args.rollout_steps,
            "lr": args.lr,
            "gamma": args.gamma,
            "gae_lambda": args.gae_lambda,
            "clip_epsilon": args.clip_epsilon,
            "value_coef": args.value_coef,
            "entropy_coef": args.entropy_coef,
            "ppo_epochs": args.ppo_epochs,
            "batch_size": args.batch_size,
            "hidden_dims": hidden_dims,
            "num_players": args.num_players,
            "map_type": args.map_type,
            "representation": args.representation,
            "invalid_action_reward": args.invalid_action_reward,
            "max_invalid_actions": args.max_invalid_actions,
        }
        wandb_config = {
            "project": args.wandb_project,
            "name": args.wandb_run_name,
            "config": config_dict,
        }

    train(
        num_players=args.num_players,
        map_type=args.map_type,
        vps_to_win=args.vps_to_win,
        representation=args.representation,
        episodes=args.episodes,
        rollout_steps=args.rollout_steps,
        lr=args.lr,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_epsilon=args.clip_epsilon,
        value_coef=args.value_coef,
        entropy_coef=args.entropy_coef,
        ppo_epochs=args.ppo_epochs,
        batch_size=args.batch_size,
        hidden_dims=hidden_dims,
        save_path=args.save_path,
        save_freq=args.save_freq,
        load_weights=args.load_weights,
        wandb_config=wandb_config,
        invalid_action_reward=args.invalid_action_reward,
        max_invalid_actions=args.max_invalid_actions,
        seed=args.seed,
        max_grad_norm=args.max_grad_norm,
        deterministic_policy=args.deterministic_policy,
    )

    print("\n" + "=" * 60)
    print("Multi-agent training complete!")
    print("=" * 60)
    if args.save_path:
        print(f"Model saved to: {args.save_path}")

    if args.wandb:
        wandb.finish()


if __name__ == "__main__":
    main()

