"""
Centralized-critic PPO trainer for the multi-agent PettingZoo environment.

Each agent acts on its own (partial) observation while a separate critic
network consumes the perfect-information feature vector returned by
`full_game_to_features`
"""

from __future__ import annotations

import os
import random
from collections import deque
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import wandb
from tqdm import tqdm

from catanatron.gym.envs.catanatron_env import ACTION_SPACE_SIZE
from pufferlib.emulation import nativize

from ...envs import compute_multiagent_input_dim, flatten_marl_observation
from ...envs.zoo.multi_env import make_vectorized_envs as make_marl_vectorized_envs
from ...features.catanatron_utils import (
    get_full_numeric_feature_names,
    get_numeric_feature_names,
)
from ...eval.training_eval import evaluate_against_baselines
from ...models.backbones import BackboneConfig, MLPBackboneConfig
from ...models.models import (
    build_flat_policy_network,
    build_hierarchical_policy_network,
    build_value_network,
)
from ...models.wrappers import PolicyNetworkWrapper, ValueNetworkWrapper
from .gae import compute_gae_batched
from .buffers import CentralCriticExperienceBuffer


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
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Vectorized action selection for multiple parallel environments."""
        self.model.eval()
        with torch.no_grad():
            policy_logits = self._policy_logits(states)
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

        return actions, log_probs

    def evaluate_actions(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        valid_action_masks: np.ndarray,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Evaluate log-probs and entropy for PPO updates."""
        policy_logits = self._policy_logits(states)
        masked_logits, _ = self._mask_logits(policy_logits, valid_action_masks)
        log_probs_all = torch.log_softmax(masked_logits, dim=-1)
        probs = torch.softmax(masked_logits, dim=-1)

        log_probs = log_probs_all.gather(1, actions.unsqueeze(1)).squeeze(1)
        entropy = -(probs * log_probs_all).sum(dim=-1)
        return log_probs, entropy


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
    n_epochs: int,
    batch_size: int,
    device: str,
    last_critic_states: np.ndarray,
    gamma: float,
    gae_lambda: float,
    max_grad_norm: float,
) -> Dict[str, float]:
    """Run PPO updates over the centralized buffer."""
    if len(buffer) == 0:
        return {"policy_loss": 0.0, "value_loss": 0.0, "entropy_loss": 0.0, "total_loss": 0.0}

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

    # Filter out experiences with only one available action
    multi_action_mask = valid_action_masks.sum(axis=-1) > 1
    actor_states = actor_states[multi_action_mask]
    critic_states = critic_states[multi_action_mask]
    actions = actions[multi_action_mask]
    old_log_probs = old_log_probs[multi_action_mask]
    advantages = advantages[multi_action_mask]
    returns = returns[multi_action_mask]
    valid_action_masks = valid_action_masks[multi_action_mask]

    if len(advantages) == 0:
        return {"policy_loss": 0.0, "value_loss": 0.0, "entropy_loss": 0.0, "total_loss": 0.0}

    if advantages.std() > 1e-8:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    actor_states_t = torch.from_numpy(actor_states).float().to(device)
    critic_states_t = torch.from_numpy(critic_states).float().to(device)
    actions_t = torch.from_numpy(actions).long().to(device)
    old_log_probs_t = torch.from_numpy(old_log_probs).float().to(device)
    advantages_t = torch.from_numpy(advantages).float().to(device)
    returns_t = torch.from_numpy(returns).float().to(device)

    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_entropy_loss = 0.0
    total_loss = 0.0
    n_updates = 0

    for _ in range(n_epochs):
        indices = np.random.permutation(len(actor_states))
        for start in range(0, len(actor_states), batch_size):
            end = start + batch_size
            batch_idx = indices[start:end]
            if len(batch_idx) < 2:
                continue

            batch_actor_states = actor_states_t[batch_idx]
            batch_critic_states = critic_states_t[batch_idx]
            batch_actions = actions_t[batch_idx]
            batch_old_log_probs = old_log_probs_t[batch_idx]
            batch_advantages = advantages_t[batch_idx]
            batch_returns = returns_t[batch_idx]
            batch_valid_masks = valid_action_masks[batch_idx]

            log_probs, entropy = policy_agent.evaluate_actions(
                batch_actor_states, batch_actions, batch_valid_masks
            )
            ratio = torch.exp(log_probs - batch_old_log_probs)
            surr1 = ratio * batch_advantages
            surr2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * batch_advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            entropy_loss = -entropy.mean()

            policy_optimizer.zero_grad()
            (policy_loss + entropy_coef * entropy_loss).backward()
            torch.nn.utils.clip_grad_norm_(policy_agent.model.parameters(), max_grad_norm)
            if any(
                param.grad is not None and torch.isnan(param.grad).any()
                for param in policy_agent.model.parameters()
            ):
                policy_optimizer.zero_grad()
                print("NaN grad in policy optimizer")
                continue
            policy_optimizer.step()

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
            total_loss += float(
                policy_loss.item()
                + entropy_coef * entropy_loss.item()
                + value_coef * value_loss.item()
            )
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
    num_players: int = 2,
    map_type: str = "BASE",
    model_type: str = "flat",
    episodes: int = 500,
    rollout_steps: int = 4096,
    policy_lr: float = 3e-4,
    critic_lr: float = 3e-4,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_epsilon: float = 0.2,
    value_coef: float = 0.5,
    entropy_coef: float = 0.01,
    ppo_epochs: int = 4,
    batch_size: int = 256,
    policy_hidden_dims: Sequence[int] = (512, 512),
    critic_hidden_dims: Sequence[int] = (512, 512),
    save_path: str = "weights/policy_value_marl_central.pt",
    save_freq: int = 50,
    load_policy_weights: Optional[str] = None,
    load_critic_weights: Optional[str] = None,
    wandb_config: Optional[Dict] = None,
    seed: Optional[int] = 42,
    max_grad_norm: float = 0.5,
    deterministic_policy: bool = False,
    eval_games_per_opponent: int = 250,
    eval_freq: Optional[int] = None,
    num_envs: int = 2,
    reward_function: str = "shaped",
    metric_window: int = 200,
) -> Tuple[PolicyNetworkWrapper, ValueNetworkWrapper]:
    """Train a shared policy with a centralized critic using PPO."""

    assert 2 <= num_players <= 4, "num_players must be between 2 and 4"
    assert num_envs >= 1, "num_envs must be >= 1"

    set_global_seeds(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    actor_input_dim, _, numeric_dim = compute_multiagent_input_dim(num_players, map_type)
    numeric_dim = numeric_dim or len(get_numeric_feature_names(num_players, map_type))
    board_dim = actor_input_dim - numeric_dim
    full_numeric_len = len(get_full_numeric_feature_names(num_players, map_type))
    critic_input_dim = full_numeric_len + board_dim

    print(f"\n{'=' * 60}")
    print("Centralized Critic Multi-Agent PPO Training")
    print(f"{'=' * 60}")
    print(f"Device: {device}")
    print(f"Map type: {map_type} | Players: {num_players}")
    print(f"Actor input dim: {actor_input_dim} | Critic input dim: {critic_input_dim}")
    print(f"Episodes: {episodes} | Rollout steps: {rollout_steps}")
    print(f"Parallel environments: {num_envs}")
    if eval_freq:
        print(f"Evaluation frequency: every {eval_freq} PPO updates")
    else:
        print("Evaluation: every PPO update (when buffer is full)")

    if wandb_config:
        wandb.init(**wandb_config)
        print(
            f"Initialized wandb: {wandb_config['project']}/"
            f"{wandb_config.get('name', wandb.run.name)}"
        )
    else:
        wandb.init(mode="disabled")

    policy_backbone_config = BackboneConfig(
        architecture="mlp",
        args=MLPBackboneConfig(input_dim=actor_input_dim, hidden_dims=list(policy_hidden_dims)),
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

    critic_backbone_config = BackboneConfig(
        architecture="mlp",
        args=MLPBackboneConfig(input_dim=critic_input_dim, hidden_dims=list(critic_hidden_dims)),
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
    best_avg_return = -float("inf")
    global_step = 0
    total_episodes = 0
    ppo_update_count = 0
    episode_rewards = np.zeros(num_envs, dtype=np.float32)
    episode_lengths = np.zeros(num_envs, dtype=np.int32)
    reward_window = deque(maxlen=metric_window)
    length_window = deque(maxlen=metric_window)

    observations, infos = envs.reset()

    try:
        with tqdm(total=episodes, desc="Episodes") as pbar:
            while total_episodes < episodes:
                actor_batch, critic_batch, valid_masks = decode_observations(observations)
                actor_tensor = torch.from_numpy(actor_batch).float().to(device)
                actions, log_probs = policy_agent.select_actions_batch(
                    actor_tensor, valid_masks, deterministic=deterministic_policy
                )

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
                global_step += len(actions)

                for env_idx, start in enumerate(env_offsets):
                    end = start + agents_per_env[env_idx]
                    episode_rewards[env_idx] += rewards[start]
                    episode_lengths[env_idx] += 1
                    if dones[start:end].all():
                        total_episodes += 1
                        pbar.update(1)
                        reward_window.append(episode_rewards[env_idx])
                        length_window.append(episode_lengths[env_idx])
                        avg_reward = float(np.mean(reward_window))
                        avg_length = float(np.mean(length_window))
                        wandb.log(
                            {
                                "episode/return": episode_rewards[env_idx],
                                "episode/length": episode_lengths[env_idx],
                                "episode/avg_return": avg_reward,
                                "episode/avg_length": avg_length,
                                "episode": total_episodes,
                                "global_step": global_step,
                            },
                            step=global_step,
                        )
                        if len(reward_window) >= metric_window and avg_reward > best_avg_return:
                            best_avg_return = avg_reward
                            if save_path:
                                policy_path = save_path
                                critic_path = f"{os.path.splitext(save_path)[0]}_critic.pt"
                                torch.save(policy_model.state_dict(), policy_path)
                                torch.save(critic_model.state_dict(), critic_path)
                                if wandb.run is not None:
                                    wandb.run.summary["best_avg_reward"] = best_avg_return
                                print(f"  → Saved best models (avg reward: {best_avg_return:.2f})")
                        if save_path and save_freq > 0 and total_episodes % save_freq == 0:
                            base_dir = os.path.dirname(save_path)
                            policy_ckpt = (
                                os.path.join(base_dir, f"checkpoint_ep{total_episodes}.pt")
                                if base_dir
                                else f"checkpoint_ep{total_episodes}.pt"
                            )
                            critic_ckpt = f"{os.path.splitext(policy_ckpt)[0]}_critic.pt"
                            torch.save(policy_model.state_dict(), policy_ckpt)
                            torch.save(critic_model.state_dict(), critic_ckpt)
                            print(f"  → Saved checkpoint: {policy_ckpt}")

                        episode_rewards[env_idx] = 0.0
                        episode_lengths[env_idx] = 0

                observations = next_observations

                if len(buffer) >= max(batch_size * 2, rollout_steps):
                    _, bootstrap_critic_batch, _ = decode_observations(observations)
                    # Evaluate policy against catanatron bots
                    policy_agent.model.eval()
                    with torch.no_grad():
                        evaluate_against_baselines(
                            policy_model=policy_model,
                            model_type=model_type,
                            map_type=map_type,
                            num_games=eval_games_per_opponent,
                            seed=seed if seed is not None else 42,
                            log_to_wandb=True,
                            global_step=global_step,
                        )

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
                        n_epochs=ppo_epochs,
                        batch_size=batch_size,
                        device=device,
                        last_critic_states=bootstrap_critic_batch,
                        gamma=gamma,
                        gae_lambda=gae_lambda,
                        max_grad_norm=max_grad_norm,
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
                            "train/total_loss": metrics["total_loss"],
                        },
                        step=global_step,
                    )
                    print(
                        f"\n  → PPO Update | "
                        f"Policy Loss: {metrics['policy_loss']:.4f} | "
                        f"Value Loss: {metrics['value_loss']:.4f} | "
                        f"Entropy: {-metrics['entropy_loss']:.4f}"
                    )
    finally:
        envs.close()

    print(f"\nTraining complete. Best avg reward: {best_avg_return:.2f}")
    return policy_model, critic_model
