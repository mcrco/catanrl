"""
Vectorized rollout helpers for policy/value evaluation.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from pufferlib.emulation import nativize

from catanatron.gym.envs.catanatron_env import ACTION_SPACE_SIZE

from ..envs.gym.puffer_rollout_utils import (
    EpisodeBuffer,
    flatten_puffer_observation,
    get_action_mask_from_obs,
    init_episode_buffers,
)
from ..envs.gym.single_env import compute_single_agent_dims, make_puffer_vectorized_envs
from ..models.wrappers import PolicyNetworkWrapper, ValueNetworkWrapper


def _get_policy_logits(
    policy_model: PolicyNetworkWrapper,
    states: torch.Tensor,
    model_type: str,
) -> torch.Tensor:
    if model_type == "flat":
        return policy_model(states)
    if model_type == "hierarchical":
        action_type_logits, param_logits = policy_model(states)
        return policy_model.get_flat_action_logits(action_type_logits, param_logits)
    raise ValueError(f"Unknown model_type '{model_type}'")


def run_policy_value_eval_vectorized(
    policy_model: PolicyNetworkWrapper,
    critic_model: ValueNetworkWrapper,
    model_type: str,
    map_type: str,
    num_games: int,
    gamma: float,
    opponent_configs: Sequence[str],
    device: Optional[str] = None,
    num_envs: Optional[int] = None,
) -> Tuple[int, List[int], List[float], List[float]]:
    """Run vectorized eval for a policy/critic against a single opponent set.

    Returns:
        wins: number of wins out of num_games
        turns_list: episode lengths
        value_preds: critic predictions across all episodes
        returns: discounted returns across all episodes
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_device = torch.device(device)

    policy_model.eval()
    critic_model.eval()

    num_players = len(opponent_configs) + 1
    dims = compute_single_agent_dims(num_players, map_type)
    actor_dim = dims["actor_dim"]
    critic_dim = dims["critic_dim"]

    if num_envs is None:
        num_envs = min(8, num_games)
    else:
        num_envs = min(num_envs, num_games)

    envs = make_puffer_vectorized_envs(
        reward_function="shaped",
        map_type=map_type,
        opponent_configs=list(opponent_configs),
        num_envs=num_envs,
        expert_config=None,
    )

    driver_env = envs.driver_env
    if hasattr(driver_env, "env_single_observation_space"):
        obs_space = driver_env.env_single_observation_space
    else:
        obs_space = driver_env.observation_space
    if hasattr(driver_env, "obs_dtype"):
        obs_dtype = driver_env.obs_dtype
    else:
        obs_dtype = np.float32

    episodes_completed = 0
    wins = 0
    turns_list: List[int] = []
    all_value_preds: List[float] = []
    all_returns: List[float] = []

    ep_buffers = init_episode_buffers(num_envs)

    observations, _ = envs.reset()

    try:
        while episodes_completed < num_games:
            batch_size = observations.shape[0]
            actor_batch = np.zeros((batch_size, actor_dim), dtype=np.float32)
            critic_batch = np.zeros((batch_size, critic_dim), dtype=np.float32)
            action_masks = np.zeros((batch_size, ACTION_SPACE_SIZE), dtype=np.bool_)

            for idx in range(batch_size):
                structured = nativize(observations[idx], obs_space, obs_dtype)
                actor_batch[idx], critic_batch[idx] = flatten_puffer_observation(
                    structured
                )
                action_masks[idx] = get_action_mask_from_obs(structured)

            for idx in range(batch_size):
                ep_buffers[idx].critic_states.append(critic_batch[idx].copy())
                ep_buffers[idx].steps += 1

            with torch.no_grad():
                actor_tensor = torch.from_numpy(actor_batch).float().to(torch_device)
                policy_logits = _get_policy_logits(policy_model, actor_tensor, model_type)

                mask_tensor = torch.as_tensor(
                    action_masks, dtype=torch.bool, device=torch_device
                )
                masked_logits = torch.where(
                    mask_tensor,
                    policy_logits,
                    torch.full_like(policy_logits, float("-inf")),
                )
                masked_logits = torch.clamp(masked_logits, min=-100, max=100)
                probs = torch.softmax(masked_logits, dim=-1)
                dist = torch.distributions.Categorical(probs=probs)
                actions = dist.sample().cpu().numpy()

            next_observations, rewards, terminations, truncations, _ = envs.step(actions)
            rewards = rewards.astype(np.float32)
            dones = np.logical_or(terminations, truncations)

            for idx in range(batch_size):
                ep_buffers[idx].rewards.append(rewards[idx])

                if dones[idx]:
                    ep_rewards = np.array(ep_buffers[idx].rewards, dtype=np.float32)
                    ep_critic_states = np.array(
                        ep_buffers[idx].critic_states, dtype=np.float32
                    )

                    final_reward = ep_rewards[-1]
                    nn_won = abs(final_reward - 1.0) < 1e-5
                    if nn_won:
                        wins += 1

                    turns_list.append(ep_buffers[idx].steps)

                    T = len(ep_rewards)
                    returns = np.zeros(T, dtype=np.float32)
                    running_return = 0.0
                    for t in reversed(range(T)):
                        running_return = ep_rewards[t] + gamma * running_return
                        returns[t] = running_return

                    with torch.no_grad():
                        critic_tensor = (
                            torch.from_numpy(ep_critic_states)
                            .float()
                            .to(torch_device)
                        )
                        value_preds = (
                            critic_model(critic_tensor).squeeze(-1).cpu().numpy()
                        )

                    all_value_preds.extend(value_preds.tolist())
                    all_returns.extend(returns.tolist())
                    episodes_completed += 1

                    ep_buffers[idx] = EpisodeBuffer(
                        critic_states=[], rewards=[], steps=0
                    )

            observations = next_observations
    finally:
        envs.close()

    return wins, turns_list, all_value_preds, all_returns
