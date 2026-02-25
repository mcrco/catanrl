"""
Vectorized rollout helpers for policy/value evaluation.
"""

from __future__ import annotations

from typing import Any, Callable, List, Optional, Sequence, Tuple, Literal, cast

import numpy as np
import torch
from pufferlib.emulation import nativize

from catanatron.gym.envs.catanatron_env import ACTION_SPACE_SIZE

from ..envs.gym.puffer_rollout_utils import (
    EpisodeBuffer,
    extract_expert_actions_from_infos,
    flatten_puffer_observation,
    get_action_mask_from_obs,
    init_episode_buffers,
)
from ..envs.gym.single_env import compute_single_agent_dims, make_puffer_vectorized_envs
from ..models.wrappers import PolicyNetworkWrapper, ValueNetworkWrapper


def _extract_nn_won_from_infos(infos: object, batch_size: int) -> Tuple[Any, Any]:
    """Extract per-env win flags from vectorized infos.

    Returns:
        nn_won: bool array with inferred win flags.
        has_signal: bool array indicating whether nn_won[idx] came from infos.
    """
    nn_won = np.zeros(batch_size, dtype=np.bool_)
    has_signal = np.zeros(batch_size, dtype=np.bool_)

    if isinstance(infos, dict):
        infos_dict = cast(dict[str, Any], infos)
        won = infos_dict.get("nn_won")
        if won is None:
            return nn_won, has_signal
        won_arr = np.asarray(won, dtype=np.bool_)
        if won_arr.ndim == 0:
            nn_won[:] = bool(won_arr.item())
            has_signal[:] = True
        else:
            flat = won_arr.reshape(-1)
            size = min(batch_size, flat.shape[0])
            nn_won[:size] = flat[:size]
            has_signal[:size] = True
        return nn_won, has_signal

    if isinstance(infos, (list, tuple)):
        for idx, info in enumerate(infos):
            if idx >= batch_size:
                break
            if isinstance(info, dict) and "nn_won" in info:
                nn_won[idx] = bool(info["nn_won"])
                has_signal[idx] = True
        return nn_won, has_signal

    return nn_won, has_signal


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


def _assert_actions_are_unmasked(
    actions: np.ndarray,
    action_masks: np.ndarray,
    context: str,
) -> None:
    """Raise if any action is invalid under the provided mask."""
    if actions.shape[0] != action_masks.shape[0]:
        raise ValueError(
            f"Mismatched batch sizes for action validity check in {context}: "
            f"{actions.shape[0]} actions vs {action_masks.shape[0]} masks."
        )
    if np.any(actions < 0) or np.any(actions >= action_masks.shape[1]):
        raise RuntimeError(
            f"Out-of-range action found in {context}. "
            f"Action space size={action_masks.shape[1]}."
        )

    valid = action_masks[np.arange(actions.shape[0]), actions]
    if np.all(valid):
        return

    bad_indices = np.flatnonzero(~valid)
    preview = bad_indices[:10]
    examples = ", ".join(
        f"(idx={int(i)}, action={int(actions[i])}, valid_count={int(action_masks[i].sum())})"
        for i in preview
    )
    raise RuntimeError(
        f"Found {bad_indices.size} masked expert actions in {context}. "
        f"Examples: {examples}"
    )


def run_policy_value_eval_vectorized(
    policy_model: PolicyNetworkWrapper,
    critic_model: ValueNetworkWrapper,
    model_type: str,
    map_type: Literal["BASE", "TOURNAMENT", "MINI"],
    num_envs: int,
    num_games: int,
    gamma: float,
    opponent_configs: Sequence[str],
    device: Optional[str] = None,
    deterministic: bool = True,
    compare_to_expert: bool = False,
    expert_config: Optional[str] = None,
    seed: Optional[int] = None,
    progress_callback: Optional[Callable[[int], None]] = None,
) -> Tuple[int, List[int], List[float], List[float], List[int], List[int]]:
    """Run vectorized eval for a policy/critic against a single opponent set.

    Returns:
        wins: number of wins out of num_games
        turns_list: episode lengths
        value_preds: critic predictions across all episodes
        returns: discounted returns across all episodes
        expert_labels: expert action labels (filtered)
        expert_masked_preds: predicted actions from masked logits (filtered)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_device = torch.device(device)

    if compare_to_expert and not expert_config:
        raise ValueError("expert_config must be provided when compare_to_expert is True")

    policy_model.eval()
    critic_model.eval()

    num_players = len(opponent_configs) + 1
    dims = compute_single_agent_dims(num_players, map_type)
    actor_dim = dims["actor_dim"]
    critic_dim = dims["critic_dim"]

    if num_envs < 1:
        raise ValueError(f"num_envs must be >= 1, got {num_envs}")
    num_envs = min(num_envs, num_games)

    envs = make_puffer_vectorized_envs(
        reward_function="shaped",
        map_type=map_type,
        opponent_configs=list(opponent_configs),
        num_envs=num_envs,
        expert_config=expert_config if compare_to_expert else None,
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
    all_expert_labels: List[int] = []
    all_expert_masked_preds: List[int] = []

    ep_buffers = init_episode_buffers(num_envs)

    if seed is not None:
        try:
            observations, infos = envs.reset(seed=seed)
        except TypeError:
            observations, infos = envs.reset()
    else:
        observations, infos = envs.reset()

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
                policy_logits = _get_policy_logits(
                    policy_model, actor_tensor, model_type
                )
                mask_tensor = torch.as_tensor(
                    action_masks, dtype=torch.bool, device=torch_device
                )
                masked_logits = torch.where(
                    mask_tensor,
                    policy_logits,
                    torch.full_like(policy_logits, float("-inf")),
                )
                masked_logits = torch.clamp(masked_logits, min=-100, max=100)
                if deterministic:
                    actions = torch.argmax(masked_logits, dim=-1).cpu().numpy()
                else:
                    probs = torch.softmax(masked_logits, dim=-1)
                    dist = torch.distributions.Categorical(probs=probs)
                    actions = dist.sample().cpu().numpy()

            if compare_to_expert:
                expert_actions = extract_expert_actions_from_infos(infos, batch_size)
                _assert_actions_are_unmasked(
                    actions=expert_actions,
                    action_masks=action_masks,
                    context="evaluation",
                )
                non_single_mask = action_masks.sum(axis=1) > 1
                pred_masked = torch.argmax(masked_logits, dim=-1).cpu().numpy()
                for idx in range(batch_size):
                    if non_single_mask[idx]:
                        all_expert_labels.append(int(expert_actions[idx]))
                        all_expert_masked_preds.append(int(pred_masked[idx]))

            next_observations, rewards, terminations, truncations, infos = envs.step(
                actions
            )
            rewards = rewards.astype(np.float32)
            dones = np.logical_or(terminations, truncations)
            nn_won_batch, nn_won_has_signal = _extract_nn_won_from_infos(infos, batch_size)

            for idx in range(batch_size):
                ep_buffers[idx].rewards.append(rewards[idx])

                if dones[idx]:
                    if episodes_completed >= num_games:
                        continue
                    ep_rewards = np.array(ep_buffers[idx].rewards, dtype=np.float32)
                    ep_critic_states = np.array(
                        ep_buffers[idx].critic_states, dtype=np.float32
                    )

                    # Prefer explicit winner signal from env info; only fall back to reward
                    # for compatibility with older environments that do not expose nn_won.
                    final_reward = ep_rewards[-1]
                    nn_won = (
                        bool(nn_won_batch[idx])
                        if nn_won_has_signal[idx]
                        else abs(final_reward - 1.0) < 1e-5
                    )
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
                    if progress_callback is not None:
                        progress_callback(1)

                    ep_buffers[idx] = EpisodeBuffer(
                        critic_states=[], rewards=[], steps=0
                    )

            observations = next_observations
    finally:
        envs.close()

    return (
        wins,
        turns_list,
        all_value_preds,
        all_returns,
        all_expert_labels,
        all_expert_masked_preds,
    )
