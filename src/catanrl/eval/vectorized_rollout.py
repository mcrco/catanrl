"""
Vectorized rollout helpers for policy/value evaluation.
"""

from __future__ import annotations

from typing import Any, Callable, List, Literal, Optional, Sequence, Tuple, cast

import numpy as np
import torch
from pufferlib.emulation import nativize

from ..envs import (
    decode_puffer_batch,
    EpisodeBuffer,
    extract_expert_actions_from_infos,
    flatten_puffer_observation,
    get_action_mask_from_obs,
    init_episode_buffers,
)
from ..envs.puffer.multi_agent_env import make_vectorized_envs as make_marl_vectorized_envs
from ..envs.puffer.single_agent_env import compute_single_agent_dims, make_puffer_vectorized_envs
from ..features.catanatron_utils import (
    ActorObservationLevel,
    CriticObservationLevel,
    get_observation_indices_from_full,
)
from ..models.wrappers import PolicyNetworkWrapper, PolicyValueNetworkWrapper, ValueNetworkWrapper
from ..utils.seeding import derive_seed

SeatOption = Literal["first", "second", "random"]


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
    policy_model: PolicyNetworkWrapper | PolicyValueNetworkWrapper,
    states: torch.Tensor,
    model_type: str,
) -> torch.Tensor:
    if model_type == "flat":
        outputs = policy_model(states)
        if isinstance(policy_model, PolicyValueNetworkWrapper):
            logits, _ = outputs
            return logits
        return outputs
    if model_type == "hierarchical":
        outputs = policy_model(states)
        if isinstance(policy_model, PolicyValueNetworkWrapper):
            action_type_logits, param_logits, _ = outputs
        else:
            action_type_logits, param_logits = outputs
        return policy_model.get_flat_action_logits(action_type_logits, param_logits)
    raise ValueError(f"Unknown model_type '{model_type}'")


def _get_value_preds(
    value_model: ValueNetworkWrapper | PolicyValueNetworkWrapper,
    states: torch.Tensor,
) -> torch.Tensor:
    if isinstance(value_model, PolicyValueNetworkWrapper):
        outputs = value_model(states)
        return outputs[-1].squeeze(-1)
    return value_model(states).squeeze(-1)


def _select_masked_actions(
    policy_model: PolicyNetworkWrapper | PolicyValueNetworkWrapper,
    states: np.ndarray,
    action_masks: np.ndarray,
    model_type: str,
    device: torch.device,
    deterministic: bool,
) -> np.ndarray:
    if states.shape[0] == 0:
        return np.zeros(0, dtype=np.int64)

    with torch.no_grad():
        state_tensor = torch.from_numpy(states).float().to(device)
        logits = _get_policy_logits(policy_model, state_tensor, model_type)
        mask_tensor = torch.as_tensor(action_masks, dtype=torch.bool, device=device)
        masked_logits = torch.where(
            mask_tensor,
            logits,
            torch.full_like(logits, float("-inf")),
        )
        if deterministic:
            return torch.argmax(masked_logits, dim=-1).cpu().numpy()

        probs = torch.softmax(masked_logits, dim=-1)
        dist = torch.distributions.Categorical(probs=probs)
        return dist.sample().cpu().numpy()


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
            f"Out-of-range action found in {context}. Action space size={action_masks.shape[1]}."
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
        f"Found {bad_indices.size} masked expert actions in {context}. Examples: {examples}"
    )

def run_policy_value_eval_vectorized(
    policy_model: PolicyNetworkWrapper | PolicyValueNetworkWrapper,
    critic_model: ValueNetworkWrapper | PolicyValueNetworkWrapper,
    model_type: str,
    map_type: Literal["BASE", "TOURNAMENT", "MINI"],
    num_envs: int,
    num_games: int,
    gamma: float,
    opponent_configs: Sequence[str],
    device: Optional[str] = None,
    vps_to_win: int = 15,
    discard_limit: int = 9,
    deterministic: bool = True,
    compare_to_expert: bool = False,
    expert_config: Optional[str] = None,
    actor_observation_level: ActorObservationLevel = "private",
    critic_observation_level: CriticObservationLevel = "full",
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
    critic_uses_actor_states = isinstance(critic_model, PolicyValueNetworkWrapper)

    if num_games <= 0:
        return (0, [], [], [], [], [])

    num_players = len(opponent_configs) + 1
    dims = compute_single_agent_dims(
        num_players,
        map_type,
        actor_observation_level=actor_observation_level,
        critic_observation_level=critic_observation_level,
    )
    full_dims = compute_single_agent_dims(
        num_players,
        map_type,
        actor_observation_level=actor_observation_level,
        critic_observation_level="full",
    )
    actor_dim = dims["actor_dim"]
    full_critic_dim = full_dims["critic_dim"]
    critic_observation_indices = get_observation_indices_from_full(
        num_players,
        map_type,
        level=critic_observation_level,
    )

    if num_envs < 1:
        raise ValueError(f"num_envs must be >= 1, got {num_envs}")
    num_envs = min(num_envs, num_games)

    envs = make_puffer_vectorized_envs(
        reward_function="shaped",
        map_type=map_type,
        opponent_configs=list(opponent_configs),
        num_envs=num_envs,
        vps_to_win=vps_to_win,
        discard_limit=discard_limit,
        expert_config=expert_config if compare_to_expert else None,
        actor_observation_level=actor_observation_level,
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

    reset_seeds = None
    if seed is not None:
        reset_seeds = [derive_seed(seed, "env", env_idx) for env_idx in range(num_envs)]

    if reset_seeds is not None:
        try:
            observations, infos = envs.reset(seed=reset_seeds)
        except TypeError:
            observations, infos = envs.reset()
    else:
        observations, infos = envs.reset()

    try:
        while episodes_completed < num_games:
            batch_size = observations.shape[0]
            actor_batch = np.zeros((batch_size, actor_dim), dtype=np.float32)
            critic_batch = np.zeros((batch_size, full_critic_dim), dtype=np.float32)
            action_masks: np.ndarray | None = None

            for idx in range(batch_size):
                structured = nativize(observations[idx], obs_space, obs_dtype)
                actor_batch[idx], critic_batch[idx] = flatten_puffer_observation(structured)
                mask = get_action_mask_from_obs(structured)
                if action_masks is None:
                    action_masks = np.zeros((batch_size, mask.shape[0]), dtype=np.bool_)
                action_masks[idx] = mask
            if action_masks is None:
                raise ValueError("No action masks available from vectorized observations.")

            for idx in range(batch_size):
                value_state = (
                    actor_batch[idx]
                    if critic_uses_actor_states
                    else critic_batch[idx, critic_observation_indices]
                )
                ep_buffers[idx].critic_states.append(value_state.copy())
                ep_buffers[idx].steps += 1

            with torch.no_grad():
                actor_tensor = torch.from_numpy(actor_batch).float().to(torch_device)
                policy_logits = _get_policy_logits(policy_model, actor_tensor, model_type)
                mask_tensor = torch.as_tensor(action_masks, dtype=torch.bool, device=torch_device)
                masked_logits = torch.where(
                    mask_tensor,
                    policy_logits,
                    torch.full_like(policy_logits, float("-inf")),
                )
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

            next_observations, rewards, terminations, truncations, infos = envs.step(actions)
            rewards = rewards.astype(np.float32)
            dones = np.logical_or(terminations, truncations)
            nn_won_batch, nn_won_has_signal = _extract_nn_won_from_infos(infos, batch_size)

            for idx in range(batch_size):
                ep_buffers[idx].rewards.append(rewards[idx])

                if dones[idx]:
                    if episodes_completed >= num_games:
                        continue
                    ep_rewards = np.array(ep_buffers[idx].rewards, dtype=np.float32)
                    ep_critic_states = np.array(ep_buffers[idx].critic_states, dtype=np.float32)

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
                        critic_tensor = torch.from_numpy(ep_critic_states).float().to(torch_device)
                        value_preds = _get_value_preds(critic_model, critic_tensor).cpu().numpy()

                    all_value_preds.extend(value_preds.tolist())
                    all_returns.extend(returns.tolist())
                    episodes_completed += 1
                    if progress_callback is not None:
                        progress_callback(1)

                    ep_buffers[idx] = EpisodeBuffer(critic_states=[], rewards=[], steps=0)
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


def _extract_winner_agents_from_infos(
    infos: object,
    batch_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    winner_agents = np.full(batch_size, -1, dtype=np.int64)
    has_signal = np.zeros(batch_size, dtype=np.bool_)

    if isinstance(infos, dict):
        infos_dict = cast(dict[str, Any], infos)
        winner = infos_dict.get("winner_agent_index")
        if winner is None:
            return winner_agents, has_signal
        winner_arr = np.asarray(winner)
        flat = np.full(batch_size, -1, dtype=np.int64)
        if winner_arr.ndim == 0:
            flat[:] = int(winner_arr.item())
        else:
            size = min(batch_size, winner_arr.size)
            flat[:size] = winner_arr.reshape(-1)[:size].astype(np.int64, copy=False)
        winner_agents[:] = flat
        has_signal[:] = flat >= 0
        return winner_agents, has_signal

    if isinstance(infos, (list, tuple)):
        for idx, info in enumerate(infos):
            if idx >= batch_size:
                break
            if isinstance(info, dict) and info.get("winner_agent_index") is not None:
                winner_agents[idx] = int(info["winner_agent_index"])
                has_signal[idx] = winner_agents[idx] >= 0
        return winner_agents, has_signal

    return winner_agents, has_signal


def _current_slots_for_seat(
    nn_seat: SeatOption,
    num_envs: int,
    num_players: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if nn_seat == "first":
        return np.zeros(num_envs, dtype=np.int64)
    if nn_seat == "second":
        return np.ones(num_envs, dtype=np.int64)
    if nn_seat == "random":
        return rng.integers(0, num_players, size=num_envs, dtype=np.int64)
    raise ValueError(f"Unknown nn_seat '{nn_seat}'")


def run_policy_h2h_eval_vectorized(
    policy_model: PolicyNetworkWrapper,
    champion_model: PolicyNetworkWrapper,
    model_type: str,
    map_type: Literal["BASE", "TOURNAMENT", "MINI"],
    num_players: int,
    num_envs: int,
    num_games: int,
    device: Optional[str] = None,
    vps_to_win: int = 15,
    discard_limit: int = 9,
    deterministic: bool = True,
    nn_seat: SeatOption = "first",
    actor_observation_level: ActorObservationLevel = "private",
    critic_observation_level: CriticObservationLevel = "full",
    seed: Optional[int] = None,
    progress_callback: Optional[Callable[[int], None]] = None,
) -> Tuple[int, List[int]]:
    """Run vectorized current-policy-vs-champion H2H evaluation."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_device = torch.device(device)

    if num_players < 2:
        raise ValueError("H2H eval requires at least two players.")
    if nn_seat == "second" and num_players < 2:
        raise ValueError("Second-seat H2H eval requires at least two players.")
    if num_games <= 0:
        return 0, []
    if num_envs < 1:
        raise ValueError(f"num_envs must be >= 1, got {num_envs}")

    num_envs = min(num_envs, num_games)
    rng = np.random.default_rng(seed)
    policy_model.eval()
    champion_model.eval()

    envs = make_marl_vectorized_envs(
        num_players=num_players,
        map_type=map_type,
        vps_to_win=vps_to_win,
        discard_limit=discard_limit,
        shared_critic=False,
        reward_function="win",
        num_envs=num_envs,
        actor_observation_level=actor_observation_level,
    )

    driver_env = envs.driver_env
    obs_space = driver_env.env_single_observation_space
    obs_dtype = driver_env.obs_dtype
    actor_dim = int(driver_env.vector_dim)
    agents_per_env = getattr(envs, "agents_per_env", [num_players] * num_envs)
    env_offsets: list[int] = []
    ptr = 0
    for count in agents_per_env:
        env_offsets.append(ptr)
        ptr += int(count)

    current_slots = _current_slots_for_seat(nn_seat, num_envs, num_players, rng)
    episode_steps = np.zeros(num_envs, dtype=np.int64)
    episodes_completed = 0
    wins = 0
    turns_list: List[int] = []

    reset_seeds = None
    if seed is not None:
        reset_seeds = [derive_seed(seed, "h2h_env", env_idx) for env_idx in range(num_envs)]

    if reset_seeds is not None:
        try:
            observations, infos = envs.reset(seed=reset_seeds)
        except TypeError:
            observations, infos = envs.reset()
    else:
        observations, infos = envs.reset()

    try:
        while episodes_completed < num_games:
            actor_batch, _, action_masks = decode_puffer_batch(
                flat_obs=observations,
                obs_space=obs_space,
                obs_dtype=obs_dtype,
                actor_dim=actor_dim,
                critic_dim=None,
            )
            batch_size = actor_batch.shape[0]
            current_mask = np.zeros(batch_size, dtype=np.bool_)
            slots = np.zeros(batch_size, dtype=np.int64)
            for env_idx, start in enumerate(env_offsets):
                count = int(agents_per_env[env_idx])
                end = start + count
                slots[start:end] = np.arange(count, dtype=np.int64)
                current_mask[start:end] = slots[start:end] == current_slots[env_idx]

            actions = np.zeros(batch_size, dtype=np.int64)
            actions[current_mask] = _select_masked_actions(
                policy_model=policy_model,
                states=actor_batch[current_mask],
                action_masks=action_masks[current_mask],
                model_type=model_type,
                device=torch_device,
                deterministic=deterministic,
            )
            champion_mask = ~current_mask
            actions[champion_mask] = _select_masked_actions(
                policy_model=champion_model,
                states=actor_batch[champion_mask],
                action_masks=action_masks[champion_mask],
                model_type=model_type,
                device=torch_device,
                deterministic=deterministic,
            )

            next_observations, rewards, terminations, truncations, infos = envs.step(actions)
            dones = np.logical_or(terminations, truncations)
            winner_agents, winner_has_signal = _extract_winner_agents_from_infos(infos, batch_size)
            episode_steps += 1

            for env_idx, start in enumerate(env_offsets):
                end = start + int(agents_per_env[env_idx])
                if not bool(np.all(dones[start:end])):
                    continue
                if episodes_completed >= num_games:
                    continue

                row = start + int(current_slots[env_idx])
                if winner_has_signal[row]:
                    current_won = int(winner_agents[row]) == int(current_slots[env_idx])
                else:
                    current_won = bool(rewards[row] > 0.0)
                if current_won:
                    wins += 1
                turns_list.append(int(episode_steps[env_idx]))
                episodes_completed += 1
                if progress_callback is not None:
                    progress_callback(1)

                episode_steps[env_idx] = 0
                if nn_seat == "random":
                    current_slots[env_idx] = int(rng.integers(0, num_players))

            observations = next_observations
    finally:
        envs.close()

    return wins, turns_list
