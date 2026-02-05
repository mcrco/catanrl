"""
Shared evaluation utilities for PPO training loops.

Provides a unified interface for evaluating trained policies against
standard Catanatron opponents (RandomPlayer, ValueFunctionPlayer).
"""

from typing import Dict, Optional, Literal, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import wandb
from catanatron.models.player import RandomPlayer
from catanatron.players.value import ValueFunctionPlayer
from pufferlib.emulation import nativize

from ..features.catanatron_utils import COLOR_ORDER
from ..models.wrappers import PolicyNetworkWrapper, ValueNetworkWrapper
from ..players.nn_policy_player import NNPolicyPlayer
from .eval_nn_vs_catanatron import eval


def eval_policy_against_baselines(
    policy_model: PolicyNetworkWrapper,
    model_type: str,
    map_type: Literal["BASE", "TOURNAMENT", "MINI"],
    num_games: int = 250,
    seed: int = 42,
    log_to_wandb: bool = True,
    global_step: Optional[int] = None,
) -> Dict[str, float]:
    """
    Evaluate a trained policy against RandomPlayer and ValueFunctionPlayer.

    Args:
        policy_model: The trained policy network to evaluate.
        model_type: Model architecture type ("flat" or "hierarchical").
        map_type: Catan map type to use for evaluation games.
        num_games: Number of games to play against each opponent.
        seed: Random seed for reproducibility.
        log_to_wandb: Whether to log metrics to wandb.
        global_step: Global training step for wandb logging.

    Returns:
        Dictionary containing all evaluation metrics.
    """
    metrics: Dict[str, float] = {}

    nn_player = NNPolicyPlayer(
        COLOR_ORDER[0],
        model_type=model_type,
        model=policy_model,
    )

    # Evaluate against RandomPlayer
    wins, vps, total_vps, turns = eval(
        nn_player,
        [RandomPlayer(COLOR_ORDER[1], is_bot=True)],
        map_type=map_type,
        num_games=num_games,
        seed=seed,
    )
    metrics["eval/win_rate_vs_random"] = wins / num_games
    metrics["eval/avg_vps_vs_random"] = sum(vps) / len(vps)
    metrics["eval/avg_total_vps_vs_random"] = sum(total_vps) / len(total_vps)
    metrics["eval/avg_turns_vs_random"] = sum(turns) / len(turns)

    # Evaluate against ValueFunctionPlayer
    wins, vps, total_vps, turns = eval(
        nn_player,
        [ValueFunctionPlayer(COLOR_ORDER[1])],
        map_type=map_type,
        num_games=num_games,
        seed=seed,
    )
    metrics["eval/win_rate_vs_value"] = wins / num_games
    metrics["eval/avg_vps_vs_value"] = sum(vps) / len(vps)
    metrics["eval/avg_total_vps_vs_value"] = sum(total_vps) / len(total_vps)
    metrics["eval/avg_turns_vs_value"] = sum(turns) / len(turns)

    if log_to_wandb and global_step is not None:
        wandb.log(metrics, step=global_step)

    return metrics


def _flatten_puffer_observation(obs: Dict, num_players: int, map_type: str) -> Tuple[np.ndarray, np.ndarray]:
    """Convert Puffer observation dict to flattened actor and critic vectors."""
    actor_obs = obs["observation"]
    numeric = np.asarray(actor_obs["numeric"], dtype=np.float32).reshape(-1)
    board = np.asarray(actor_obs["board"], dtype=np.float32)
    board_wh_last = np.transpose(board, (1, 2, 0))
    board_flat = board_wh_last.reshape(-1)
    actor_vec = np.concatenate([numeric, board_flat], axis=0)
    critic_vec = np.asarray(obs["critic"], dtype=np.float32).reshape(-1)
    return actor_vec, critic_vec


def _get_action_mask_from_obs(obs: Dict) -> np.ndarray:
    """Extract action mask from Puffer observation."""
    return np.asarray(obs["action_mask"], dtype=np.int8) > 0


def eval_policy_value_against_baselines(
    policy_model: PolicyNetworkWrapper,
    critic_model: ValueNetworkWrapper,
    model_type: str,
    map_type: Literal["BASE", "TOURNAMENT", "MINI"],
    num_games: int = 250,
    gamma: float = 0.99,
    seed: int = 42,
    log_to_wandb: bool = True,
    global_step: Optional[int] = None,
    device: Optional[str] = None,
) -> Dict[str, float]:
    """
    Evaluate policy against baselines and critic prediction accuracy in a single loop.

    Uses vectorized puffer environments for fast inference while playing games
    against RandomPlayer and ValueFunctionPlayer. Collects critic value predictions
    during gameplay. It then:
    1. Reports policy metrics (win rates, VPs, turns)
    2. Computes actual discounted returns at episode end
    3. Reports value prediction metrics (MSE, explained variance, MAE)

    Args:
        policy_model: The trained policy network.
        critic_model: The trained critic (value) network.
        model_type: Model architecture type ("flat" or "hierarchical").
        map_type: Catan map type.
        num_games: Number of games to play against EACH opponent.
        gamma: Discount factor for computing returns.
        seed: Random seed.
        log_to_wandb: Whether to log metrics to wandb.
        global_step: Global training step for wandb logging.
        device: Torch device.

    Returns:
        Dictionary containing all evaluation metrics.
    """
    from catanatron.gym.envs.catanatron_env import ACTION_SPACE_SIZE
    from ..envs.gym.single_env import make_puffer_vectorized_envs, compute_single_agent_dims

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_device = torch.device(device)

    metrics: Dict[str, float] = {}

    policy_model.eval()
    critic_model.eval()

    # Define opponent types to evaluate against
    opponent_configs_list = [
        ("random", ["random"]),
        ("value", ["F"]),
    ]

    # We will collect value predictions and returns across all games
    all_value_preds = []
    all_returns = []

    for opponent_name, opponent_configs in opponent_configs_list:
        num_players = len(opponent_configs) + 1
        dims = compute_single_agent_dims(num_players, map_type)
        actor_dim = dims["actor_dim"]
        critic_dim = dims["critic_dim"]

        # Use multiple parallel envs for faster collection
        num_envs = min(8, num_games)
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

        # Episode tracking
        episodes_completed = 0
        wins = 0
        vps_list = []
        total_vps_list = []
        turns_list = []

        # Episode buffers for each env
        ep_buffers = [{"critic_states": [], "rewards": [], "steps": 0} for _ in range(num_envs)]

        observations, infos = envs.reset()

        try:
            while episodes_completed < num_games:
                batch_size = observations.shape[0]
                actor_batch = np.zeros((batch_size, actor_dim), dtype=np.float32)
                critic_batch = np.zeros((batch_size, critic_dim), dtype=np.float32)
                action_masks = np.zeros((batch_size, ACTION_SPACE_SIZE), dtype=np.bool_)

                # Decode observations
                for idx in range(batch_size):
                    structured = nativize(observations[idx], obs_space, obs_dtype)
                    actor_batch[idx], critic_batch[idx] = _flatten_puffer_observation(
                        structured, num_players, map_type
                    )
                    action_masks[idx] = _get_action_mask_from_obs(structured)

                # Store critic states for each env
                for idx in range(batch_size):
                    ep_buffers[idx]["critic_states"].append(critic_batch[idx].copy())
                    ep_buffers[idx]["steps"] += 1

                # Get actions from policy
                with torch.no_grad():
                    actor_tensor = torch.from_numpy(actor_batch).float().to(torch_device)

                    # Get policy logits and sample actions
                    if model_type == "flat":
                        policy_logits = policy_model(actor_tensor)
                    else:
                        action_type_logits, param_logits = policy_model(actor_tensor)
                        policy_logits = policy_model.get_flat_action_logits(action_type_logits, param_logits)

                    # Mask invalid actions
                    mask_tensor = torch.as_tensor(action_masks, dtype=torch.bool, device=torch_device)
                    masked_logits = torch.where(
                        mask_tensor,
                        policy_logits,
                        torch.full_like(policy_logits, float("-inf")),
                    )
                    masked_logits = torch.clamp(masked_logits, min=-100, max=100)
                    probs = torch.softmax(masked_logits, dim=-1)
                    dist = torch.distributions.Categorical(probs=probs)
                    actions = dist.sample().cpu().numpy()

                # Step environments
                next_observations, rewards, terminations, truncations, infos = envs.step(actions)
                rewards = rewards.astype(np.float32)
                dones = np.logical_or(terminations, truncations)

                for idx in range(batch_size):
                    ep_buffers[idx]["rewards"].append(rewards[idx])

                    if dones[idx]:
                        # Episode finished - compute returns and value predictions
                        ep_rewards = np.array(ep_buffers[idx]["rewards"], dtype=np.float32)
                        ep_critic_states = np.array(ep_buffers[idx]["critic_states"], dtype=np.float32)

                        # Extract episode metrics from info
                        # The final reward indicates win (+1) or loss (-1)
                        episode_reward = float(np.sum(ep_rewards))
                        nn_won = ep_rewards[-1] > 0  # Win reward is positive

                        if nn_won:
                            wins += 1

                        # We don't have direct access to VPs in vectorized env, so use episode length as proxy
                        turns_list.append(ep_buffers[idx]["steps"])
                        # Approximate VPs based on shaped reward (this is a rough estimate)
                        vps_list.append(max(0, episode_reward))
                        total_vps_list.append(10)  # Placeholder since we can't get opponent VPs easily

                        # Compute discounted returns
                        T = len(ep_rewards)
                        returns = np.zeros(T, dtype=np.float32)
                        running_return = 0.0
                        for t in reversed(range(T)):
                            running_return = ep_rewards[t] + gamma * running_return
                            returns[t] = running_return

                        # Get critic predictions
                        with torch.no_grad():
                            critic_tensor = torch.from_numpy(ep_critic_states).float().to(torch_device)
                            value_preds = critic_model(critic_tensor).squeeze(-1).cpu().numpy()

                        all_value_preds.extend(value_preds.tolist())
                        all_returns.extend(returns.tolist())
                        episodes_completed += 1

                        # Reset buffer
                        ep_buffers[idx] = {"critic_states": [], "rewards": [], "steps": 0}

                observations = next_observations
        finally:
            envs.close()

        # Log policy metrics for this opponent
        metrics[f"eval/win_rate_vs_{opponent_name}"] = wins / num_games if num_games > 0 else 0.0
        metrics[f"eval/avg_vps_vs_{opponent_name}"] = sum(vps_list) / len(vps_list) if vps_list else 0.0
        metrics[f"eval/avg_total_vps_vs_{opponent_name}"] = sum(total_vps_list) / len(total_vps_list) if total_vps_list else 0.0
        metrics[f"eval/avg_turns_vs_{opponent_name}"] = sum(turns_list) / len(turns_list) if turns_list else 0.0

    # Compute critic metrics
    if len(all_value_preds) > 0:
        value_preds = np.array(all_value_preds, dtype=np.float32)
        returns = np.array(all_returns, dtype=np.float32)

        # Mean Squared Error
        mse = float(np.mean((value_preds - returns) ** 2))

        # Mean Absolute Error
        mae = float(np.mean(np.abs(value_preds - returns)))

        # Explained Variance: 1 - Var(y - y_pred) / Var(y)
        var_returns = np.var(returns)
        if var_returns > 1e-8:
            explained_var = float(1.0 - np.var(returns - value_preds) / var_returns)
        else:
            explained_var = 0.0

        metrics["eval/value_mse"] = mse
        metrics["eval/value_mae"] = mae
        metrics["eval/value_explained_variance"] = explained_var
        metrics["eval/value_mean_pred"] = float(np.mean(value_preds))
        metrics["eval/value_mean_return"] = float(np.mean(returns))

    if log_to_wandb and global_step is not None:
        wandb.log(metrics, step=global_step)

    return metrics
