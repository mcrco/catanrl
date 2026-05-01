"""
Shared evaluation utilities for PPO training loops.

Provides a unified interface for evaluating trained policies against
standard Catanatron opponents (RandomPlayer, ValueFunctionPlayer).
"""

import copy
import os

from typing import Dict, Optional, Literal, Sequence

import numpy as np
import torch
import wandb
from tqdm import tqdm
from catanatron.models.player import RandomPlayer
from catanatron.players.value import ValueFunctionPlayer
from sklearn.metrics import f1_score

from ..features.catanatron_utils import ActorObservationLevel, COLOR_ORDER
from ..models.wrappers import PolicyNetworkWrapper, PolicyValueNetworkWrapper, ValueNetworkWrapper
from ..players.nn_policy_player import NNPolicyPlayer
from ..utils.seeding import derive_seed
from .eval_nn_vs_catanatron import eval
from .vectorized_rollout import SeatOption, run_policy_h2h_eval_vectorized, run_policy_value_eval_vectorized


def eval_policy_against_baselines(
    policy_model: PolicyNetworkWrapper,
    model_type: str,
    map_type: Literal["BASE", "TOURNAMENT", "MINI"],
    num_games: int = 250,
    seed: int = 42,
    vps_to_win: int = 15,
    discard_limit: int = 9,
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
        map_type=map_type,
    )

    # Evaluate against RandomPlayer
    wins, vps, total_vps, turns = eval(
        nn_player,
        [RandomPlayer(COLOR_ORDER[1], is_bot=True)],
        map_type=map_type,
        num_games=num_games,
        seed=seed,
        vps_to_win=vps_to_win,
        discard_limit=discard_limit,
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
        vps_to_win=vps_to_win,
        discard_limit=discard_limit,
    )
    metrics["eval/win_rate_vs_value"] = wins / num_games
    metrics["eval/avg_vps_vs_value"] = sum(vps) / len(vps)
    metrics["eval/avg_total_vps_vs_value"] = sum(total_vps) / len(total_vps)
    metrics["eval/avg_turns_vs_value"] = sum(turns) / len(turns)

    if log_to_wandb and global_step is not None:
        wandb.log(metrics, step=global_step)

    return metrics


def eval_policy_against_champion(
    policy_model: PolicyNetworkWrapper,
    model_type: str,
    map_type: Literal["BASE", "TOURNAMENT", "MINI"],
    num_players: int,
    champion_policy_path: str,
    num_games: int = 50,
    seed: int = 42,
    vps_to_win: int = 15,
    discard_limit: int = 9,
    log_to_wandb: bool = True,
    global_step: Optional[int] = None,
    num_envs: int = 1,
    device: Optional[str] = None,
    nn_seat: SeatOption = "random",
    actor_observation_level: ActorObservationLevel = "private",
) -> Dict[str, float]:
    """Evaluate the current policy against a frozen champion checkpoint."""
    metrics: Dict[str, float] = {}
    if num_games <= 0 or not champion_policy_path or not os.path.exists(champion_policy_path):
        return metrics
    if num_players < 2:
        raise ValueError("Champion eval requires at least 2 players.")

    torch_device = torch.device(device) if device is not None else next(policy_model.parameters()).device
    champion_model = copy.deepcopy(policy_model).to(torch_device)
    champion_state = torch.load(champion_policy_path, map_location=torch_device)
    champion_model.load_state_dict(champion_state)
    champion_model.eval()
    policy_model.eval()

    wins, turns = run_policy_h2h_eval_vectorized(
        policy_model=policy_model,
        champion_model=champion_model,
        model_type=model_type,
        map_type=map_type,
        num_players=num_players,
        num_envs=num_envs,
        num_games=num_games,
        device=str(torch_device),
        vps_to_win=vps_to_win,
        discard_limit=discard_limit,
        nn_seat=nn_seat,
        actor_observation_level=actor_observation_level,
        seed=seed,
    )

    metrics["eval_h2h/win_rate_vs_champion"] = wins / num_games if num_games else 0.0
    metrics["eval_h2h/avg_turns_vs_champion"] = (
        sum(turns) / len(turns) if turns else 0.0
    )
    metrics["eval_h2h/games"] = float(num_games)
    metrics["eval_h2h/seat_first"] = float(nn_seat == "first")
    metrics["eval_h2h/seat_second"] = float(nn_seat == "second")
    metrics["eval_h2h/seat_random"] = float(nn_seat == "random")

    if log_to_wandb and global_step is not None:
        wandb.log(metrics, step=global_step)

    return metrics


def eval_policy_value_against_baselines(
    policy_model: PolicyNetworkWrapper | PolicyValueNetworkWrapper,
    critic_model: ValueNetworkWrapper | PolicyValueNetworkWrapper,
    model_type: str,
    map_type: Literal["BASE", "TOURNAMENT", "MINI"],
    num_envs: int,
    eval_opponent_configs: Optional[Sequence[str]] = None,
    num_games: int = 250,
    gamma: float = 0.99,
    seed: int = 42,
    vps_to_win: int = 15,
    discard_limit: int = 9,
    log_to_wandb: bool = True,
    global_step: Optional[int] = None,
    device: Optional[str] = None,
    deterministic: bool = True,
    compare_to_expert: bool = False,
    expert_config: Optional[str] = None,
    actor_observation_level: ActorObservationLevel = "private",
    progress_desc: Optional[str] = None,
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
        eval_opponent_configs: Opponent configs from training/eval setup. Only the
            number of opponent slots is used so baseline evals run with the same
            player count as training.
        num_games: Number of games to play against EACH opponent.
        gamma: Discount factor for computing returns.
        seed: Random seed.
        log_to_wandb: Whether to log metrics to wandb.
        global_step: Global training step for wandb logging.
        device: Torch device.
        compare_to_expert: Whether to compute accuracy/F1 against expert labels.
        expert_config: Expert config to use for labels when compare_to_expert is True.

    Returns:
        Dictionary containing all evaluation metrics.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    metrics: Dict[str, float] = {}

    policy_model.eval()
    critic_model.eval()

    num_opponents = len(eval_opponent_configs) if eval_opponent_configs is not None else 1
    if num_opponents < 1:
        raise ValueError("eval_opponent_configs must contain at least one opponent.")

    # Define opponent types to evaluate against while preserving player count.
    opponent_configs_list = [
        ("random", ["random"] * num_opponents),
        ("value", ["F"] * num_opponents),
    ]

    if num_games <= 0:
        return metrics

    # We will collect value predictions and returns across all games
    all_value_preds = []
    all_returns = []

    all_expert_labels = []
    all_expert_masked_preds = []

    eval_total_games = len(opponent_configs_list) * num_games
    if progress_desc and eval_total_games > 0:
        with tqdm(
            total=eval_total_games,
            desc=progress_desc,
            unit="game",
            leave=False,
            position=1,
        ) as eval_pbar:
            for opponent_idx, (opponent_name, opponent_configs) in enumerate(opponent_configs_list):
                opponent_seed = derive_seed(seed, "opponent", opponent_name, opponent_idx)
                (
                    wins,
                    turns_list,
                    value_preds,
                    returns,
                    expert_labels,
                    expert_masked_preds,
                ) = run_policy_value_eval_vectorized(
                    policy_model=policy_model,
                    critic_model=critic_model,
                    model_type=model_type,
                    map_type=map_type,
                    num_games=num_games,
                    gamma=gamma,
                    opponent_configs=opponent_configs,
                    device=device,
                    num_envs=num_envs,
                    vps_to_win=vps_to_win,
                    discard_limit=discard_limit,
                    deterministic=deterministic,
                    compare_to_expert=compare_to_expert,
                    expert_config=expert_config,
                    actor_observation_level=actor_observation_level,
                    seed=opponent_seed,
                    progress_callback=eval_pbar.update,
                )

                # Log policy metrics for this opponent
                metrics[f"eval/win_rate_vs_{opponent_name}"] = (
                    wins / num_games if num_games > 0 else 0.0
                )
                metrics[f"eval/avg_turns_vs_{opponent_name}"] = (
                    sum(turns_list) / len(turns_list) if turns_list else 0.0
                )

                all_value_preds.extend(value_preds)
                all_returns.extend(returns)
                all_expert_labels.extend(expert_labels)
                all_expert_masked_preds.extend(expert_masked_preds)
    else:
        for opponent_idx, (opponent_name, opponent_configs) in enumerate(opponent_configs_list):
            opponent_seed = derive_seed(seed, "opponent", opponent_name, opponent_idx)
            (
                wins,
                turns_list,
                value_preds,
                returns,
                expert_labels,
                expert_masked_preds,
            ) = run_policy_value_eval_vectorized(
                policy_model=policy_model,
                critic_model=critic_model,
                model_type=model_type,
                map_type=map_type,
                num_games=num_games,
                gamma=gamma,
                opponent_configs=opponent_configs,
                device=device,
                num_envs=num_envs,
                vps_to_win=vps_to_win,
                discard_limit=discard_limit,
                deterministic=deterministic,
                compare_to_expert=compare_to_expert,
                expert_config=expert_config,
                actor_observation_level=actor_observation_level,
                seed=opponent_seed,
            )

            # Log policy metrics for this opponent
            metrics[f"eval/win_rate_vs_{opponent_name}"] = (
                wins / num_games if num_games > 0 else 0.0
            )
            metrics[f"eval/avg_turns_vs_{opponent_name}"] = (
                sum(turns_list) / len(turns_list) if turns_list else 0.0
            )

            all_value_preds.extend(value_preds)
            all_returns.extend(returns)
            all_expert_labels.extend(expert_labels)
            all_expert_masked_preds.extend(expert_masked_preds)

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

    if compare_to_expert and all_expert_labels:
        labels = np.array(all_expert_labels, dtype=np.int64)
        masked_preds = np.array(all_expert_masked_preds, dtype=np.int64)
        metrics["eval/acc_vs_expert_masked_logits"] = float(np.mean(labels == masked_preds))
        metrics["eval/f1_score_vs_expert_masked_logits"] = float(
            f1_score(labels, masked_preds, average="macro", zero_division=0.0)
        )

    if log_to_wandb and global_step is not None:
        wandb.log(metrics, step=global_step)

    return metrics
