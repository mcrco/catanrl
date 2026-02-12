"""
Shared evaluation utilities for PPO training loops.

Provides a unified interface for evaluating trained policies against
standard Catanatron opponents (RandomPlayer, ValueFunctionPlayer).
"""

from typing import Dict, Optional, Literal

import numpy as np
import torch
import wandb
from tqdm import tqdm
from catanatron.models.player import RandomPlayer
from catanatron.players.value import ValueFunctionPlayer
from sklearn.metrics import f1_score

from ..features.catanatron_utils import COLOR_ORDER
from ..models.wrappers import PolicyNetworkWrapper, ValueNetworkWrapper
from ..players.nn_policy_player import NNPolicyPlayer
from .eval_nn_vs_catanatron import eval
from .vectorized_rollout import run_policy_value_eval_vectorized


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
    deterministic: bool = True,
    compare_to_expert: bool = False,
    expert_config: Optional[str] = None,
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

    # Define opponent types to evaluate against
    opponent_configs_list = [
        ("random", ["random"]),
        ("value", ["F"]),
    ]

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
            for opponent_name, opponent_configs in opponent_configs_list:
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
                    deterministic=deterministic,
                    compare_to_expert=compare_to_expert,
                    expert_config=expert_config,
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
        for opponent_name, opponent_configs in opponent_configs_list:
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
                deterministic=deterministic,
                compare_to_expert=compare_to_expert,
                expert_config=expert_config,
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
