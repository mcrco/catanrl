"""
Shared evaluation utilities for PPO training loops.

Provides a unified interface for evaluating trained policies against
standard Catanatron opponents (RandomPlayer, ValueFunctionPlayer).
"""

from typing import Dict, Optional

import wandb
from catanatron.models.player import RandomPlayer
from catanatron.players.value import ValueFunctionPlayer

from ..features.catanatron_utils import COLOR_ORDER
from ..models.wrappers import PolicyNetworkWrapper
from ..players.nn_policy_player import NNPolicyPlayer
from .eval_nn_vs_catanatron import eval


def evaluate_against_baselines(
    policy_model: PolicyNetworkWrapper,
    model_type: str,
    map_type: str,
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

