"""Builders for the separate policy and critic networks used by the unified
NN MCTS engine (training and evaluation share these)."""

from __future__ import annotations

from typing import Literal, Sequence

import torch

from catanrl.envs.puffer.common import compute_single_agent_dims
from catanrl.features.catanatron_utils import (
    ActorObservationLevel,
    CriticObservationLevel,
)
from catanrl.models.backbones import (
    BackboneConfig,
    CrossDimensionalBackboneConfig,
    MLPBackboneConfig,
)
from catanrl.models.models import (
    build_flat_policy_network,
    build_hierarchical_policy_network,
    build_value_network,
)
from catanrl.models.wrappers import PolicyNetworkWrapper, ValueNetworkWrapper
from catanrl.utils.catanatron_action_space import get_action_space_size

BOARD_WIDTH = 21
BOARD_HEIGHT = 11


def build_policy_model(
    backbone_type: str,
    model_type: str,
    hidden_dims: Sequence[int],
    num_players: int,
    map_type: Literal["BASE", "MINI", "TOURNAMENT"],
    actor_observation_level: ActorObservationLevel,
    device: torch.device | str,
) -> PolicyNetworkWrapper:
    dims = compute_single_agent_dims(
        num_players,
        map_type,
        actor_observation_level=actor_observation_level,
    )
    actor_dim = dims["actor_dim"]
    numeric_dim = dims["numeric_dim"]
    board_channels = dims["board_channels"]

    if backbone_type == "mlp":
        backbone_config = BackboneConfig(
            architecture="mlp",
            args=MLPBackboneConfig(input_dim=actor_dim, hidden_dims=list(hidden_dims)),
        )
    elif backbone_type == "xdim":
        backbone_config = BackboneConfig(
            architecture="cross_dimensional",
            args=CrossDimensionalBackboneConfig(
                board_height=BOARD_HEIGHT,
                board_width=BOARD_WIDTH,
                board_channels=board_channels,
                numeric_dim=numeric_dim,
                cnn_channels=[64, 128, 128],
                cnn_kernel_size=(3, 5),
                numeric_hidden_dims=list(hidden_dims),
                fusion_hidden_dim=hidden_dims[-1] if hidden_dims else 256,
                output_dim=hidden_dims[-1] if hidden_dims else 256,
            ),
        )
    else:
        raise ValueError(f"Unknown backbone_type '{backbone_type}'")

    if model_type == "flat":
        model = build_flat_policy_network(
            backbone_config=backbone_config,
            num_actions=get_action_space_size(num_players, map_type),
        )
    elif model_type == "hierarchical":
        model = build_hierarchical_policy_network(
            backbone_config=backbone_config,
            num_players=num_players,
            map_type=map_type,
        )
    else:
        raise ValueError(f"Unknown model_type '{model_type}'")

    return model.to(device)


def build_critic_model(
    backbone_type: str,
    hidden_dims: Sequence[int],
    num_players: int,
    map_type: Literal["BASE", "MINI", "TOURNAMENT"],
    critic_observation_level: CriticObservationLevel,
    device: torch.device | str,
) -> ValueNetworkWrapper:
    dims = compute_single_agent_dims(
        num_players,
        map_type,
        critic_observation_level=critic_observation_level,
    )
    critic_dim = dims["critic_dim"]
    critic_numeric_dim = dims["critic_numeric_dim"]
    board_channels = dims["board_channels"]

    if backbone_type == "mlp":
        backbone_config = BackboneConfig(
            architecture="mlp",
            args=MLPBackboneConfig(input_dim=critic_dim, hidden_dims=list(hidden_dims)),
        )
    elif backbone_type == "xdim":
        backbone_config = BackboneConfig(
            architecture="cross_dimensional",
            args=CrossDimensionalBackboneConfig(
                board_height=BOARD_HEIGHT,
                board_width=BOARD_WIDTH,
                board_channels=board_channels,
                numeric_dim=critic_numeric_dim,
                cnn_channels=[64, 128, 128],
                cnn_kernel_size=(3, 5),
                numeric_hidden_dims=list(hidden_dims),
                fusion_hidden_dim=hidden_dims[-1] if hidden_dims else 256,
                output_dim=hidden_dims[-1] if hidden_dims else 256,
            ),
        )
    else:
        raise ValueError(f"Unknown backbone_type '{backbone_type}'")

    model = build_value_network(backbone_config=backbone_config)
    return model.to(device)
