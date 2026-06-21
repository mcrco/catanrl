"""Build policy, critic, and joint policy-value networks for training and inference."""

from __future__ import annotations

from typing import Literal, Sequence, Tuple

import torch

from catanrl.envs.puffer.common import BOARD_HEIGHT, BOARD_WIDTH, compute_single_agent_dims
from catanrl.features.catanatron_utils import (
    ActorObservationLevel,
    CriticObservationLevel,
)
from catanrl.utils.catanatron_action_space import get_action_space_size

from .backbone_builder import build_backbone_config
from .models import (
    build_flat_policy_network,
    build_flat_policy_value_network,
    build_hierarchical_policy_network,
    build_hierarchical_policy_value_network,
    build_value_network,
)
from .wrappers import PolicyNetworkWrapper, PolicyValueNetworkWrapper, ValueNetworkWrapper


def build_policy_model(
    backbone_type: str,
    model_type: str,
    hidden_dims: Sequence[int],
    num_players: int,
    map_type: Literal["BASE", "MINI", "TOURNAMENT"],
    actor_observation_level: ActorObservationLevel,
    device: torch.device | str,
    *,
    xdim_cnn_channels: Sequence[int] = (64, 128, 128),
    xdim_cnn_kernel_size: Tuple[int, int] = (3, 5),
    xdim_fusion_hidden_dim: int | None = None,
) -> PolicyNetworkWrapper:
    dims = compute_single_agent_dims(
        num_players,
        map_type,
        actor_observation_level=actor_observation_level,
    )
    backbone_config = build_backbone_config(
        backbone_type=backbone_type,
        hidden_dims=hidden_dims,
        input_dim=dims["actor_dim"],
        board_height=BOARD_HEIGHT,
        board_width=BOARD_WIDTH,
        board_channels=dims["board_channels"],
        numeric_dim=dims["numeric_dim"],
        xdim_cnn_channels=xdim_cnn_channels,
        xdim_cnn_kernel_size=xdim_cnn_kernel_size,
        xdim_fusion_hidden_dim=xdim_fusion_hidden_dim,
    )

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
    *,
    xdim_cnn_channels: Sequence[int] = (64, 128, 128),
    xdim_cnn_kernel_size: Tuple[int, int] = (3, 5),
    xdim_fusion_hidden_dim: int | None = None,
) -> ValueNetworkWrapper:
    dims = compute_single_agent_dims(
        num_players,
        map_type,
        critic_observation_level=critic_observation_level,
    )
    backbone_config = build_backbone_config(
        backbone_type=backbone_type,
        hidden_dims=hidden_dims,
        input_dim=dims["critic_dim"],
        board_height=BOARD_HEIGHT,
        board_width=BOARD_WIDTH,
        board_channels=dims["board_channels"],
        numeric_dim=dims["critic_numeric_dim"],
        xdim_cnn_channels=xdim_cnn_channels,
        xdim_cnn_kernel_size=xdim_cnn_kernel_size,
        xdim_fusion_hidden_dim=xdim_fusion_hidden_dim,
    )

    model = build_value_network(backbone_config=backbone_config)
    return model.to(device)


def build_policy_value_model(
    backbone_type: str,
    model_type: str,
    hidden_dims: Sequence[int],
    num_players: int,
    map_type: Literal["BASE", "MINI", "TOURNAMENT"],
    actor_observation_level: ActorObservationLevel,
    critic_observation_level: CriticObservationLevel,
    device: torch.device | str,
    *,
    xdim_cnn_channels: Sequence[int] = (64, 128, 128),
    xdim_cnn_kernel_size: Tuple[int, int] = (3, 5),
    xdim_fusion_hidden_dim: int | None = None,
) -> PolicyValueNetworkWrapper:
    """Build a joint policy-value network with a shared backbone."""
    dims = compute_single_agent_dims(
        num_players,
        map_type,
        actor_observation_level=actor_observation_level,
        critic_observation_level=critic_observation_level,
    )
    backbone_config = build_backbone_config(
        backbone_type=backbone_type,
        hidden_dims=hidden_dims,
        input_dim=dims["actor_dim"],
        board_height=BOARD_HEIGHT,
        board_width=BOARD_WIDTH,
        board_channels=dims["board_channels"],
        numeric_dim=dims["numeric_dim"],
        xdim_cnn_channels=xdim_cnn_channels,
        xdim_cnn_kernel_size=xdim_cnn_kernel_size,
        xdim_fusion_hidden_dim=xdim_fusion_hidden_dim,
    )

    if model_type == "flat":
        model = build_flat_policy_value_network(
            backbone_config=backbone_config,
            num_actions=get_action_space_size(num_players, map_type),
        )
    elif model_type == "hierarchical":
        model = build_hierarchical_policy_value_network(
            backbone_config=backbone_config,
            num_players=num_players,
            map_type=map_type,
        )
    else:
        raise ValueError(f"Unknown model_type '{model_type}'")

    return model.to(device)


__all__ = ["build_policy_model", "build_critic_model", "build_policy_value_model"]
