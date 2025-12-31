from __future__ import annotations

from typing import Sequence

from catanatron.gym.envs.catanatron_env import ACTION_SPACE_SIZE

from .backbones import BackboneConfig, MLPBackboneConfig, create_backbone
from .heads import FlatPolicyHead, HierarchicalPolicyHead, ValueHead
from .wrappers import PolicyValueNetworkWrapper

DEFAULT_HIDDEN_DIMS: Sequence[int] = (512, 512)


def _resolve_backbone_config(
    backbone_config: BackboneConfig | None,
    input_dim: int | None,
    hidden_dims: Sequence[int] | None,
) -> BackboneConfig:
    if backbone_config is not None:
        return backbone_config

    if input_dim is None:
        raise ValueError("input_dim must be provided when backbone_config is None")

    dims = list(hidden_dims) if hidden_dims is not None else list(DEFAULT_HIDDEN_DIMS)
    if not dims:
        raise ValueError("hidden_dims must include at least one dimension")

    return BackboneConfig(
        architecture="mlp",
        args=MLPBackboneConfig(input_dim=input_dim, hidden_dims=list(dims)),
    )


def _attach_hierarchical_metadata(
    model: PolicyValueNetworkWrapper, policy_head: HierarchicalPolicyHead
) -> None:
    model.flat_to_hierarchical = policy_head.flat_to_hierarchical
    model.hierarchical_to_flat = policy_head.hierarchical_to_flat
    model.action_space_size = policy_head.action_space_size
    model.NUM_ACTION_TYPES = policy_head.NUM_ACTION_TYPES
    model.NUM_RESOURCES = policy_head.NUM_RESOURCES
    model.num_tiles = policy_head.num_tiles
    model.num_edges = policy_head.num_edges
    model.num_nodes = policy_head.num_nodes
    model.num_year_of_plenty_combos = policy_head.num_year_of_plenty_combos
    model.num_maritime_trades = policy_head.num_maritime_trades
    model.num_discard_combinations = policy_head.num_discard_combinations


def build_flat_policy_value_network(
    backbone_config: BackboneConfig | None = None,
    num_actions: int | None = None,
    *,
    input_dim: int | None = None,
    hidden_dims: Sequence[int] | None = None,
) -> PolicyValueNetworkWrapper:
    """Create a flat policy-value network wrapper with a shared backbone."""

    resolved_config = _resolve_backbone_config(backbone_config, input_dim, hidden_dims)
    action_dim = num_actions or ACTION_SPACE_SIZE

    backbone, feature_dim = create_backbone(resolved_config)
    policy_head = FlatPolicyHead(feature_dim, action_dim)
    value_head = ValueHead(feature_dim)

    model = PolicyValueNetworkWrapper(backbone, policy_head, value_head)
    model.backbone_config = resolved_config
    model.action_space_size = action_dim
    return model


def build_hierarchical_policy_value_network(
    backbone_config: BackboneConfig | None = None,
    *,
    input_dim: int | None = None,
    hidden_dims: Sequence[int] | None = None,
) -> PolicyValueNetworkWrapper:
    """Create a hierarchical policy-value network wrapper with a shared backbone."""

    resolved_config = _resolve_backbone_config(backbone_config, input_dim, hidden_dims)

    backbone, feature_dim = create_backbone(resolved_config)
    policy_head = HierarchicalPolicyHead(feature_dim)
    value_head = ValueHead(feature_dim)

    model = PolicyValueNetworkWrapper(backbone, policy_head, value_head)
    model.backbone_config = resolved_config
    _attach_hierarchical_metadata(model, policy_head)
    return model
