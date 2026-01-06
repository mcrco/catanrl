from __future__ import annotations

from catanatron.gym.envs.catanatron_env import ACTION_SPACE_SIZE

from .backbones import BackboneConfig, create_backbone
from .heads import FlatPolicyHead, HierarchicalPolicyHead, ValueHead
from .wrappers import PolicyNetworkWrapper, PolicyValueNetworkWrapper, ValueNetworkWrapper


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
    backbone_config: BackboneConfig,
    num_actions: int | None = None,
) -> PolicyValueNetworkWrapper:
    """Create a flat policy-value network wrapper with a shared backbone."""
    action_dim = num_actions or ACTION_SPACE_SIZE
    backbone, feature_dim = create_backbone(backbone_config)
    policy_head = FlatPolicyHead(feature_dim, action_dim)
    value_head = ValueHead(feature_dim)
    model = PolicyValueNetworkWrapper(backbone, policy_head, value_head)
    model.backbone_config = backbone_config
    model.action_space_size = action_dim
    return model


def build_flat_policy_network(
    backbone_config: BackboneConfig,
    num_actions: int | None = None,
) -> PolicyNetworkWrapper:
    action_dim = num_actions or ACTION_SPACE_SIZE
    backbone, feature_dim = create_backbone(backbone_config)
    policy_head = FlatPolicyHead(feature_dim, action_dim)
    model = PolicyNetworkWrapper(backbone, policy_head)
    model.backbone_config = backbone_config
    model.action_space_size = action_dim
    return model


def build_hierarchical_policy_network(
    backbone_config: BackboneConfig,
) -> PolicyNetworkWrapper:
    backbone, feature_dim = create_backbone(backbone_config)
    policy_head = HierarchicalPolicyHead(feature_dim)
    model = PolicyNetworkWrapper(backbone, policy_head)
    model.backbone_config = backbone_config
    _attach_hierarchical_metadata(model, policy_head)
    return model


def build_hierarchical_policy_value_network(
    backbone_config: BackboneConfig,
) -> PolicyValueNetworkWrapper:
    """Create a hierarchical policy-value network wrapper with a shared backbone."""

    backbone, feature_dim = create_backbone(backbone_config)
    policy_head = HierarchicalPolicyHead(feature_dim)
    value_head = ValueHead(feature_dim)

    model = PolicyValueNetworkWrapper(backbone, policy_head, value_head)
    model.backbone_config = backbone_config
    _attach_hierarchical_metadata(model, policy_head)
    return model


def build_value_network(
    backbone_config: BackboneConfig,
) -> ValueNetworkWrapper:
    backbone, feature_dim = create_backbone(backbone_config)
    value_head = ValueHead(feature_dim)
    model = ValueNetworkWrapper(backbone, value_head)
    model.backbone_config = backbone_config
    return model
