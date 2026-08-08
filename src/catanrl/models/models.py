from __future__ import annotations

from ..utils.catanatron_action_space import MapType, get_action_array, get_action_space_size
from .backbones import BackboneConfig, CatanGraphBackbone, create_backbone
from .heads import (
    CatanGraphPolicyHead,
    CatanGraphValueHead,
    CatanGraphWDLValueHead,
    FlatPolicyHead,
    HierarchicalPolicyHead,
    ValueHead,
    WDLValueHead,
)
from .wrappers import PolicyNetworkWrapper, PolicyValueNetworkWrapper, ValueNetworkWrapper


def _attach_hierarchical_metadata(
    model: PolicyNetworkWrapper | PolicyValueNetworkWrapper,
    policy_head: HierarchicalPolicyHead,
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
    num_actions: int,
    value_head_type: str = "scalar",
) -> PolicyValueNetworkWrapper:
    """Create a flat policy-value network wrapper with a shared backbone."""
    action_dim = num_actions
    backbone, feature_dim = create_backbone(backbone_config)
    if isinstance(backbone, CatanGraphBackbone):
        if num_actions != get_action_space_size(
            backbone.config.num_players,
            backbone.config.map_type,  # type: ignore[arg-type]
        ):
            raise ValueError("Catan graph backbone/action-space dimensions disagree")
        policy_head = CatanGraphPolicyHead(
            backbone,
            backbone.config.num_players,
            backbone.config.map_type,
        )
        value_head = _build_graph_value_head(backbone, value_head_type)
    else:
        policy_head = FlatPolicyHead(feature_dim, action_dim)
        value_head = _build_value_head(feature_dim, value_head_type)
    model = PolicyValueNetworkWrapper(backbone, policy_head, value_head)
    model.backbone_config = backbone_config
    model.action_space_size = action_dim
    return model


def build_flat_policy_network(
    backbone_config: BackboneConfig,
    num_actions: int,
) -> PolicyNetworkWrapper:
    action_dim = num_actions
    backbone, feature_dim = create_backbone(backbone_config)
    if isinstance(backbone, CatanGraphBackbone):
        if num_actions != get_action_space_size(
            backbone.config.num_players,
            backbone.config.map_type,  # type: ignore[arg-type]
        ):
            raise ValueError("Catan graph backbone/action-space dimensions disagree")
        policy_head = CatanGraphPolicyHead(
            backbone,
            backbone.config.num_players,
            backbone.config.map_type,
        )
    else:
        policy_head = FlatPolicyHead(feature_dim, action_dim)
    model = PolicyNetworkWrapper(backbone, policy_head)
    model.backbone_config = backbone_config
    model.action_space_size = action_dim
    return model


def build_hierarchical_policy_network(
    backbone_config: BackboneConfig,
    num_players: int,
    map_type: MapType,
) -> PolicyNetworkWrapper:
    if backbone_config.architecture == "catan_graph":
        raise ValueError("Catan graph policy heads require the flat action model")
    backbone, feature_dim = create_backbone(backbone_config)
    policy_head = HierarchicalPolicyHead(
        feature_dim,
        get_action_array(num_players, map_type),
    )
    model = PolicyNetworkWrapper(backbone, policy_head)
    model.backbone_config = backbone_config
    model.action_space_size = get_action_space_size(num_players, map_type)
    _attach_hierarchical_metadata(model, policy_head)
    return model


def build_hierarchical_policy_value_network(
    backbone_config: BackboneConfig,
    num_players: int,
    map_type: MapType,
    value_head_type: str = "scalar",
) -> PolicyValueNetworkWrapper:
    """Create a hierarchical policy-value network wrapper with a shared backbone."""

    if backbone_config.architecture == "catan_graph":
        raise ValueError("Catan graph policy heads require the flat action model")
    backbone, feature_dim = create_backbone(backbone_config)
    policy_head = HierarchicalPolicyHead(
        feature_dim,
        get_action_array(num_players, map_type),
    )
    value_head = _build_value_head(feature_dim, value_head_type)

    model = PolicyValueNetworkWrapper(backbone, policy_head, value_head)
    model.backbone_config = backbone_config
    model.action_space_size = get_action_space_size(num_players, map_type)
    _attach_hierarchical_metadata(model, policy_head)
    return model


def build_value_network(
    backbone_config: BackboneConfig,
) -> ValueNetworkWrapper:
    backbone, feature_dim = create_backbone(backbone_config)
    value_head = (
        CatanGraphValueHead(backbone)
        if isinstance(backbone, CatanGraphBackbone)
        else ValueHead(feature_dim)
    )
    model = ValueNetworkWrapper(backbone, value_head)
    model.backbone_config = backbone_config
    return model


def _build_value_head(feature_dim: int, value_head_type: str) -> ValueHead | WDLValueHead:
    if value_head_type == "scalar":
        return ValueHead(feature_dim)
    if value_head_type == "wdl":
        return WDLValueHead(feature_dim)
    raise ValueError(f"Unknown value_head_type '{value_head_type}'")


def _build_graph_value_head(
    backbone: CatanGraphBackbone,
    value_head_type: str,
) -> CatanGraphValueHead | CatanGraphWDLValueHead:
    if value_head_type == "scalar":
        return CatanGraphValueHead(backbone)
    if value_head_type == "wdl":
        return CatanGraphWDLValueHead(backbone)
    raise ValueError(f"Unknown value_head_type '{value_head_type}'")
