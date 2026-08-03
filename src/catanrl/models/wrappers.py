import torch.nn as nn
from typing import Any

from .heads import FlatPolicyHead, HierarchicalPolicyHead, ValueHead, WDLValueHead


class PolicyNetworkWrapper(nn.Module):
    """Wrapper for policy network."""

    backbone_config: Any
    action_space_size: int
    flat_to_hierarchical: dict[Any, Any]
    hierarchical_to_flat: dict[Any, Any]
    NUM_ACTION_TYPES: int
    NUM_RESOURCES: int
    num_tiles: int
    num_edges: int
    num_nodes: int
    num_year_of_plenty_combos: int
    num_maritime_trades: int
    num_discard_combinations: int

    def __init__(self, backbone: nn.Module, policy_head: FlatPolicyHead | HierarchicalPolicyHead):
        super().__init__()
        self.backbone = backbone
        self.policy_head = policy_head

    def forward(self, x):
        features = self.backbone(x)
        return self.policy_head(features)

    def get_flat_action_logits(self, action_type_logits, param_logits):
        if isinstance(self.policy_head, HierarchicalPolicyHead):
            return self.policy_head.get_flat_action_logits(action_type_logits, param_logits)
        raise AttributeError("Policy head does not expose get_flat_action_logits")


class ValueNetworkWrapper(nn.Module):
    """Wrapper for value network."""

    backbone_config: Any

    def __init__(self, backbone: nn.Module, value_head: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.value_head = value_head

    def forward(self, x):
        features = self.backbone(x)
        return self.value_head(features)


class PolicyValueNetworkWrapper(nn.Module):
    """Wrapper for joint policy and value networks."""

    backbone_config: Any
    action_space_size: int
    flat_to_hierarchical: dict[Any, Any]
    hierarchical_to_flat: dict[Any, Any]
    NUM_ACTION_TYPES: int
    NUM_RESOURCES: int
    num_tiles: int
    num_edges: int
    num_nodes: int
    num_year_of_plenty_combos: int
    num_maritime_trades: int
    num_discard_combinations: int

    def __init__(
        self,
        backbone: nn.Module,
        policy_head: FlatPolicyHead | HierarchicalPolicyHead,
        value_head: ValueHead | WDLValueHead,
    ):
        super().__init__()
        self.backbone = backbone
        self.policy_head = policy_head
        self.value_head = value_head

    def forward(self, x):
        features = self.backbone(x)
        policy_outputs = self.policy_head(features)
        value_outputs = self.value_head(features)

        if isinstance(policy_outputs, tuple):
            return (*policy_outputs, value_outputs)

        return policy_outputs, value_outputs

    def get_flat_action_logits(self, action_type_logits, param_logits):
        if isinstance(self.policy_head, HierarchicalPolicyHead):
            return self.policy_head.get_flat_action_logits(action_type_logits, param_logits)
        raise AttributeError("Policy head does not expose get_flat_action_logits")


def policy_value_to_policy_only(
    policy_value_network: PolicyValueNetworkWrapper,
) -> PolicyNetworkWrapper:
    return PolicyNetworkWrapper(policy_value_network.backbone, policy_value_network.policy_head)
