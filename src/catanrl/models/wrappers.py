import torch.nn as nn


class PolicyNetworkWrapper(nn.Module):
    """Wrapper for policy network."""

    def __init__(self, backbone: nn.Module, policy_head: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.policy_head = policy_head

    def forward(self, x):
        features = self.backbone(x)
        return self.policy_head(features)

    def get_flat_action_logits(self, action_type_logits, param_logits):
        if hasattr(self.policy_head, "get_flat_action_logits"):
            return self.policy_head.get_flat_action_logits(action_type_logits, param_logits)
        raise AttributeError("Policy head does not expose get_flat_action_logits")


class ValueNetworkWrapper(nn.Module):
    """Wrapper for value network."""

    def __init__(self, backbone: nn.Module, value_head: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.value_head = value_head

    def forward(self, x):
        features = self.backbone(x)
        return self.value_head(features)


class PolicyValueNetworkWrapper(nn.Module):
    """Wrapper for joint policy and value networks."""

    def __init__(self, backbone: nn.Module, policy_head: nn.Module, value_head: nn.Module):
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
        if hasattr(self.policy_head, "get_flat_action_logits"):
            return self.policy_head.get_flat_action_logits(action_type_logits, param_logits)
        raise AttributeError("Policy head does not expose get_flat_action_logits")


def policy_value_to_policy_only(
    policy_value_network: PolicyValueNetworkWrapper,
) -> PolicyNetworkWrapper:
    return PolicyNetworkWrapper(policy_value_network.backbone, policy_value_network.policy_head)
