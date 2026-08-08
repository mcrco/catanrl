"""Shared inference helpers for policy and value networks."""

from __future__ import annotations

import torch

from .heads import WDLValueHead
from .wrappers import PolicyNetworkWrapper, PolicyValueNetworkWrapper, ValueNetworkWrapper


def values_to_wdl_targets(values: torch.Tensor) -> torch.Tensor:
    """Map scalar Q values to Canopy-compatible W/D/L training distributions."""
    bounded = values.clamp(-1.0, 1.0)
    return torch.stack(
        (
            (1.0 + bounded) * 0.5,
            torch.zeros_like(bounded),
            (1.0 - bounded) * 0.5,
        ),
        dim=-1,
    )


def forward_shared_policy_value_training(
    model: PolicyValueNetworkWrapper,
    states: torch.Tensor,
    model_type: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Return policy logits, scalar values, and optional categorical WDL logits."""
    features = model.backbone(states)
    policy_outputs = model.policy_head(features)
    if model_type == "flat":
        if not isinstance(policy_outputs, torch.Tensor):
            raise TypeError("Flat policy head returned hierarchical outputs.")
        policy_logits = policy_outputs
    elif model_type == "hierarchical":
        if not isinstance(policy_outputs, tuple):
            raise TypeError("Hierarchical policy head returned flat outputs.")
        policy_logits = model.get_flat_action_logits(*policy_outputs)
    else:
        raise ValueError(f"Unknown model_type '{model_type}'")

    if isinstance(model.value_head, WDLValueHead):
        value_logits = model.value_head.logits(features)
        values = WDLValueHead.values_from_logits(value_logits)
        return policy_logits, values.view(-1), value_logits
    return policy_logits, model.value_head(features).view(-1), None


def forward_policy_value(
    model: PolicyNetworkWrapper | PolicyValueNetworkWrapper,
    critic_model: ValueNetworkWrapper | None,
    states: torch.Tensor,
    model_type: str,
    critic_states: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run policy and value inference, using a single forward when ``model`` is joint.

    Returns:
        ``(flat_logits, values)`` where ``values`` has shape ``[batch]``.
    """
    if isinstance(model, PolicyValueNetworkWrapper):
        if model_type == "flat":
            logits, values = model(states)
            return logits, values.view(-1)
        if model_type == "hierarchical":
            action_type_logits, param_logits, values = model(states)
            logits = model.get_flat_action_logits(action_type_logits, param_logits)
            return logits, values.view(-1)
        raise ValueError(f"Unknown model_type '{model_type}'")

    if critic_model is None:
        raise ValueError("critic_model is required when model is not a PolicyValueNetworkWrapper.")

    if model_type == "flat":
        logits = model(states)
    elif model_type == "hierarchical":
        action_type_logits, param_logits = model(states)
        logits = model.get_flat_action_logits(action_type_logits, param_logits)
    else:
        raise ValueError(f"Unknown model_type '{model_type}'")

    value_input = critic_states if critic_states is not None else states
    values = critic_model(value_input).view(-1)
    return logits, values


def forward_policy_value_wdl(
    model: PolicyNetworkWrapper | PolicyValueNetworkWrapper,
    critic_model: ValueNetworkWrapper | None,
    states: torch.Tensor,
    model_type: str,
    critic_states: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Return scalar Q plus full WDL probabilities when the model exposes them."""
    if isinstance(model, PolicyValueNetworkWrapper) and isinstance(
        model.value_head, WDLValueHead
    ):
        logits, values, value_logits = forward_shared_policy_value_training(
            model,
            states,
            model_type,
        )
        assert value_logits is not None
        return logits, values, torch.softmax(value_logits, dim=-1)
    logits, values = forward_policy_value(
        model,
        critic_model,
        states,
        model_type,
        critic_states=critic_states,
    )
    return logits, values, None
