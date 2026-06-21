"""Shared inference helpers for policy and value networks."""

from __future__ import annotations

import torch

from .wrappers import PolicyNetworkWrapper, PolicyValueNetworkWrapper, ValueNetworkWrapper


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
