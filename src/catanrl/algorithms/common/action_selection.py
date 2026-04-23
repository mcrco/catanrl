from __future__ import annotations

from typing import Tuple

import numpy as np
import numpy.typing as npt
import torch

from ...models.wrappers import PolicyNetworkWrapper, PolicyValueNetworkWrapper


def mask_action_logits(
    policy_logits: torch.Tensor,
    valid_action_masks: npt.NDArray[np.bool_] | np.ndarray,
    clamp_range: Tuple[float, float] | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Mask invalid actions while ensuring every row keeps at least one choice."""
    mask = torch.as_tensor(valid_action_masks, dtype=torch.bool, device=policy_logits.device)
    if mask.ndim == 1:
        mask = mask.unsqueeze(0)
    no_valid = ~mask.any(dim=1, keepdim=True)
    mask = torch.where(no_valid, torch.ones_like(mask), mask)
    masked_logits = torch.where(mask, policy_logits, torch.full_like(policy_logits, float("-inf")))
    if clamp_range is not None:
        min_value, max_value = clamp_range
        masked_logits = torch.clamp(masked_logits, min=min_value, max=max_value)
    return masked_logits, mask


class PolicyAgent:
    """Policy-only helper for masked action selection and log-prob evaluation."""

    def __init__(
        self,
        model: PolicyNetworkWrapper | PolicyValueNetworkWrapper,
        model_type: str,
        device: str | torch.device,
    ):
        self.model = model
        self.model_type = model_type
        self.device = device

    def policy_logits(self, states: torch.Tensor) -> torch.Tensor:
        if self.model_type == "flat":
            policy_outputs = self.model(states)
            if isinstance(policy_outputs, tuple):
                return policy_outputs[0]
            return policy_outputs
        if self.model_type == "hierarchical":
            policy_outputs = self.model(states)
            if len(policy_outputs) == 3:
                action_type_logits, param_logits, _ = policy_outputs
            else:
                action_type_logits, param_logits = policy_outputs
            return self.model.get_flat_action_logits(action_type_logits, param_logits)
        raise ValueError(f"Unknown model_type '{self.model_type}'")

    def select_action(
        self,
        state_vec: np.ndarray,
        valid_action_mask: npt.NDArray[np.bool_] | np.ndarray,
        deterministic: bool = False,
        clamp_range: Tuple[float, float] | None = None,
    ) -> Tuple[int, float]:
        """Select one action from a single masked state."""
        state_tensor = torch.from_numpy(state_vec).float().unsqueeze(0).to(self.device)
        was_training = self.model.training
        self.model.eval()
        try:
            with torch.no_grad():
                policy_logits = self.policy_logits(state_tensor)
                masked_logits, _ = mask_action_logits(
                    policy_logits, valid_action_mask, clamp_range=clamp_range
                )
                dist = torch.distributions.Categorical(logits=masked_logits)
                if deterministic:
                    action_tensor = torch.argmax(masked_logits, dim=-1)
                else:
                    action_tensor = dist.sample()
                action = int(action_tensor.item())
                log_prob = float(dist.log_prob(action_tensor).item())
                return action, log_prob
        finally:
            if was_training:
                self.model.train()

    def select_actions_batch(
        self,
        states: torch.Tensor,
        valid_action_masks: npt.NDArray[np.bool_] | np.ndarray,
        deterministic: bool = False,
        clamp_range: Tuple[float, float] | None = None,
    ) -> Tuple[npt.NDArray[np.int64], npt.NDArray[np.float32], npt.NDArray[np.int64]]:
        """Select masked actions for a batch of actor states."""
        was_training = self.model.training
        self.model.eval()
        try:
            with torch.no_grad():
                policy_logits = self.policy_logits(states)
                raw_argmax_tensor = torch.argmax(policy_logits, dim=-1)
                masked_logits, _ = mask_action_logits(
                    policy_logits, valid_action_masks, clamp_range=clamp_range
                )
                dist = torch.distributions.Categorical(logits=masked_logits)
                if deterministic:
                    action_tensor = torch.argmax(masked_logits, dim=-1)
                else:
                    action_tensor = dist.sample()
                log_prob_tensor = dist.log_prob(action_tensor)
                return (
                    action_tensor.detach().cpu().numpy().astype(np.int64, copy=False),
                    log_prob_tensor.detach().cpu().numpy().astype(np.float32, copy=False),
                    raw_argmax_tensor.detach().cpu().numpy().astype(np.int64, copy=False),
                )
        finally:
            if was_training:
                self.model.train()

    def evaluate_actions(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        valid_action_masks: npt.NDArray[np.bool_] | np.ndarray,
        clamp_range: Tuple[float, float] | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return action log-probs, entropies, and raw logits for PPO updates."""
        policy_logits = self.policy_logits(states)
        masked_logits, _ = mask_action_logits(
            policy_logits, valid_action_masks, clamp_range=clamp_range
        )
        dist = torch.distributions.Categorical(logits=masked_logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, entropy, policy_logits

__all__ = ["PolicyAgent", "mask_action_logits"]
