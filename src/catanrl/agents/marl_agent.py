from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch

from catanatron.gym.envs.catanatron_env import ACTION_SPACE_SIZE

from ..models.models import PolicyValueNetwork


class MultiAgentRLAgent:
    """Shared policy/value agent for multi-agent self-play training."""

    def __init__(self, model: PolicyValueNetwork, device: str):
        self.model = model
        self.device = device

    def select_action(
        self,
        state_vec: np.ndarray,
        valid_actions: np.ndarray,
        deterministic: bool = False,
    ) -> Tuple[int, float, float]:
        """Select an action for the current PettingZoo agent."""
        if valid_actions.size == 0:
            valid_actions = np.arange(ACTION_SPACE_SIZE, dtype=np.int64)

        state_tensor = torch.from_numpy(state_vec).float().unsqueeze(0).to(self.device)
        self.model.eval()
        with torch.no_grad():
            policy_logits, value = self.model(state_tensor)

            mask = torch.full_like(policy_logits, float("-inf"))
            mask[0, valid_actions] = 0.0
            masked_logits = torch.clamp(policy_logits + mask, min=-100, max=100)

            probs = torch.softmax(masked_logits, dim=-1)
            log_probs_all = torch.log_softmax(masked_logits, dim=-1)

            if deterministic:
                action_idx = torch.argmax(probs[0, valid_actions]).item()
                action = int(valid_actions[action_idx])
            else:
                valid_probs = probs[0, valid_actions]
                valid_probs = valid_probs / (valid_probs.sum() + 1e-12)
                dist = torch.distributions.Categorical(valid_probs)
                action = int(valid_actions[dist.sample().item()])

            log_prob = float(log_probs_all[0, action].item())
            value_out = float(value.item())

        return action, log_prob, value_out

    def evaluate_actions(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        valid_actions_batch: List[np.ndarray],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate log-probs, values, and entropy for PPO updates."""
        policy_logits, values = self.model(states)
        batch_size, num_actions = policy_logits.shape

        masks = torch.full(
            (batch_size, num_actions), float("-inf"), device=states.device
        )
        for idx, valid in enumerate(valid_actions_batch):
            if valid.size == 0:
                masks[idx] = 0.0
            else:
                masks[idx, valid] = 0.0

        masked_logits = torch.clamp(policy_logits + masks, min=-100, max=100)
        log_probs_all = torch.log_softmax(masked_logits, dim=-1)
        probs = torch.softmax(masked_logits, dim=-1)

        log_probs = log_probs_all.gather(1, actions.unsqueeze(1)).squeeze(1)
        entropy = -(probs * log_probs_all).sum(dim=-1)
        return log_probs, values.squeeze(-1), entropy

