from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from ..models.models import PolicyValueNetwork, HierarchicalPolicyValueNetwork


class SARLAgent:
    """Agent wrapper around PolicyValueNetwork for single-agent PPO training."""

    def __init__(self, model: PolicyValueNetwork | HierarchicalPolicyValueNetwork, model_type: str, device: str):
        self.model = model
        self.model_type = model_type
        self.device = device

    def select_action(
        self,
        state: torch.Tensor,
        valid_actions: np.ndarray,
        deterministic: bool = False,
    ) -> Tuple[int, float, float]:
        """
        Select one action for a single environment step.

        Returns:
            action_idx: Index in ACTION_SPACE_SIZE
            log_prob: Log probability of selected action
            value: Value estimate
        """
        self.model.eval()
        with torch.no_grad():
            if self.model_type == 'flat':
                policy_logits, value = self.model(state)
            elif self.model_type == 'hierarchical': 
                action_type_logits, param_logits, value = self.model(state)
                policy_logits = self.model.get_flat_action_logits(action_type_logits, param_logits)

            if torch.isnan(policy_logits).any() or torch.isnan(value).any():
                print("WARNING: NaN detected in model output!")
                action_idx = np.random.choice(valid_actions)
                log_prob = -np.log(len(valid_actions))
                return action_idx, log_prob, 0.0

            mask = torch.full_like(policy_logits, float("-inf"))
            mask[0, valid_actions] = 0.0
            masked_logits = torch.clamp(policy_logits + mask, min=-100, max=100)

            probs = F.softmax(masked_logits, dim=-1)
            log_probs_all = F.log_softmax(masked_logits, dim=-1)

            if deterministic:
                action_idx = torch.argmax(probs[0, valid_actions]).item()
                action_idx = valid_actions[action_idx]
            else:
                valid_probs = probs[0, valid_actions]
                valid_probs = valid_probs / valid_probs.sum()
                dist = torch.distributions.Categorical(probs=valid_probs)
                sampled_idx = dist.sample().item()
                action_idx = valid_actions[sampled_idx]

            log_prob = log_probs_all[0, action_idx].item()

        return int(action_idx), float(log_prob), float(value.item())

    def select_actions_batch(
        self,
        states: torch.Tensor,
        valid_actions_batch: List[np.ndarray],
        deterministic: bool = False,
    ) -> Tuple[List[int], List[float], List[float]]:
        """Vectorized action selection for multiple parallel environments."""
        self.model.eval()
        actions: List[int] = []
        log_probs_out: List[float] = []
        values_out: List[float] = []

        with torch.no_grad():
            if self.model_type == 'flat':
                policy_logits, values = self.model(states)
            elif self.model_type == 'hierarchical':
                action_type_logits, param_logits, values = self.model(states)
                policy_logits = self.model.get_flat_action_logits(action_type_logits, param_logits)
            batch_size, num_actions = policy_logits.shape

            masks = torch.full((batch_size, num_actions), float("-inf"), device=self.device)
            for i, valid_acts in enumerate(valid_actions_batch):
                masks[i, valid_acts] = 0.0

            masked_logits = torch.clamp(policy_logits + masks, min=-100, max=100)
            probs = F.softmax(masked_logits, dim=-1)
            log_probs_all = F.log_softmax(masked_logits, dim=-1)

            for i in range(batch_size):
                valid_actions = valid_actions_batch[i]
                if deterministic:
                    local_idx = torch.argmax(probs[i, valid_actions]).item()
                    action_idx = int(valid_actions[local_idx])
                else:
                    valid_probs = probs[i, valid_actions]
                    valid_probs = valid_probs / (valid_probs.sum() + 1e-12)
                    dist = torch.distributions.Categorical(probs=valid_probs)
                    sampled_idx = dist.sample().item()
                    action_idx = int(valid_actions[sampled_idx])

                actions.append(action_idx)
                log_probs_out.append(float(log_probs_all[i, action_idx].item()))
                values_out.append(float(values[i].item()))

        return actions, log_probs_out, values_out

    def evaluate_actions(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        valid_actions_batch: List[np.ndarray],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate actions for PPO updates."""
        if self.model_type == 'flat':
            policy_logits, values = self.model(states)
        elif self.model_type == 'hierarchical':
            action_type_logits, param_logits, values = self.model(states)
            policy_logits = self.model.get_flat_action_logits(action_type_logits, param_logits)

        batch_size = states.shape[0]
        num_actions = policy_logits.shape[1]
        masks = torch.full((batch_size, num_actions), float("-inf"), device=self.device)
        for i, valid_acts in enumerate(valid_actions_batch):
            masks[i, valid_acts] = 0.0

        masked_logits = torch.clamp(policy_logits + masks, min=-100, max=100)
        log_probs_all = F.log_softmax(masked_logits, dim=-1)
        probs = F.softmax(masked_logits, dim=-1)

        log_probs = log_probs_all.gather(1, actions.unsqueeze(1)).squeeze(1)
        entropy = -(probs * log_probs_all).sum(dim=-1)

        return log_probs, values.squeeze(-1), entropy

