import torch
import numpy as np
from typing import Tuple
from ..models.models import PolicyValueNetwork

class SARLAgent:
    """Agent wrapper for PolicyValueNetwork with action selection."""

    def __init__(self, model: PolicyValueNetwork, device: str):
        self.model = model
        self.device = device

    def select_action(
        self, state: torch.Tensor, valid_actions: np.ndarray, deterministic: bool = False
    ) -> Tuple[int, float, float]:
        """
        Select action using the policy network.

        Returns:
            action_idx: Index in ACTION_SPACE_SIZE
            log_prob: Log probability of selected action
            value: Value estimate
        """
        self.model.eval()  # Ensure eval mode for BatchNorm
        with torch.no_grad():
            policy_logits, value = self.model(state)  # [1, num_actions], [1]

            # Check for NaN in model outputs
            if torch.isnan(policy_logits).any() or torch.isnan(value).any():
                print("WARNING: NaN detected in model output!")
                # Fallback to random action
                action_idx = np.random.choice(valid_actions)
                log_prob = -np.log(len(valid_actions))
                return action_idx, log_prob, 0.0

            # Mask invalid actions
            mask = torch.full_like(policy_logits, float("-inf"))
            mask[0, valid_actions] = 0.0
            masked_logits = policy_logits + mask

            # Clamp logits to prevent overflow in softmax
            masked_logits = torch.clamp(masked_logits, min=-100, max=100)

            # Get action probabilities
            probs = F.softmax(masked_logits, dim=-1)  # [1, num_actions]

            if deterministic:
                action_idx = torch.argmax(probs[0, valid_actions]).item()
                action_idx = valid_actions[action_idx]
            else:
                # Sample from valid actions only
                valid_probs = probs[0, valid_actions]
                valid_probs = valid_probs / valid_probs.sum()  # Renormalize
                dist = torch.distributions.Categorical(probs=valid_probs)
                sampled_idx = dist.sample().item()
                action_idx = valid_actions[sampled_idx]

            # Calculate log probability for the selected action
            log_probs_all = F.log_softmax(masked_logits, dim=-1)  # [1, num_actions]
            log_prob = log_probs_all[0, action_idx].item()

        return action_idx, log_prob, value.item()

    def select_actions_batch(
        self, states: torch.Tensor, valid_actions_batch: List[np.ndarray], deterministic: bool = False
    ) -> Tuple[List[int], List[float], List[float]]:
        """
        Batched action selection for vectorized environments.
        Args:
            states: [batch, input_dim]
            valid_actions_batch: list of np.ndarray of valid action indices per env
        Returns:
            actions: list[int] length batch
            log_probs: list[float] length batch
            values: list[float] length batch
        """
        self.model.eval()
        actions: List[int] = []
        log_probs_out: List[float] = []
        values_out: List[float] = []
        with torch.no_grad():
            policy_logits, values = self.model(states)  # [B, A], [B]
            batch_size, num_actions = policy_logits.shape

            # Build mask
            masks = torch.full((batch_size, num_actions), float("-inf"), device=self.device)
            for i, valid_acts in enumerate(valid_actions_batch):
                masks[i, valid_acts] = 0.0

            masked_logits = policy_logits + masks
            masked_logits = torch.clamp(masked_logits, min=-100, max=100)

            probs = F.softmax(masked_logits, dim=-1)  # [B, A]
            log_probs_all = F.log_softmax(masked_logits, dim=-1)  # [B, A]

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
        self, states: torch.Tensor, actions: torch.Tensor, valid_actions_batch: List[np.ndarray]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions for PPO update.

        Returns:
            log_probs: Log probabilities of actions [batch_size]
            values: Value estimates [batch_size]
            entropy: Policy entropy [batch_size]
        """
        policy_logits, values = self.model(states)  # [batch_size, num_actions], [batch_size]

        # Create masks for each sample in batch
        batch_size = states.shape[0]
        num_actions = policy_logits.shape[1]
        masks = torch.full((batch_size, num_actions), float("-inf"), device=self.device)
        for i, valid_acts in enumerate(valid_actions_batch):
            masks[i, valid_acts] = 0.0

        masked_logits = policy_logits + masks

        # Clamp for numerical stability
        masked_logits = torch.clamp(masked_logits, min=-100, max=100)

        log_probs_all = F.log_softmax(masked_logits, dim=-1)  # [batch_size, num_actions]

        # Get log probs for taken actions
        log_probs = log_probs_all.gather(1, actions.unsqueeze(1)).squeeze(1)  # [batch_size]

        # Calculate entropy
        probs = F.softmax(masked_logits, dim=-1)  # [batch_size, num_actions]
        entropy = -(probs * log_probs_all).sum(dim=-1)  # [batch_size]

        return log_probs, values, entropy
