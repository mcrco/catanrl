import torch
import torch.nn as nn


def compute_hierarchical_loss(
    self,
    action_type_logits,
    param_logits,
    value_pred,
    target_actions,
    target_values,
    action_type_weight: float = 1.0,
    param_weight: float = 1.0,
    value_weight: float = 1.0,
):
    """
    Compute loss for hierarchical policy head.

    Args:
        action_type_logits: [batch_size, 13]
        param_logits: Dict of parameter logits
        value_pred: [batch_size]
        target_actions: [batch_size] flat action indices
        target_values: [batch_size] target values
        action_type_weight: Weight for action type loss
        param_weight: Weight for parameter loss
        value_weight: Weight for value loss

    Returns:
        total_loss, action_type_loss, param_loss, value_loss
    """
    batch_size = target_actions.shape[0]
    device = action_type_logits.device

    # Convert flat actions to hierarchical (action_type, param_idx)
    target_action_types = torch.zeros(batch_size, dtype=torch.long, device=device)
    target_param_indices = {}

    for i in range(batch_size):
        flat_idx = target_actions[i].item()
        action_type_idx, param_idx = self.flat_to_hierarchical[flat_idx]
        target_action_types[i] = action_type_idx

        # Group by action type for efficient loss computation
        if action_type_idx not in target_param_indices:
            target_param_indices[action_type_idx] = []
        target_param_indices[action_type_idx].append((i, param_idx))

    # Action type loss
    action_type_loss = nn.functional.cross_entropy(action_type_logits, target_action_types)

    # Parameter loss (only for samples with parameterized actions)
    param_loss = 0.0
    num_param_samples = 0

    for action_type_idx, samples in target_param_indices.items():
        if action_type_idx in param_logits:
            batch_indices = [s[0] for s in samples]
            param_targets = torch.tensor([s[1] for s in samples], dtype=torch.long, device=device)

            # Extract logits for these samples
            logits_for_type = param_logits[action_type_idx][batch_indices]
            param_loss += nn.functional.cross_entropy(
                logits_for_type, param_targets, reduction="sum"
            )
            num_param_samples += len(samples)

    if num_param_samples > 0:
        param_loss = param_loss / num_param_samples

    # Value loss (MSE)
    value_loss = nn.functional.mse_loss(value_pred, target_values)

    # Total loss
    total_loss = (
        action_type_weight * action_type_loss
        + param_weight * param_loss
        + value_weight * value_loss
    )

    return total_loss, action_type_loss, param_loss, value_loss
