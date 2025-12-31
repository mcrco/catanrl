"""
Loss utilities for training different types of Catan policy-value networks.
Supports both flat action space models and hierarchical models with configurable class weights.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional, Tuple
from collections import Counter


def compute_action_weights(
    actions: np.ndarray,
    action_space_size: int,
    weight_power: float = 0.5,
    min_weight: float = 0.1,
    max_weight: float = 10.0,
) -> torch.Tensor:
    """
    Compute class weights for action distribution to handle imbalanced actions.

    Common actions (like ROLL, END_TURN) will get lower weights, rare actions get higher weights.

    Args:
        actions: Array of action indices from training data
        action_space_size: Total number of actions in action space
        weight_power: Power to raise inverse frequency to (0.5 = sqrt, 1.0 = inverse frequency)
                     Lower values = less aggressive reweighting
        min_weight: Minimum weight for any action class
        max_weight: Maximum weight for any action class

    Returns:
        Tensor of weights [action_space_size] for use in CrossEntropyLoss
    """
    # Count action frequencies
    action_counts = Counter(actions)

    # Initialize weights array
    weights = np.ones(action_space_size)

    # Compute inverse frequency weights
    total_samples = len(actions)
    for action_idx in range(action_space_size):
        count = action_counts.get(action_idx, 0)
        if count > 0:
            # Inverse frequency: more common actions get lower weight
            frequency = count / total_samples
            weight = (1.0 / frequency) ** weight_power
            weights[action_idx] = weight
        else:
            # Actions never seen get high weight (but capped)
            weights[action_idx] = max_weight

    # Normalize so mean weight is 1.0
    weights = weights / weights.mean()

    # Clip to min/max range
    weights = np.clip(weights, min_weight, max_weight)

    return torch.FloatTensor(weights)


def compute_hierarchical_action_weights(
    flat_actions: np.ndarray,
    flat_to_hierarchical_map: Dict[int, Tuple[int, int]],
    num_action_types: int,
    param_space_sizes: Dict[int, int],
    weight_power: float = 0.5,
    min_weight: float = 0.1,
    max_weight: float = 10.0,
) -> Tuple[torch.Tensor, Dict[int, torch.Tensor]]:
    """
    Compute class weights for hierarchical action space.

    Args:
        flat_actions: Array of flat action indices from training data
        flat_to_hierarchical_map: Mapping from flat_idx -> (action_type_idx, param_idx)
        num_action_types: Number of action types (e.g., 13)
        param_space_sizes: Dict mapping action_type_idx -> number of parameters
        weight_power: Power for inverse frequency weighting
        min_weight: Minimum weight
        max_weight: Maximum weight

    Returns:
        action_type_weights: [num_action_types]
        param_weights: Dict[action_type_idx] -> [param_space_size]
    """
    # Convert flat actions to hierarchical
    action_types = []
    param_indices_by_type = {i: [] for i in range(num_action_types)}

    for flat_action in flat_actions:
        action_type_idx, param_idx = flat_to_hierarchical_map[int(flat_action)]
        action_types.append(action_type_idx)
        param_indices_by_type[action_type_idx].append(param_idx)

    # Compute action type weights
    action_type_weights = compute_action_weights(
        np.array(action_types), num_action_types, weight_power, min_weight, max_weight
    )

    # Compute parameter weights for each action type
    param_weights = {}
    for action_type_idx, param_space_size in param_space_sizes.items():
        if len(param_indices_by_type[action_type_idx]) > 0:
            param_weights[action_type_idx] = compute_action_weights(
                np.array(param_indices_by_type[action_type_idx]),
                param_space_size,
                weight_power,
                min_weight,
                max_weight,
            )
        else:
            # No samples for this action type, use uniform weights
            param_weights[action_type_idx] = torch.ones(param_space_size)

    return action_type_weights, param_weights


class PolicyValueLoss:
    """Unified loss computation for policy-value networks."""

    def __init__(
        self,
        model_type: str = "flat",  # 'flat' or 'hierarchical'
        policy_weight: float = 1.0,
        value_weight: float = 1.0,
        action_weights: Optional[torch.Tensor] = None,
        device: str = "cuda",
    ):
        """
        Initialize loss computer.

        Args:
            model_type: 'flat' for standard policy/value wrappers, 'hierarchical' otherwise
            policy_weight: Weight for policy loss in total loss
            value_weight: Weight for value loss in total loss
            action_weights: Class weights for action distribution (for flat model)
            device: Device for computation
        """
        self.model_type = model_type
        self.policy_weight = policy_weight
        self.value_weight = value_weight
        self.device = device

        # Value loss is always MSE
        self.value_criterion = nn.MSELoss()

        if model_type == "flat":
            # Standard cross entropy for flat action space
            if action_weights is not None:
                action_weights = action_weights.to(device)
            self.policy_criterion = nn.CrossEntropyLoss(weight=action_weights)

        elif model_type == "hierarchical":
            # Will be set up with model-specific info
            self.action_type_criterion = None
            self.param_criteria = {}
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

    def setup_hierarchical_weights(
        self, action_type_weights: torch.Tensor, param_weights: Dict[int, torch.Tensor]
    ):
        """
        Setup weights for hierarchical model.

        Args:
            action_type_weights: [num_action_types]
            param_weights: Dict[action_type_idx] -> [param_space_size]
        """
        if self.model_type != "hierarchical":
            raise ValueError("Can only call setup_hierarchical_weights for hierarchical models")

        self.action_type_criterion = nn.CrossEntropyLoss(weight=action_type_weights.to(self.device))

        self.param_criteria = {}
        for action_type_idx, weights in param_weights.items():
            self.param_criteria[action_type_idx] = nn.CrossEntropyLoss(
                weight=weights.to(self.device)
            )

    def compute_flat_loss(
        self,
        policy_logits: torch.Tensor,
        value_pred: torch.Tensor,
        target_actions: torch.Tensor,
        target_values: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute loss for flat policy-value network.

        Args:
            policy_logits: [batch_size, action_space_size]
            value_pred: [batch_size]
            target_actions: [batch_size] flat action indices
            target_values: [batch_size] target returns

        Returns:
            total_loss, policy_loss, value_loss
        """
        policy_loss = self.policy_criterion(policy_logits, target_actions)
        value_loss = self.value_criterion(value_pred, target_values)
        total_loss = self.policy_weight * policy_loss + self.value_weight * value_loss

        return total_loss, policy_loss, value_loss

    def compute_hierarchical_loss(
        self,
        action_type_logits: torch.Tensor,
        param_logits: Dict[int, torch.Tensor],
        value_pred: torch.Tensor,
        target_actions: torch.Tensor,
        target_values: torch.Tensor,
        flat_to_hierarchical_map: Dict[int, Tuple[int, int]],
        action_type_weight: float = 1.0,
        param_weight: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute loss for hierarchical policy-value network.

        Args:
            action_type_logits: [batch_size, num_action_types]
            param_logits: Dict[action_type_idx] -> [batch_size, param_space_size]
            value_pred: [batch_size]
            target_actions: [batch_size] flat action indices
            target_values: [batch_size] target returns
            flat_to_hierarchical_map: Mapping from flat_idx -> (action_type_idx, param_idx)
            action_type_weight: Weight for action type loss
            param_weight: Weight for parameter loss

        Returns:
            total_loss, action_type_loss, param_loss, value_loss
        """
        batch_size = target_actions.shape[0]

        # Convert flat actions to hierarchical (action_type, param_idx)
        target_action_types = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        target_param_indices = {}

        for i in range(batch_size):
            flat_idx = target_actions[i].item()
            action_type_idx, param_idx = flat_to_hierarchical_map[flat_idx]
            target_action_types[i] = action_type_idx

            # Group by action type for efficient loss computation
            if action_type_idx not in target_param_indices:
                target_param_indices[action_type_idx] = []
            target_param_indices[action_type_idx].append((i, param_idx))

        # Action type loss
        action_type_loss = self.action_type_criterion(action_type_logits, target_action_types)

        # Parameter loss (only for samples with parameterized actions)
        param_loss = 0.0
        num_param_samples = 0

        for action_type_idx, samples in target_param_indices.items():
            if action_type_idx in param_logits:
                batch_indices = [s[0] for s in samples]
                param_targets = torch.tensor(
                    [s[1] for s in samples], dtype=torch.long, device=self.device
                )

                # Extract logits for these samples
                logits_for_type = param_logits[action_type_idx][batch_indices]

                # Use weighted criterion if available
                if action_type_idx in self.param_criteria:
                    param_loss += self.param_criteria[action_type_idx](
                        logits_for_type, param_targets
                    ) * len(samples)
                else:
                    param_loss += nn.functional.cross_entropy(
                        logits_for_type, param_targets, reduction="sum"
                    )

                num_param_samples += len(samples)

        if num_param_samples > 0:
            param_loss = param_loss / num_param_samples

        # Value loss
        value_loss = self.value_criterion(value_pred, target_values)

        # Total loss
        total_loss = (
            action_type_weight * action_type_loss
            + param_weight * param_loss
            + self.value_weight * value_loss
        )

        return total_loss, action_type_loss, param_loss, value_loss

    def __call__(self, *args, **kwargs):
        """Route to appropriate loss computation based on model type."""
        if self.model_type == "flat":
            return self.compute_flat_loss(*args, **kwargs)
        elif self.model_type == "hierarchical":
            return self.compute_hierarchical_loss(*args, **kwargs)
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")


def create_loss_computer(
    model,
    train_actions: Optional[np.ndarray] = None,
    policy_weight: float = 1.0,
    value_weight: float = 4.0,
    action_type_weight: float = 1.0,
    param_weight: float = 1.0,
    use_class_weights: bool = True,
    weight_power: float = 0.5,
    device: str = "cuda",
) -> PolicyValueLoss:
    """
    Factory function to create appropriate loss computer for a model.

    Args:
        model: The policy-value network wrapper (flat or hierarchical)
        train_actions: Training action indices for computing class weights (optional)
        policy_weight: Weight for policy loss
        value_weight: Weight for value loss
        action_type_weight: Weight for action type loss (hierarchical only)
        param_weight: Weight for parameter loss (hierarchical only)
        use_class_weights: Whether to compute and use class weights
        weight_power: Power for inverse frequency weighting
        device: Device for computation

    Returns:
        PolicyValueLoss instance configured for the model
    """
    model_type = None

    # Detect model type (hierarchical models expose flat_to_hierarchical mapping)
    if hasattr(model, "flat_to_hierarchical"):
        model_type = "hierarchical"
    elif hasattr(model, "policy_head") and hasattr(model, "value_head"):
        model_type = "flat"
    else:
        raise ValueError("Unable to detect model type")

    if model_type == "flat":
        # Compute action weights if requested
        action_weights = None
        if use_class_weights and train_actions is not None:
            from catanatron.gym.envs.catanatron_env import ACTION_SPACE_SIZE

            action_weights = compute_action_weights(
                train_actions, ACTION_SPACE_SIZE, weight_power=weight_power
            )
            print(f"Computed class weights for flat action space")
            print(f"  Weight range: [{action_weights.min():.3f}, {action_weights.max():.3f}]")

        loss_computer = PolicyValueLoss(
            model_type="flat",
            policy_weight=policy_weight,
            value_weight=value_weight,
            action_weights=action_weights,
            device=device,
        )

    elif model_type == "hierarchical":
        loss_computer = PolicyValueLoss(
            model_type="hierarchical",
            policy_weight=policy_weight,
            value_weight=value_weight,
            device=device,
        )

        # Compute hierarchical weights if requested
        if use_class_weights and train_actions is not None:
            # Get parameter space sizes (dynamically determined from ACTIONS_ARRAY)
            param_space_sizes = {}
            if hasattr(model, "tile_head"):
                param_space_sizes[model.MOVE_ROBBER] = model.num_tiles
            if hasattr(model, "edge_head"):
                param_space_sizes[model.BUILD_ROAD] = model.num_edges
            if hasattr(model, "settlement_node_head"):
                param_space_sizes[model.BUILD_SETTLEMENT] = model.num_nodes
            if hasattr(model, "city_node_head"):
                param_space_sizes[model.BUILD_CITY] = model.num_nodes
            if hasattr(model, "year_of_plenty_head"):
                param_space_sizes[model.PLAY_YEAR_OF_PLENTY] = model.num_year_of_plenty_combos
            if hasattr(model, "monopoly_head"):
                param_space_sizes[model.PLAY_MONOPOLY] = model.NUM_RESOURCES
            if hasattr(model, "maritime_trade_head"):
                param_space_sizes[model.MARITIME_TRADE] = model.num_maritime_trades

            action_type_weights, param_weights = compute_hierarchical_action_weights(
                train_actions,
                model.flat_to_hierarchical,
                model.NUM_ACTION_TYPES,
                param_space_sizes,
                weight_power=weight_power,
            )

            loss_computer.setup_hierarchical_weights(action_type_weights, param_weights)

            print(f"Computed class weights for hierarchical action space")
            print(
                f"  Action type weight range: [{action_type_weights.min():.3f}, {action_type_weights.max():.3f}]"
            )
            print(f"  Parameter weight ranges by action type:")
            for action_type_idx, weights in param_weights.items():
                print(f"    Type {action_type_idx}: [{weights.min():.3f}, {weights.max():.3f}]")

        # Store additional weights for hierarchical loss
        loss_computer.action_type_weight = action_type_weight
        loss_computer.param_weight = param_weight

    return loss_computer
