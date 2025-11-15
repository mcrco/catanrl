"""
Metrics tracking for training Catan policy-value networks.
Memory-efficient accumulation using confusion matrices and running counts.
"""

import torch
import numpy as np
from typing import Dict, Optional


class PolicyValueMetrics:
    """
    Memory-efficient metrics tracker for policy-value networks.
    
    Accumulates confusion matrix for policy and TP/FP/FN for value,
    allowing computation of accuracy, F1, and other metrics without
    storing all predictions in memory.
    """
    
    def __init__(self, num_classes: int, device: str = 'cpu'):
        """
        Initialize metrics tracker.
        
        Args:
            num_classes: Number of action classes
            device: Device for tensor operations
        """
        self.num_classes = num_classes
        self.device = device
        self.reset()
    
    def reset(self):
        """Clear all accumulated statistics."""
        # Policy metrics (confusion matrix)
        self.policy_cm = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
        self.policy_total_loss = 0.0
        self.policy_num_batches = 0
        
        # Value metrics (binary classification: positive vs negative return)
        self.value_tp = 0  # True positives (predicted +, actual +)
        self.value_fp = 0  # False positives (predicted +, actual -)
        self.value_fn = 0  # False negatives (predicted -, actual +)
        self.value_tn = 0  # True negatives (predicted -, actual -)
        self.value_total_loss = 0.0
        self.value_num_batches = 0
        
        # Combined loss
        self.total_loss = 0.0
        self.num_batches = 0
        
        # Hierarchical-specific (optional)
        self.action_type_loss = 0.0
        self.param_loss = 0.0
    
    def update(
        self,
        policy_logits: torch.Tensor,
        value_pred: torch.Tensor,
        actions: torch.Tensor,
        returns: torch.Tensor,
        policy_loss: Optional[float] = None,
        value_loss: Optional[float] = None,
        total_loss: Optional[float] = None,
        action_type_loss: Optional[float] = None,
        param_loss: Optional[float] = None,
    ) -> Dict[str, float]:
        """
        Process one batch and return batch-level metrics.
        
        Args:
            policy_logits: [batch_size, num_classes] policy logits
            value_pred: [batch_size] value predictions
            actions: [batch_size] ground truth actions
            returns: [batch_size] ground truth returns
            policy_loss: Policy loss for this batch (optional)
            value_loss: Value loss for this batch (optional)
            total_loss: Combined loss for this batch (optional)
            action_type_loss: Action type loss (hierarchical only, optional)
            param_loss: Parameter loss (hierarchical only, optional)
            
        Returns:
            Dictionary of batch-level metrics
        """
        batch_size = actions.size(0)
        
        # Policy metrics
        _, predicted = torch.max(policy_logits, 1)
        policy_correct = (predicted == actions).sum().item()
        policy_acc = 100.0 * policy_correct / batch_size
        
        # Update confusion matrix
        true_np = actions.cpu().numpy()
        pred_np = predicted.cpu().numpy()
        for t, p in zip(true_np, pred_np):
            self.policy_cm[t, p] += 1
        
        # Value metrics (sign-based: predict winner)
        pred_sign = torch.sign(value_pred)
        true_sign = torch.sign(returns)
        
        pred_binary = (pred_sign > 0).cpu().numpy().astype(int)
        true_binary = (true_sign > 0).cpu().numpy().astype(int)
        
        # Update TP/FP/FN/TN
        self.value_tp += np.sum((pred_binary == 1) & (true_binary == 1))
        self.value_fp += np.sum((pred_binary == 1) & (true_binary == 0))
        self.value_fn += np.sum((pred_binary == 0) & (true_binary == 1))
        self.value_tn += np.sum((pred_binary == 0) & (true_binary == 0))
        
        value_correct = (pred_sign == true_sign).sum().item()
        value_acc = value_correct / batch_size
        
        # Accumulate losses
        if policy_loss is not None:
            self.policy_total_loss += policy_loss
            self.policy_num_batches += 1
        
        if value_loss is not None:
            self.value_total_loss += value_loss
            self.value_num_batches += 1
        
        if total_loss is not None:
            self.total_loss += total_loss
            self.num_batches += 1
        
        if action_type_loss is not None:
            self.action_type_loss += action_type_loss
        
        if param_loss is not None:
            self.param_loss += param_loss
        
        # Return batch-level metrics
        batch_metrics = {
            'policy_acc': policy_acc,
            'policy_correct': policy_correct,
            'policy_total': batch_size,
            'value_acc': value_acc,
            'value_correct': value_correct,
            'value_total': batch_size,
        }
        
        if policy_loss is not None:
            batch_metrics['policy_loss'] = policy_loss
        if value_loss is not None:
            batch_metrics['value_loss'] = value_loss
        if total_loss is not None:
            batch_metrics['total_loss'] = total_loss
        if action_type_loss is not None:
            batch_metrics['action_type_loss'] = action_type_loss
        if param_loss is not None:
            batch_metrics['param_loss'] = param_loss
        
        return batch_metrics
    
    def compute_epoch_metrics(self) -> Dict[str, float]:
        """
        Compute epoch-level metrics from accumulated statistics.
        
        Returns:
            Dictionary of epoch-level metrics including F1 scores
        """
        metrics = {}
        
        # Policy metrics
        if self.policy_num_batches > 0:
            metrics['policy_loss'] = self.policy_total_loss / self.policy_num_batches
        
        # Policy accuracy from confusion matrix
        total_samples = self.policy_cm.sum()
        if total_samples > 0:
            policy_correct = np.trace(self.policy_cm)
            metrics['policy_acc'] = 100.0 * policy_correct / total_samples
        else:
            metrics['policy_acc'] = 0.0
        
        # Policy F1 (macro) from confusion matrix
        tp = np.diag(self.policy_cm)
        fp = self.policy_cm.sum(axis=0) - tp
        fn = self.policy_cm.sum(axis=1) - tp
        
        # Per-class F1
        with np.errstate(divide='ignore', invalid='ignore'):
            f1_per_class = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
            f1_per_class = np.nan_to_num(f1_per_class, nan=0.0)
        
        metrics['policy_f1'] = float(f1_per_class.mean())
        
        # Value metrics
        if self.value_num_batches > 0:
            metrics['value_loss'] = self.value_total_loss / self.value_num_batches
        
        # Value accuracy
        value_total = self.value_tp + self.value_fp + self.value_fn + self.value_tn
        if value_total > 0:
            metrics['value_acc'] = (self.value_tp + self.value_tn) / value_total
        else:
            metrics['value_acc'] = 0.0
        
        # Value F1 (binary)
        value_f1_denom = 2 * self.value_tp + self.value_fp + self.value_fn
        if value_f1_denom > 0:
            metrics['value_f1'] = (2 * self.value_tp) / value_f1_denom
        else:
            metrics['value_f1'] = 0.0
        
        # Combined loss
        if self.num_batches > 0:
            metrics['total_loss'] = self.total_loss / self.num_batches
        
        # Hierarchical-specific
        if self.num_batches > 0 and self.action_type_loss > 0:
            metrics['action_type_loss'] = self.action_type_loss / self.num_batches
        
        if self.num_batches > 0 and self.param_loss > 0:
            metrics['param_loss'] = self.param_loss / self.num_batches
        
        return metrics
    
    def get_policy_confusion_matrix(self) -> np.ndarray:
        """Return the accumulated confusion matrix."""
        return self.policy_cm.copy()
    
    def get_value_counts(self) -> Dict[str, int]:
        """Return value prediction counts (TP/FP/FN/TN)."""
        return {
            'tp': self.value_tp,
            'fp': self.value_fp,
            'fn': self.value_fn,
            'tn': self.value_tn,
        }

