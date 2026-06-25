"""Evaluation utilities for CatanRL."""

from .dagger_eval import (
    FrozenImitationEvalSet,
    eval_policy_imitation,
    expert_supports_frozen_imitation_eval,
    is_better_critic_checkpoint,
    is_better_imitation_checkpoint,
    is_better_policy_checkpoint,
)
from .eval_nn_vs_catanatron import eval
from .training_eval import eval_policy_against_baselines, eval_policy_value_against_baselines

__all__ = [
    "eval",
    "eval_policy_against_baselines",
    "eval_policy_value_against_baselines",
    "eval_policy_imitation",
    "expert_supports_frozen_imitation_eval",
    "FrozenImitationEvalSet",
    "is_better_critic_checkpoint",
    "is_better_imitation_checkpoint",
    "is_better_policy_checkpoint",
]
