"""
Placeholder for centralized critic PPO components.

Future work: implement critic networks that consume joint state across agents
and expose trainer utilities similar to trainer_marl.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class CentralCriticConfig:
    """Configuration stub for centralized critic trainers."""

    value_coef: float = 0.5
    entropy_coef: float = 0.01
    clip_epsilon: float = 0.2
    gamma: float = 0.99
    gae_lambda: float = 0.95
    max_grad_norm: float = 0.5
    hidden_dim: int = 512


class CentralCriticTrainer:
    """
    Not yet implemented.
    Provides a documented entry point for future centralized critic training.
    """

    def __init__(self, config: Optional[CentralCriticConfig] = None):
        self.config = config or CentralCriticConfig()

    def train(self, *args: Any, **kwargs: Any) -> Dict[str, float]:  # pragma: no cover - stub
        raise NotImplementedError(
            "CentralCriticTrainer is a placeholder. Implement centralized critic PPO here."
        )

