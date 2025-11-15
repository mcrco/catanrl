"""
Placeholder AlphaZero trainer definitions.

The goal is to make it easy to plug in real AlphaZero-style training later by
defining a consistent interface today.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class AlphaZeroConfig:
    """Basic configuration stub."""

    simulations: int = 800
    temperature: float = 1.0
    dirichlet_alpha: float = 0.3
    dirichlet_frac: float = 0.25
    gamma: float = 0.99


class AlphaZeroTrainer:
    """Not implemented yet – placeholder for future work."""

    def __init__(self, config: Optional[AlphaZeroConfig] = None):
        self.config = config or AlphaZeroConfig()

    def train(self, *args: Any, **kwargs: Any) -> Dict[str, float]:  # pragma: no cover - stub
        raise NotImplementedError(
            "AlphaZeroTrainer is a placeholder. Implement AlphaZero-style training here."
        )

