"""Backward-compatible import for the legacy GymAgent name."""

from .sarl_agent import SARLAgent as GymAgent

__all__ = ["GymAgent", "SARLAgent"]
