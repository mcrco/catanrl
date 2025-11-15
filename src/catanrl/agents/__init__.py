"""
Agent implementations used across different training regimes.
"""

from .sarl_agent import SARLAgent
from .marl_agent import MultiAgentRLAgent

__all__ = ["SARLAgent", "MultiAgentRLAgent"]

