"""Backward-compatible imports for the native Puffer env modules."""

from .multi_agent_env import MultiAgentCatanatronEnvConfig, ParallelCatanatronPufferEnv
from .single_agent_env import SingleAgentCatanatronPufferEnv

__all__ = [
    "MultiAgentCatanatronEnvConfig",
    "ParallelCatanatronPufferEnv",
    "SingleAgentCatanatronPufferEnv",
]
