"""
Environment helpers for Catan RL training pipelines.

Provides thin wrappers around the underlying Catanatron gym / PettingZoo
implementations so algorithms do not have to worry about the exact backend.
"""

from .gym.single_env import make_vectorized_envs
from .zoo.multi_env import (
    compute_multiagent_input_dim,
    flatten_marl_observation,
    get_valid_actions,
    MultiAgentCatanatronEnvConfig,
)
from .zoo.aec_env import AecCatanatronEnv
from .zoo.parallel_env import ParallelCatanatronEnv

__all__ = [
    "make_vectorized_envs",
    "compute_multiagent_input_dim",
    "flatten_marl_observation",
    "get_valid_actions",
    "MultiAgentCatanatronEnvConfig",
    "AecCatanatronEnv",
    "ParallelCatanatronEnv",
]
