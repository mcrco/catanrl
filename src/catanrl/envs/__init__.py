"""
Environment helpers for Catan RL training pipelines.

Provides thin wrappers around the underlying Catanatron gym / PettingZoo
implementations so algorithms do not have to worry about the exact backend.
"""

from .single_env import make_vectorized_envs
from .multi_env import (
    build_marl_env,
    compute_multiagent_input_dim,
    flatten_marl_observation,
    get_valid_actions,
)

__all__ = [
    "make_vectorized_envs",
    "build_marl_env",
    "compute_multiagent_input_dim",
    "flatten_marl_observation",
    "get_valid_actions",
]

