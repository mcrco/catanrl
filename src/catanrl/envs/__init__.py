"""
Environment helpers for Catan RL training pipelines.

Provides thin wrappers around the underlying Catanatron gym / PettingZoo
implementations so algorithms do not have to worry about the exact backend.
"""

from .gym.single_env import (
    make_vectorized_envs,
    make_puffer_vectorized_envs,
    compute_single_agent_dims,
    SingleAgentCatanatronEnv,
)
from .zoo.multi_env import (
    compute_multiagent_input_dim,
    flatten_marl_observation,
    get_valid_actions,
    MultiAgentCatanatronEnvConfig,
)
from .zoo.aec_env import AecCatanatronEnv
from .zoo.parallel_env import ParallelCatanatronEnv
from .puffer.rollout_utils import (
    EpisodeBuffer,
    decode_puffer_batch,
    extract_expert_actions_from_infos,
    flatten_puffer_observation,
    get_action_mask_from_obs,
    init_episode_buffers,
)

__all__ = [
    "make_vectorized_envs",
    "make_puffer_vectorized_envs",
    "compute_single_agent_dims",
    "SingleAgentCatanatronEnv",
    "compute_multiagent_input_dim",
    "flatten_marl_observation",
    "get_valid_actions",
    "MultiAgentCatanatronEnvConfig",
    "AecCatanatronEnv",
    "ParallelCatanatronEnv",
    "EpisodeBuffer",
    "decode_puffer_batch",
    "extract_expert_actions_from_infos",
    "flatten_puffer_observation",
    "get_action_mask_from_obs",
    "init_episode_buffers",
]
