"""Environment helpers for Catan RL training pipelines."""

from .puffer import (
    EpisodeBuffer,
    compute_multiagent_input_dim,
    compute_single_agent_dims,
    create_expert,
    create_opponents,
    decode_puffer_batch,
    extract_expert_actions_from_infos,
    flatten_marl_observation,
    flatten_puffer_observation,
    get_action_mask_from_obs,
    get_valid_actions,
    init_episode_buffers,
    make_puffer_vectorized_envs,
    make_vectorized_envs,
)
from .puffer.multi_agent_env import (
    MultiAgentCatanatronEnvConfig,
    ParallelCatanatronPufferEnv,
)

__all__ = [
    "make_vectorized_envs",
    "make_puffer_vectorized_envs",
    "compute_single_agent_dims",
    "create_expert",
    "create_opponents",
    "compute_multiagent_input_dim",
    "flatten_marl_observation",
    "get_valid_actions",
    "MultiAgentCatanatronEnvConfig",
    "ParallelCatanatronPufferEnv",
    "EpisodeBuffer",
    "decode_puffer_batch",
    "extract_expert_actions_from_infos",
    "flatten_puffer_observation",
    "get_action_mask_from_obs",
    "init_episode_buffers",
]
