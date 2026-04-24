"""Compatibility re-exports for the native multi-agent Puffer env."""

from __future__ import annotations

from typing import Callable, Literal, TypedDict

from pettingzoo.utils.env import AECEnv

from catanrl.envs.puffer.common import BOARD_HEIGHT, BOARD_WIDTH, COLOR_ORDER
from catanrl.envs.puffer.multi_agent_env import (
    LocalObservation,
    MultiAgentCatanatronEnvConfig,
    ParallelCatanatronPufferEnv,
    _make_parallel_env,
    compute_multiagent_input_dim,
    flatten_marl_observation,
    get_valid_actions,
    make_vectorized_envs,
)


class ActorObservation(TypedDict):
    observation: LocalObservation
    action_mask: object


class SharedCriticObservation(ActorObservation):
    critic: object


def _make_aec_env(
    num_players: int,
    map_type: Literal["BASE", "TOURNAMENT", "MINI"],
    vps_to_win: int,
    discard_limit: int,
    shared_critic: bool,
    reward_function: Literal["shaped", "win"],
) -> Callable[[], AECEnv]:
    from catanrl.envs.zoo.aec_env import AecCatanatronEnv

    def _init():
        return AecCatanatronEnv(
            MultiAgentCatanatronEnvConfig(
                num_players=num_players,
                map_type=map_type,
                vps_to_win=vps_to_win,
                discard_limit=discard_limit,
                shared_critic=shared_critic,
                reward_function=reward_function,
            )
        )

    return _init
