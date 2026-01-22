"""
PettingZoo AEC environment used by the MARL training helpers.

Compared to the upstream Catanatron PettingZoo env, observations here are built
from the feature conversion utilities under `catanrl.features.catanatron_utils`.
This ensures:
    - Per-agent observations match `game_to_features`.
    - When `shared_critic=True`, an additional perfect-information vector built
      via `full_game_to_features` is returned alongside the actor observation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union, TypedDict, Callable, Literal
import pufferlib
import pufferlib.vector as puffer_vector
import numpy as np
from pettingzoo.utils.env import AECEnv

from catanatron.models.player import Color
from catanatron.gym.board_tensor_features import get_channels
from catanatron.gym.envs.catanatron_env import (
    MixedObservation,
)
from catanrl.features.catanatron_utils import (
    compute_feature_vector_dim,
    get_numeric_feature_names,
)

BOARD_WIDTH = 21
BOARD_HEIGHT = 11
COLOR_ORDER: Tuple[Color, ...] = (
    Color.BLUE,
    Color.RED,
    Color.WHITE,
    Color.ORANGE,
)


LocalObservation = Union[np.ndarray, MixedObservation]


class ActorObservation(TypedDict):
    observation: LocalObservation
    action_mask: np.ndarray


class SharedCriticObservation(ActorObservation):
    critic: np.ndarray


@dataclass
class MultiAgentCatanatronEnvConfig:
    num_players: int = 2
    map_type: Literal["BASE", "TOURNAMENT", "MINI"] = "BASE"
    vps_to_win: int = 10
    shared_critic: bool = False
    reward_function: Literal["shaped", "win"] = "shaped"


def compute_multiagent_input_dim(
    num_players: int,
    map_type: Literal["BASE", "TOURNAMENT", "MINI"],
) -> Tuple[int, Optional[Tuple[int, int, int]], Optional[int]]:
    numeric_dim = len(get_numeric_feature_names(num_players, map_type))
    vector_dim = compute_feature_vector_dim(num_players, map_type)
    board_shape = (get_channels(num_players), BOARD_WIDTH, BOARD_HEIGHT)
    return vector_dim, board_shape, numeric_dim


def flatten_marl_observation(
    observation: Union[ActorObservation, SharedCriticObservation],
) -> np.ndarray:
    def _flatten_single(obs: LocalObservation) -> np.ndarray:
        numeric = np.asarray(obs["numeric"], dtype=np.float32).reshape(-1)
        board = np.asarray(obs["board"], dtype=np.float32).reshape(-1)
        return np.concatenate([numeric, board], axis=0)

    return _flatten_single(observation["observation"])


def get_valid_actions(
    info: Optional[Dict],
    observation: Optional[Union[ActorObservation, SharedCriticObservation]] = None,
) -> np.ndarray:
    if observation is not None and "action_mask" in observation:
        mask = np.asarray(observation["action_mask"], dtype=np.int8).reshape(-1)
        return np.where(mask > 0)[0].astype(np.int64)
    if not info:
        return np.array([], dtype=np.int64)
    valid = info.get("valid_actions", [])
    return np.asarray(valid, dtype=np.int64)


def _make_aec_env(
    num_players: int,
    map_type: Literal["BASE", "TOURNAMENT", "MINI"],
    shared_critic: bool,
    reward_function: Literal["shaped", "win"],
) -> Callable[[], AECEnv]:
    """Factory for individual multi-agent Catanatron environments."""
    from catanrl.envs.zoo.aec_env import AecCatanatronEnv

    if reward_function not in ("shaped", "win"):
        raise ValueError(f"Unknown reward function: {reward_function}")

    def _init():
        return AecCatanatronEnv(
            MultiAgentCatanatronEnvConfig(
                num_players=num_players,
                map_type=map_type,
                shared_critic=shared_critic,
                reward_function=reward_function,
            )
        )

    return _init


def _make_parallel_env(
    num_players: int,
    map_type: Literal["BASE", "TOURNAMENT", "MINI"],
    shared_critic: bool,
    reward_function: Literal["shaped", "win"],
) -> Callable[[], pufferlib.emulation.PettingzooPufferEnv]:
    from catanrl.envs.zoo.parallel_env import ParallelCatanatronEnv

    def _init():
        return ParallelCatanatronEnv(
            MultiAgentCatanatronEnvConfig(
                num_players=num_players,
                map_type=map_type,
                shared_critic=shared_critic,
                reward_function=reward_function,
            )
        )

    return lambda **kwargs: pufferlib.emulation.PettingZooPufferEnv(env=_init(), **kwargs)


def make_vectorized_envs(
    num_players: int,
    map_type: Literal["BASE", "TOURNAMENT", "MINI"],
    shared_critic: bool,
    reward_function: Literal["shaped", "win"],
    num_envs: int,
):
    """
    Create a vectorized set of gym environments for SARL training.

    Returns a gym VectorEnv ready for batched interaction.
    """
    return puffer_vector.make(
        _make_parallel_env(num_players, map_type, shared_critic, reward_function),
        num_envs=num_envs,
        backend=puffer_vector.Multiprocessing,
    )
