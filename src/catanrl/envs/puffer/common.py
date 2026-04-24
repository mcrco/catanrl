from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
from gymnasium import spaces

from catanatron.gym.board_tensor_features import get_channels
from catanatron.gym.envs.catanatron_env import HIGH
from catanatron.models.player import Color, Player, RandomPlayer
from catanatron.players.mcts import MCTSPlayer
from catanatron.players.minimax import AlphaBetaPlayer, SameTurnAlphaBetaPlayer
from catanatron.players.playouts import GreedyPlayoutsPlayer
from catanatron.players.search import VictoryPointPlayer
from catanatron.players.value import ValueFunctionPlayer
from catanatron.players.weighted_random import WeightedRandomPlayer

from catanrl.features.catanatron_utils import (
    compute_feature_vector_dim,
    get_full_numeric_feature_names,
    get_numeric_feature_names,
)
from catanrl.utils.catanatron_action_space import PLAYER_COLOR_ORDER

MapType = Literal["BASE", "MINI", "TOURNAMENT"]

BOARD_WIDTH = 21
BOARD_HEIGHT = 11
COLOR_ORDER: Tuple[Color, ...] = PLAYER_COLOR_ORDER


def normalize_reset_seed(seed: Any) -> Optional[int]:
    if seed is None:
        return None
    if isinstance(seed, (list, tuple, np.ndarray)):
        if len(seed) == 0:
            return None
        return int(seed[0])
    return int(seed)


class EpisodeSeedTracker:
    def __init__(self) -> None:
        self._seed_sequence_root: int | None = None
        self._seed_sequence_index = 0

    def next_episode_seed(self, seed: int | None, derive_seed_fn) -> int | None:
        if seed is not None:
            self._seed_sequence_root = int(seed)
            self._seed_sequence_index = 0
        elif self._seed_sequence_root is None:
            return None

        assert self._seed_sequence_root is not None
        episode_seed = derive_seed_fn(
            self._seed_sequence_root,
            "env_episode",
            self._seed_sequence_index,
        )
        self._seed_sequence_index += 1
        return episode_seed


def create_opponents(opponent_configs: List[str]) -> List[Player]:
    colors = [Color.RED, Color.WHITE, Color.ORANGE]
    opponents: List[Player] = []

    for i, config in enumerate(opponent_configs):
        color = colors[i % len(colors)]
        parts = config.split(":")
        player_type = parts[0].lower()
        params = parts[1:] if len(parts) > 1 else []

        try:
            if player_type in {"random", "r"}:
                opponents.append(RandomPlayer(color))
            elif player_type in {"weighted", "w"}:
                opponents.append(WeightedRandomPlayer(color))
            elif player_type in {"value", "valuefunction", "f"}:
                opponents.append(ValueFunctionPlayer(color))
            elif player_type in {"victorypoint", "vp"}:
                opponents.append(VictoryPointPlayer(color))
            elif player_type in {"alphabeta", "ab"}:
                depth = int(params[0]) if params else 2
                pruning = params[1].lower() != "false" if len(params) > 1 else True
                opponents.append(AlphaBetaPlayer(color, depth=depth, prunning=pruning))
            elif player_type in {"sameturnalphabeta", "sab"}:
                opponents.append(SameTurnAlphaBetaPlayer(color))
            elif player_type in {"mcts", "m"}:
                num_simulations = int(params[0]) if params else 100
                opponents.append(MCTSPlayer(color, num_simulations))
            elif player_type in {"playouts", "greedyplayouts", "g"}:
                num_playouts = int(params[0]) if params else 25
                opponents.append(GreedyPlayoutsPlayer(color, num_playouts))
            else:
                print(
                    f"Warning: Unknown opponent type '{player_type}', defaulting to RandomPlayer"
                )
                opponents.append(RandomPlayer(color))
        except Exception as exc:  # pragma: no cover - defensive fallback
            print(f"Warning: Failed to create opponent '{config}': {exc}. Using RandomPlayer.")
            opponents.append(RandomPlayer(color))

    return opponents


def create_expert(expert_config: str) -> Player:
    parts = expert_config.split(":")
    player_type = parts[0].lower()
    params = parts[1:] if len(parts) > 1 else []
    color = Color.BLUE

    if player_type in {"random", "r"}:
        return RandomPlayer(color)
    if player_type in {"weighted", "w"}:
        return WeightedRandomPlayer(color)
    if player_type in {"value", "valuefunction", "f"}:
        return ValueFunctionPlayer(color)
    if player_type in {"victorypoint", "vp"}:
        return VictoryPointPlayer(color)
    if player_type in {"alphabeta", "ab"}:
        depth = int(params[0]) if params else 2
        pruning = params[1].lower() != "false" if len(params) > 1 else True
        return AlphaBetaPlayer(color, depth=depth, prunning=pruning)
    if player_type in {"sameturnalphabeta", "sab"}:
        return SameTurnAlphaBetaPlayer(color)
    if player_type in {"mcts", "m"}:
        num_simulations = int(params[0]) if params else 100
        return MCTSPlayer(color, num_simulations)
    if player_type in {"playouts", "greedyplayouts", "g"}:
        num_playouts = int(params[0]) if params else 25
        return GreedyPlayoutsPlayer(color, num_playouts)
    raise ValueError(f"Unknown expert type: {expert_config}")


def build_shared_critic_observation_space(
    numeric_dim: int,
    board_tensor_shape: tuple[int, int, int],
    critic_dim: int,
    action_space_size: int,
) -> spaces.Dict:
    actor_space = spaces.Dict(
        {
            "numeric": spaces.Box(
                low=0.0,
                high=HIGH,
                shape=(numeric_dim,),
                dtype=np.float32,
            ),
            "board": spaces.Box(
                low=0.0,
                high=1.0,
                shape=board_tensor_shape,
                dtype=np.float32,
            ),
        }
    )
    action_mask_space = spaces.Box(
        low=0,
        high=1,
        shape=(action_space_size,),
        dtype=np.int8,
    )
    critic_space = spaces.Box(
        low=0.0,
        high=HIGH,
        shape=(critic_dim,),
        dtype=np.float32,
    )
    return spaces.Dict(
        {
            "observation": actor_space,
            "action_mask": action_mask_space,
            "critic": critic_space,
        }
    )


def compute_single_agent_dims(num_players: int, map_type: MapType) -> Dict[str, int]:
    numeric_dim = len(get_numeric_feature_names(num_players, map_type))
    full_numeric_dim = len(get_full_numeric_feature_names(num_players, map_type))
    board_channels = get_channels(num_players)
    board_flat_dim = board_channels * BOARD_WIDTH * BOARD_HEIGHT
    actor_dim = numeric_dim + board_flat_dim
    critic_dim = full_numeric_dim + board_flat_dim
    return {
        "actor_dim": actor_dim,
        "critic_dim": critic_dim,
        "numeric_dim": numeric_dim,
        "full_numeric_dim": full_numeric_dim,
        "board_dim": board_flat_dim,
        "board_channels": board_channels,
    }


def compute_multiagent_input_dim(
    num_players: int,
    map_type: MapType,
) -> Tuple[int, tuple[int, int, int], int]:
    numeric_dim = len(get_numeric_feature_names(num_players, map_type))
    vector_dim = compute_feature_vector_dim(num_players, map_type)
    board_shape = (get_channels(num_players), BOARD_WIDTH, BOARD_HEIGHT)
    return vector_dim, board_shape, numeric_dim
