from __future__ import annotations

from typing import Callable, List

import gymnasium as gym

from catanatron.models.player import Color, RandomPlayer
from catanatron.players.mcts import MCTSPlayer
from catanatron.players.minimax import AlphaBetaPlayer, SameTurnAlphaBetaPlayer
from catanatron.players.playouts import GreedyPlayoutsPlayer
from catanatron.players.search import VictoryPointPlayer
from catanatron.players.value import ValueFunctionPlayer
from catanatron.players.weighted_random import WeightedRandomPlayer

# Optional dependency: keep import local to avoid cost if not needed
# from catanatron.players.mcts import MCTSPlayer


def create_opponents(opponent_configs: List[str]) -> List:
    """
    Instantiate opponent AI players from CLI-like config strings.

    Examples:
        ["random", "AB:3", "M:500"]
    """
    colors = [Color.RED, Color.WHITE, Color.ORANGE]
    opponents = []

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
                print(f"Warning: Unknown opponent type '{player_type}', defaulting to RandomPlayer")
                opponents.append(RandomPlayer(color))
        except Exception as exc:  # pragma: no cover - defensive fallback
            print(f"Warning: Failed to create opponent '{config}': {exc}. Using RandomPlayer.")
            opponents.append(RandomPlayer(color))

    return opponents


def _make_env(
    map_type: str,
    opponent_configs: List[str],
    representation: str = "mixed",
) -> Callable[[], gym.Env]:
    """Factory for individual gym environments."""

    def _init():
        return gym.make(
            "catanatron/Catanatron-v0",
            config={
                "map_type": map_type,
                "enemies": create_opponents(opponent_configs),
                "representation": representation,
            },
        )

    return _init


def make_vectorized_envs(
    map_type: str,
    opponent_configs: List[str],
    num_envs: int,
    representation: str = "mixed",
    async_vector: bool = True,
):
    """
    Create a vectorized set of gym environments for SARL training.

    Returns a gym VectorEnv ready for batched interaction.
    """
    env_fns = [
        _make_env(map_type=map_type, opponent_configs=opponent_configs, representation=representation)
        for _ in range(num_envs)
    ]
    if async_vector:
        return gym.vector.AsyncVectorEnv(env_fns)
    return gym.vector.SyncVectorEnv(env_fns)

