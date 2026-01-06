from __future__ import annotations

from typing import Callable, List, Union

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from catanatron.models.player import Color, RandomPlayer, Player
from catanatron.game import TURNS_LIMIT, Game
from catanatron.models.map import build_map
from catanatron.features import create_sample
from catanatron.gym.board_tensor_features import create_board_tensor
from catanatron.players.mcts import MCTSPlayer
from catanatron.players.minimax import AlphaBetaPlayer, SameTurnAlphaBetaPlayer
from catanatron.players.playouts import GreedyPlayoutsPlayer
from catanatron.players.search import VictoryPointPlayer
from catanatron.players.value import ValueFunctionPlayer
from catanatron.players.weighted_random import WeightedRandomPlayer
from catanatron.gym.envs.catanatron_env import (
    ACTION_SPACE_SIZE,
    get_channels,
    get_feature_ordering,
    is_graph_feature,
    MixedObservation,
    to_action_space,
    from_action_space,
    HIGH,
)

from catanrl.envs.gym.rewards import ShapedReward, WinReward, RewardFunction


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


# Pretty much exactly CatanatronEnv, but uses reward defined in catanrl.envs.rewards
class SingleAgentCatanatronEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, config=None):
        self.config = config or dict()
        self.reward_function: RewardFunction = self.config.get("reward_function")
        self.map_type = self.config.get("map_type", "BASE")
        self.vps_to_win = self.config.get("vps_to_win", 10)
        self.enemies = self.config.get("enemies", [RandomPlayer(Color.RED)])
        self.representation = self.config.get("representation", "vector")

        assert all(p.color != Color.BLUE for p in self.enemies)
        assert self.representation in ["mixed", "vector"]
        self.p0 = Player(Color.BLUE)
        self.players = [self.p0] + self.enemies  # type: ignore
        self.representation = "mixed" if self.representation == "mixed" else "vector"
        self.features = get_feature_ordering(len(self.players), self.map_type)

        # TODO: Make self.action_space tighter if possible (per map_type)
        self.action_space = spaces.Discrete(ACTION_SPACE_SIZE)

        if self.representation == "mixed":
            channels = get_channels(len(self.players))
            board_tensor_space = spaces.Box(
                low=0, high=1, shape=(channels, 21, 11), dtype=np.float64
            )
            self.numeric_features = [f for f in self.features if not is_graph_feature(f)]
            # TODO: This could be tigher (e.g. _ROADS_AVAILABLE <= 15)
            numeric_space = spaces.Box(
                low=0, high=HIGH, shape=(len(self.numeric_features),), dtype=np.float64
            )
            mixed = spaces.Dict(
                {
                    "board": board_tensor_space,
                    "numeric": numeric_space,
                }
            )
            self.observation_space = mixed
        else:
            # TODO: This could be tigher (e.g. _ROADS_AVAILABLE <= 15)
            self.observation_space = spaces.Box(
                low=0, high=HIGH, shape=(len(self.features),), dtype=np.float64
            )

        self.reset()

    def get_valid_actions(self):
        """
        Returns:
            List[int]: valid actions
        """
        return list(map(to_action_space, self.game.state.playable_actions))

    def step(self, action):
        catan_action = from_action_space(action, self.game.state.playable_actions)
        self.game.execute(catan_action)
        self._advance_until_p0_decision()

        observation = self._get_observation()
        info = dict(valid_actions=self.get_valid_actions())

        winning_color = self.game.winning_color()
        terminated = winning_color is not None
        truncated = self.game.state.num_turns >= TURNS_LIMIT
        reward = self.reward_function.reward(self, self.game, self.p0.color)
        self.reward_function.after_step(self, self.game, self.p0.color)

        return observation, reward, terminated, truncated, info

    def reset(
        self,
        seed=None,
        options=None,
    ):
        super().reset(seed=seed)

        catan_map = build_map(self.map_type)
        for player in self.players:
            player.reset_state()
        self.game = Game(
            players=self.players,
            seed=seed,
            catan_map=catan_map,
            vps_to_win=self.vps_to_win,
        )

        self._advance_until_p0_decision()
        observation = self._get_observation()
        info = dict(valid_actions=self.get_valid_actions())
        self.reward_function.after_reset(self)
        return observation, info

    def _get_observation(self) -> Union[np.ndarray, MixedObservation]:
        sample = create_sample(self.game, self.p0.color)
        if self.representation == "mixed":
            board_tensor = create_board_tensor(self.game, self.p0.color, channels_first=True)
            numeric = np.array([float(sample[i]) for i in self.numeric_features])
            return {"board": board_tensor, "numeric": numeric}

        return np.array([float(sample[i]) for i in self.features])

    def _advance_until_p0_decision(self):
        while (
            self.game.winning_color() is None and self.game.state.current_color() != self.p0.color
        ):
            self.game.play_tick()  # will play bot


def _make_env(
    reward_function: str,
    map_type: str,
    opponent_configs: List[str],
    representation: str = "mixed",
) -> Callable[[], gym.Env]:
    """Factory for individual gym environments."""
    def _init():
        if reward_function == "shaped":
            reward_fn = ShapedReward()
        elif reward_function == "win":
            reward_fn = WinReward()
        else:
             raise ValueError(f"Unknown reward function: {reward_function}")

        return SingleAgentCatanatronEnv(
            config={
                "map_type": map_type,
                "enemies": create_opponents(opponent_configs),
                "representation": representation,
                "reward_function": reward_fn,
            },
        )

    return _init


def make_vectorized_envs(
    reward_function: str,
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
        _make_env(
            reward_function=reward_function,
            map_type=map_type,
            opponent_configs=opponent_configs,
            representation=representation,
        )
        for _ in range(num_envs)
    ]
    if async_vector:
        return gym.vector.AsyncVectorEnv(env_fns)
    return gym.vector.SyncVectorEnv(env_fns)
