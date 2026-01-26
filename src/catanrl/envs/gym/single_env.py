from __future__ import annotations

from typing import Any, Callable, Dict, List, Literal, Union

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pufferlib
import pufferlib.vector as puffer_vector

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
    to_action_space,
    from_action_space,
    HIGH,
)

from catanrl.envs.gym.rewards import ShapedReward, WinReward, RewardFunction
from catanrl.features.catanatron_utils import (
    full_game_to_features,
    game_to_features,
    get_full_numeric_feature_names,
    get_numeric_feature_names,
)

BOARD_WIDTH = 21
BOARD_HEIGHT = 11


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
# With optional shared_critic mode for DAgger/PPO training with centralized critic
# and optional expert_player for imitation learning (returns expert actions in info)
class SingleAgentCatanatronEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, config=None):
        self.config = config or dict()
        self.reward_function: RewardFunction = self.config.get("reward_function")
        self.map_type: Literal["BASE", "MINI", "TOURNAMENT"] = self.config.get("map_type", "BASE")
        self.vps_to_win = self.config.get("vps_to_win", 10)
        self.enemies = self.config.get("enemies", [RandomPlayer(Color.RED)])
        self.representation = self.config.get("representation", "mixed")
        self.shared_critic = self.config.get("shared_critic", False)
        # Optional expert player for imitation learning - if set, expert_action is returned in info
        self.expert_player = self.config.get("expert_player", None)

        assert all(p.color != Color.BLUE for p in self.enemies)
        assert self.representation in ["mixed", "vector"]
        self.p0 = Player(Color.BLUE)
        self.players = [self.p0] + self.enemies
        self.num_players = len(self.players)
        self.features = get_feature_ordering(self.num_players, self.map_type)

        # Compute dimensions for shared_critic mode
        self.numeric_dim = len(get_numeric_feature_names(self.num_players, self.map_type))
        self.full_numeric_dim = len(get_full_numeric_feature_names(self.num_players, self.map_type))
        self.board_channels = get_channels(self.num_players)
        self.board_tensor_shape = (self.board_channels, BOARD_WIDTH, BOARD_HEIGHT)
        self.board_flat_dim = self.board_channels * BOARD_WIDTH * BOARD_HEIGHT
        self.critic_vector_dim = self.full_numeric_dim + self.board_flat_dim

        # Action space
        self.action_space = spaces.Discrete(ACTION_SPACE_SIZE)

        # Build observation space
        if self.shared_critic:
            # Structured observation for DAgger/PPO with centralized critic
            actor_space = spaces.Dict(
                {
                    "numeric": spaces.Box(
                        low=0.0, high=HIGH, shape=(self.numeric_dim,), dtype=np.float32
                    ),
                    "board": spaces.Box(
                        low=0.0, high=1.0, shape=self.board_tensor_shape, dtype=np.float32
                    ),
                }
            )
            action_mask_space = spaces.Box(
                low=0, high=1, shape=(ACTION_SPACE_SIZE,), dtype=np.int8
            )
            critic_space = spaces.Box(
                low=0.0, high=HIGH, shape=(self.critic_vector_dim,), dtype=np.float32
            )
            self.observation_space = spaces.Dict(
                {
                    "observation": actor_space,
                    "action_mask": action_mask_space,
                    "critic": critic_space,
                }
            )
        elif self.representation == "mixed":
            board_tensor_space = spaces.Box(
                low=0, high=1, shape=self.board_tensor_shape, dtype=np.float64
            )
            self.numeric_features = [f for f in self.features if not is_graph_feature(f)]
            numeric_space = spaces.Box(
                low=0, high=HIGH, shape=(len(self.numeric_features),), dtype=np.float64
            )
            self.observation_space = spaces.Dict(
                {
                    "board": board_tensor_space,
                    "numeric": numeric_space,
                }
            )
        else:
            self.observation_space = spaces.Box(
                low=0, high=HIGH, shape=(len(self.features),), dtype=np.float64
            )

        self.game: Game = None  # type: ignore
        self.reset()

    def get_valid_actions(self) -> List[int]:
        """Returns list of valid action indices."""
        return list(map(to_action_space, self.game.state.playable_actions))

    def step(self, action):
        catan_action = from_action_space(action, self.game.state.playable_actions)
        self.game.execute(catan_action)
        self._advance_until_p0_decision()

        observation = self._get_observation()
        info = self._build_info()

        winning_color = self.game.winning_color()
        terminated = winning_color is not None
        truncated = self.game.state.num_turns >= TURNS_LIMIT
        reward = self.reward_function.reward(self, self.game, self.p0.color)  # type: ignore
        self.reward_function.after_step(self, self.game, self.p0.color)  # type: ignore

        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
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
        # Reset expert player state if present
        if self.expert_player is not None:
            self.expert_player.reset_state()

        self._advance_until_p0_decision()
        observation = self._get_observation()
        info = self._build_info()
        self.reward_function.after_reset(self)  # type: ignore
        return observation, info

    def _build_info(self) -> Dict[str, Any]:
        """Build info dict, optionally including expert action."""
        info: Dict[str, Any] = {"valid_actions": self.get_valid_actions()}
        if self.expert_player is not None:
            info["expert_action"] = self._get_expert_action()
        return info

    def _get_expert_action(self) -> int:
        """Query the expert player for its action on the current game state."""
        if self.expert_player is None:
            raise ValueError("No expert player configured")
        # The expert needs to decide for the current player (p0)
        expert_catan_action = self.expert_player.decide(self.game, self.game.state.playable_actions)
        return to_action_space(expert_catan_action)

    def _get_observation(self) -> Union[np.ndarray, Dict[str, Any]]:
        if self.shared_critic:
            return {
                "observation": self._actor_observation(),
                "action_mask": self._action_mask(),
                "critic": self._critic_observation(),
            }
        elif self.representation == "mixed":
            sample = create_sample(self.game, self.p0.color)
            board_tensor = create_board_tensor(self.game, self.p0.color, channels_first=True)
            numeric = np.array([float(sample[i]) for i in self.numeric_features])
            return {"board": board_tensor, "numeric": numeric}
        else:
            sample = create_sample(self.game, self.p0.color)
            return np.array([float(sample[i]) for i in self.features])

    def _actor_observation(self) -> Dict[str, np.ndarray]:
        """Get actor observation (local info only). Used in shared_critic mode."""
        vector = game_to_features(self.game, self.p0.color, self.num_players, self.map_type)
        numeric = vector[: self.numeric_dim]
        board_flat = vector[self.numeric_dim :]
        board = board_flat.reshape(self.board_tensor_shape)
        return {
            "numeric": numeric.astype(np.float32),
            "board": board.astype(np.float32),
        }

    def _critic_observation(self) -> np.ndarray:
        """Get critic observation (full game info). Used in shared_critic mode."""
        return full_game_to_features(self.game, self.num_players, self.map_type)

    def _action_mask(self) -> np.ndarray:
        """Get binary mask of valid actions. Used in shared_critic mode."""
        mask = np.zeros(ACTION_SPACE_SIZE, dtype=np.int8)
        valid_actions = self.get_valid_actions()
        mask[valid_actions] = 1
        return mask

    def _advance_until_p0_decision(self):
        while (
            self.game.winning_color() is None and self.game.state.current_color() != self.p0.color
        ):
            self.game.play_tick()


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


def create_expert(expert_config: str) -> Player:
    """
    Create an expert player from a config string.

    Uses the same format as create_opponents but creates a single player
    with BLUE color (the learning agent's color).
    """
    parts = expert_config.split(":")
    player_type = parts[0].lower()
    params = parts[1:] if len(parts) > 1 else []

    # Expert uses BLUE color to match the learning agent's perspective
    color = Color.BLUE

    if player_type in {"random", "r"}:
        return RandomPlayer(color)
    elif player_type in {"weighted", "w"}:
        return WeightedRandomPlayer(color)
    elif player_type in {"value", "valuefunction", "f"}:
        return ValueFunctionPlayer(color)
    elif player_type in {"victorypoint", "vp"}:
        return VictoryPointPlayer(color)
    elif player_type in {"alphabeta", "ab"}:
        depth = int(params[0]) if params else 2
        pruning = params[1].lower() != "false" if len(params) > 1 else True
        return AlphaBetaPlayer(color, depth=depth, prunning=pruning)
    elif player_type in {"sameturnalphabeta", "sab"}:
        return SameTurnAlphaBetaPlayer(color)
    elif player_type in {"mcts", "m"}:
        num_simulations = int(params[0]) if params else 100
        return MCTSPlayer(color, num_simulations)
    elif player_type in {"playouts", "greedyplayouts", "g"}:
        num_playouts = int(params[0]) if params else 25
        return GreedyPlayoutsPlayer(color, num_playouts)
    else:
        raise ValueError(f"Unknown expert type: {expert_config}")


def _make_puffer_env(
    reward_function: str,
    map_type: Literal["BASE", "MINI", "TOURNAMENT"],
    opponent_configs: List[str],
    expert_config: str | None = None,
) -> Callable[[], pufferlib.emulation.GymnasiumPufferEnv]:
    """Factory for individual Puffer-wrapped environments with shared_critic enabled.

    Args:
        reward_function: "shaped" or "win"
        map_type: Map type for the game
        opponent_configs: List of opponent config strings
        expert_config: Optional expert config string. If provided, expert actions
            are included in info dict (useful for imitation learning).
    """

    def _init():
        if reward_function == "shaped":
            reward_fn = ShapedReward()
        elif reward_function == "win":
            reward_fn = WinReward()
        else:
            raise ValueError(f"Unknown reward function: {reward_function}")

        expert_player = create_expert(expert_config) if expert_config else None

        return SingleAgentCatanatronEnv(
            config={
                "map_type": map_type,
                "enemies": create_opponents(opponent_configs),
                "reward_function": reward_fn,
                "shared_critic": True,  # Enable structured obs for DAgger/PPO
                "expert_player": expert_player,
            }
        )

    return lambda **kwargs: pufferlib.emulation.GymnasiumPufferEnv(env=_init(), **kwargs)


def make_puffer_vectorized_envs(
    reward_function: str,
    map_type: Literal["BASE", "MINI", "TOURNAMENT"],
    opponent_configs: List[str],
    num_envs: int,
    expert_config: str | None = None,
):
    """
    Create a vectorized set of Puffer-wrapped environments for DAgger training.

    Args:
        reward_function: "shaped" or "win"
        map_type: Map type for the game
        opponent_configs: List of opponent config strings
        num_envs: Number of parallel environments
        expert_config: Optional expert config string. If provided, expert actions
            are included in info dict (useful for imitation learning).

    Returns a Puffer VectorEnv ready for batched interaction with observations
    containing actor obs, action masks, and critic obs.
    """
    return puffer_vector.make(
        _make_puffer_env(reward_function, map_type, opponent_configs, expert_config),
        num_envs=num_envs,
        backend=puffer_vector.Multiprocessing,
    )


def compute_single_agent_dims(
    num_players: int,
    map_type: Literal["BASE", "MINI", "TOURNAMENT"],
) -> Dict[str, int]:
    """
    Compute input dimensions for single-agent environment.

    Returns:
        Dict with keys: actor_dim, critic_dim, numeric_dim, board_dim
    """
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
