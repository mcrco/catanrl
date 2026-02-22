from __future__ import annotations

import random
import sys
from typing import Dict, Optional, Union

import numpy as np
from gymnasium import spaces
from pettingzoo.utils.env import ParallelEnv

from catanatron.game import Game, TURNS_LIMIT
from catanatron.models.player import Color, Player
from catanatron.models.map import build_map
from catanatron.models.enums import ActionType
from catanatron.gym.board_tensor_features import get_channels, create_board_tensor
from catanatron.gym.envs.catanatron_env import (
    ACTION_SPACE_SIZE,
    ACTIONS_ARRAY,
    from_action_space,
    to_action_space,
    HIGH,
)
from catanrl.features.catanatron_utils import (
    compute_feature_vector_dim,
    full_game_to_features,
    game_to_features,
    get_full_numeric_feature_names,
    get_numeric_feature_names,
)
from catanrl.envs.zoo.rewards import ShapedReward, WinReward
from catanrl.envs.zoo.multi_env import (
    MultiAgentCatanatronEnvConfig,
    COLOR_ORDER,
    BOARD_WIDTH,
    BOARD_HEIGHT,
    LocalObservation,
)

# Find the index of END_TURN
END_TURN_IDX = next(
    i for i, (action_type, _) in enumerate(ACTIONS_ARRAY) if action_type == ActionType.END_TURN
)


class ParallelCatanatronEnv(ParallelEnv):
    metadata = {"name": "parallel_catanatron_env_v0", "render_modes": []}

    def __init__(self, config: Optional[MultiAgentCatanatronEnvConfig] = None):
        self.config = config or MultiAgentCatanatronEnvConfig()
        self.num_players = int(self.config.num_players)
        self.map_type = self.config.map_type
        self.vps_to_win = int(self.config.vps_to_win)
        self.shared_critic = bool(self.config.shared_critic)

        self.possible_agents = [f"player_{i}" for i in range(self.num_players)]

        self.colors_order = list(COLOR_ORDER[: self.num_players])
        self.agent_name_to_color = {
            f"player_{i}": color for i, color in enumerate(self.colors_order)
        }
        self.color_to_agent_name = {
            color: f"player_{i}" for i, color in enumerate(self.colors_order)
        }

        self.reward_function_name = self.config.reward_function
        if self.reward_function_name == "shaped":
            self.reward_function = ShapedReward(self)  # type: ignore
        elif self.reward_function_name == "win":
            self.reward_function = WinReward()
        else:
            raise ValueError(f"Invalid reward function: {self.reward_function_name}")

        self.numeric_dim = len(get_numeric_feature_names(self.num_players, self.map_type))
        full_numeric_dim = len(get_full_numeric_feature_names(self.num_players, self.map_type))
        self.vector_dim = compute_feature_vector_dim(self.num_players, self.map_type)
        self.board_channels = get_channels(self.num_players)
        self.board_tensor_shape = (self.board_channels, BOARD_WIDTH, BOARD_HEIGHT)
        self.critic_vector_dim = self.vector_dim + (full_numeric_dim - self.numeric_dim)

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
        action_mask_space = spaces.Box(low=0, high=1, shape=(ACTION_SPACE_SIZE,), dtype=np.int8)
        critic_space = spaces.Box(
            low=0.0, high=HIGH, shape=(self.critic_vector_dim,), dtype=np.float32
        )

        obs_space = spaces.Dict(
            {
                "observation": actor_space,
                "action_mask": action_mask_space,
                "critic": critic_space,
            }
        )

        self.observation_spaces = {agent: obs_space for agent in self.possible_agents}
        self.action_spaces = {
            agent: spaces.Discrete(ACTION_SPACE_SIZE) for agent in self.possible_agents
        }

        self.game = None
        self.agents = []

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        rng_seed = seed if seed is not None else random.randrange(sys.maxsize)
        catan_map = build_map(self.map_type)
        players = [Player(color) for color in self.colors_order]
        for player in players:
            player.reset_state()

        self.game = Game(
            players=players,
            seed=rng_seed,
            catan_map=catan_map,
            vps_to_win=self.vps_to_win,
        )

        self.agents = self.possible_agents[:]
        self.reward_function.after_reset(self)  # type: ignore

        observations = {agent: self._get_observation(agent) for agent in self.agents}
        infos = {agent: self._get_info(agent) for agent in self.agents}

        return observations, infos

    def step(self, actions: Dict[str, int]):
        if not self.agents:
            return {}, {}, {}, {}, {}

        current_color = self.game.state.current_color()
        current_agent = self.color_to_agent_name[current_color]
        action = actions.get(current_agent)

        if action is not None:
            catan_action = from_action_space(action, self.game.state.playable_actions)
            self.game.execute(catan_action)

        rewards = {agent: 0.0 for agent in self.agents}
        for ag in self.agents:
            rewards[ag] = self.reward_function.reward(self, self.game, self.agent_name_to_color[ag])  # type: ignore

        terminations = {agent: False for agent in self.agents}
        truncations = {agent: False for agent in self.agents}

        winning_color = self.game.winning_color()
        if winning_color is not None:
            for agent in self.agents:
                terminations[agent] = True
            self.agents = []
        elif self.game.state.num_turns >= TURNS_LIMIT:
            for agent in self.agents:
                truncations[agent] = True
            self.agents = []

        for color in self.colors_order:
            self.reward_function.after_step(self, self.game, color)  # type: ignore

        observations = {}
        infos = {}

        for agent in self.agents:
            observations[agent] = self._get_observation(agent)
            infos[agent] = self._get_info(agent)

        return observations, rewards, terminations, truncations, infos

    def _get_observation(self, agent: str) -> Dict[str, np.ndarray]:
        color = self.agent_name_to_color[agent]
        return {
            "observation": self._actor_observation(color),
            "action_mask": self._action_mask(agent),
            # Critic observation should be consistent per-agent (P0=agent).
            # This avoids bootstrapping across timesteps using mixed perspectives.
            "critic": full_game_to_features(self.game, self.num_players, self.map_type, base_color=color),
        }

    def _actor_observation(self, color: Color) -> LocalObservation:
        vector = game_to_features(self.game, color, self.num_players, self.map_type)
        numeric = vector[: self.numeric_dim]
        board = create_board_tensor(self.game, color, channels_first=True).astype(
            np.float32, copy=False
        )
        return {
            "numeric": numeric,
            "board": board,
        }

    def _state(self) -> np.ndarray:
        # Global state defaults to the current player perspective.
        return full_game_to_features(self.game, self.num_players, self.map_type)

    def _action_mask(self, agent: str) -> np.ndarray:
        mask = np.zeros(ACTION_SPACE_SIZE, dtype=np.int8)

        # Only the current player has valid actions, otherwise "pad" with END_TURN
        current_color = self.game.state.current_color()
        if self.agents and self.agent_name_to_color[agent] == current_color:
            valid_actions = list(map(to_action_space, self.game.state.playable_actions))
            mask[valid_actions] = 1
        else:
            mask[END_TURN_IDX] = 1

        return np.asarray(mask, dtype=np.int8)

    def _get_info(self, agent: str) -> Dict:
        valid_actions = np.where(self._action_mask(agent) == 1)[0]
        info = {"valid_actions": valid_actions}
        if self.shared_critic:
            color = self.agent_name_to_color[agent]
            info["critic_observation"] = full_game_to_features(
                self.game, self.num_players, self.map_type, base_color=color
            )
        return info
