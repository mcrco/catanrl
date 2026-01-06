from __future__ import annotations

from typing import Dict, List, Optional, Union
import numpy as np
from gymnasium import spaces
from pettingzoo.utils.env import AECEnv
import random
import sys

from catanatron.models.player import Color, Player
from catanatron.models.map import build_map
from catanatron.game import Game, TURNS_LIMIT
from catanatron.gym.board_tensor_features import get_channels

from catanrl.envs.zoo.rewards import ShapedReward, WinReward
from catanrl.envs.zoo.multi_env import (
    MultiAgentCatanatronEnvConfig,
    ActorObservation,
    SharedCriticObservation,
    LocalObservation,
)
from catanrl.features.catanatron_utils import (
    compute_feature_vector_dim,
    full_game_to_features,
    game_to_features,
    get_numeric_feature_names,
    get_full_numeric_feature_names,
    COLOR_ORDER,
)
from catanatron.gym.envs.catanatron_env import (
    ACTION_SPACE_SIZE,
    from_action_space,
    to_action_space,
    HIGH,
)
from catanrl.envs.zoo.multi_env import BOARD_WIDTH, BOARD_HEIGHT


class AecCatanatronEnv(AECEnv):
    metadata = {"name": "catanatron_aec_v0", "render_modes": []}

    def __init__(self, config: Optional[MultiAgentCatanatronEnvConfig] = None):
        super().__init__()
        self.config = config or MultiAgentCatanatronEnvConfig()
        self.num_players = int(self.config.num_players)
        assert 2 <= self.num_players <= 4, "num_players must be in [2, 4]"
        self.map_type = self.config.map_type
        self.vps_to_win = int(self.config.vps_to_win)
        self.shared_critic = bool(self.config.shared_critic)

        self.possible_agents: List[str] = [f"player_{i}" for i in range(self.num_players)]
        self.colors_order: List[Color] = list(COLOR_ORDER[: self.num_players])
        self.agent_name_to_color: Dict[str, Color] = {
            f"player_{i}": color for i, color in enumerate(self.colors_order)
        }
        self.color_to_agent_name: Dict[Color, str] = {
            color: f"player_{i}" for i, color in enumerate(self.colors_order)
        }

        self.reward_function_name = self.config.reward_function
        assert self.reward_function_name in {"shaped", "win"}
        if self.reward_function_name == "shaped":
            self.reward_function = ShapedReward(self)
        elif self.reward_function_name == "win":
            self.reward_function = WinReward()

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
                    low=0.0,
                    high=1.0,
                    shape=self.board_tensor_shape,
                    dtype=np.float32,
                ),
            }
        )

        action_mask_space = spaces.Box(
            low=0,
            high=1,
            shape=(ACTION_SPACE_SIZE,),
            dtype=np.int8,
        )
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

        self.game: Optional[Game] = None
        self.agent_selection: Optional[str] = None
        self.agents: List[str] = []
        self.rewards: Dict[str, float] = {}
        self._cumulative_rewards: Dict[str, float] = {}
        self.terminations: Dict[str, bool] = {}
        self.truncations: Dict[str, bool] = {}
        self.infos: Dict[str, Dict] = {}

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
        self.rewards = {agent: 0.0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0.0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}

        self.agent_selection = self._current_agent()
        self.reward_function.after_reset(self)
        self._update_infos_for_current()

    def observe(self, agent: str) -> Union[ActorObservation, SharedCriticObservation]:
        assert self.game is not None, "Environment must be reset before calling observe."
        color = self.agent_name_to_color[agent]
        return {
            "observation": self._actor_observation(color),
            "action_mask": self._action_mask(),
            "critic": self.state(),
        }

    def state(self) -> np.ndarray:
        assert self.game is not None
        return full_game_to_features(self.game, self.num_players, self.map_type)

    def step(self, action: Optional[int]):
        assert self.game is not None, "reset() must be called before step()."
        if self.agent_selection is None or len(self.agents) == 0:
            return

        agent = self.agent_selection
        # Zero out award since we want to start fresh with each agent's turn.
        if agent in self._cumulative_rewards:
            self._cumulative_rewards[agent] = 0.0

        if self.terminations[agent] or self.truncations[agent]:
            self._was_dead_step(action)
            return

        self._clear_rewards()

        catan_action = from_action_space(action, self.game.state.playable_actions)
        self.game.execute(catan_action)

        for ag in self.agents:
            self.rewards[ag] = self.reward_function.reward(
                self, self.game, self.agent_name_to_color[ag]
            )

        winning_color = self.game.winning_color()
        if winning_color is not None:
            for color in self.colors_order:
                ag = self.color_to_agent_name[color]
                self.terminations[ag] = True
        elif self.game.state.num_turns >= TURNS_LIMIT:
            for ag in self.agents:
                self.truncations[ag] = True
        else:
            self.agent_selection = self._current_agent()
            self._update_infos_for_current()

        for color in self.colors_order:
            self.reward_function.after_step(self, self.game, color)
        self._accumulate_rewards()

    def render(self):
        return None

    def close(self):
        return None

    def _actor_observation(self, color: Color) -> LocalObservation:
        assert self.game is not None
        vector = game_to_features(self.game, color, self.num_players, self.map_type)
        numeric = vector[: self.numeric_dim]
        board_flat = vector[self.numeric_dim :]
        board = board_flat.reshape(self.board_tensor_shape)
        return {
            "numeric": numeric,
            "board": board,
        }

    def _current_agent(self) -> str:
        assert self.game is not None
        current_color = self.game.state.current_color()
        return self.color_to_agent_name[current_color]

    def _next_agent_if_any(self) -> Optional[str]:
        if len(self.agents) == 0:
            return None
        return self._current_agent()

    def _clear_rewards(self):
        for ag in self.rewards:
            self.rewards[ag] = 0.0

    def _accumulate_rewards(self):
        for ag in self.rewards:
            self._cumulative_rewards[ag] += self.rewards[ag]

    def _update_infos_for_current(self):
        assert self.game is not None
        for ag in self.infos:
            self.infos[ag] = {}
        valid_actions = list(map(to_action_space, self.game.state.playable_actions))
        current_agent = self._current_agent()
        info_payload: Dict[str, Union[List[int], np.ndarray]] = {"valid_actions": valid_actions}
        if self.shared_critic:
            info_payload["critic_observation"] = self.state()
        self.infos[current_agent] = info_payload

    def _action_mask(self) -> np.ndarray:
        assert self.game is not None
        mask = np.zeros(ACTION_SPACE_SIZE, dtype=np.int8)
        valid_actions = list(map(to_action_space, self.game.state.playable_actions))
        mask[valid_actions] = 1
        return mask
