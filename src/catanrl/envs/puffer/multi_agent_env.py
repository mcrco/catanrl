from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Literal, Optional

import numpy as np
import pufferlib.vector as puffer_vector
from gymnasium import spaces
from pufferlib.emulation import emulate, emulate_action_space, emulate_observation_space, nativize
from pufferlib.environment import PufferEnv

from catanatron.game import Game, TURNS_LIMIT
from catanatron.gym.board_tensor_features import create_board_tensor
from catanatron.models.player import Color, Player

from catanrl.envs.puffer.common import (
    BOARD_HEIGHT,
    BOARD_WIDTH,
    COLOR_ORDER,
    EpisodeSeedTracker,
    MapType,
    build_shared_critic_observation_space,
    compute_multiagent_input_dim,
    normalize_reset_seed,
)
from catanrl.envs.zoo.rewards import ShapedReward, WinReward
from catanrl.features.catanatron_utils import (
    ActorObservationLevel,
    full_game_to_features,
    get_actor_indices_from_full,
    get_full_numeric_feature_names,
)
from catanrl.utils.catanatron_action_space import (
    from_action_space,
    get_action_space_size,
    get_end_turn_index,
    to_action_space,
)
from catanrl.utils.catanatron_map import build_catan_map
from catanrl.utils.seeding import derive_map_and_game_seeds, derive_seed

LocalObservation = Dict[str, np.ndarray]


@dataclass
class MultiAgentCatanatronEnvConfig:
    num_players: int = 2
    map_type: Literal["BASE", "TOURNAMENT", "MINI"] = "BASE"
    vps_to_win: int = 10
    discard_limit: int = 7
    shared_critic: bool = False
    reward_function: Literal["shaped", "win"] = "shaped"
    actor_observation_level: ActorObservationLevel = "private"


class ParallelCatanatronPufferEnv(PufferEnv):
    """Native multi-agent Puffer environment backed directly by `Game`."""

    def __init__(self, config: Optional[MultiAgentCatanatronEnvConfig] = None, buf=None):
        self.config = config or MultiAgentCatanatronEnvConfig()
        self.num_players = int(self.config.num_players)
        self.map_type: MapType = self.config.map_type
        self.action_space_size = get_action_space_size(self.num_players, self.map_type)
        self.end_turn_idx = get_end_turn_index(self.num_players, self.map_type)
        self.vps_to_win = int(self.config.vps_to_win)
        self.discard_limit = int(self.config.discard_limit)
        self.shared_critic = bool(self.config.shared_critic)
        self.actor_observation_level = self.config.actor_observation_level

        self.possible_agents = [f"player_{idx}" for idx in range(self.num_players)]
        self.colors_order = list(COLOR_ORDER[: self.num_players])
        self.agent_name_to_color = {
            f"player_{idx}": color for idx, color in enumerate(self.colors_order)
        }
        self.color_to_agent_name = {
            color: f"player_{idx}" for idx, color in enumerate(self.colors_order)
        }

        if self.config.reward_function == "shaped":
            self.reward_function = ShapedReward(self)
        elif self.config.reward_function == "win":
            self.reward_function = WinReward()
        else:
            raise ValueError(f"Invalid reward function: {self.config.reward_function}")

        actor_dim, board_shape, numeric_dim = compute_multiagent_input_dim(
            self.num_players,
            self.map_type,
            actor_observation_level=self.actor_observation_level,
        )
        self.vector_dim = actor_dim
        self.board_tensor_shape = board_shape
        self.numeric_dim = numeric_dim
        full_numeric_dim = len(get_full_numeric_feature_names(self.num_players, self.map_type))
        self.critic_vector_dim = self.vector_dim + (full_numeric_dim - self.numeric_dim)
        self.actor_indices = get_actor_indices_from_full(
            self.num_players,
            self.map_type,
            level=self.actor_observation_level,
        )
        self.actor_numeric_indices = self.actor_indices[: self.numeric_dim]

        self.env_single_observation_space = build_shared_critic_observation_space(
            numeric_dim=self.numeric_dim,
            board_tensor_shape=self.board_tensor_shape,
            critic_dim=self.critic_vector_dim,
            action_space_size=self.action_space_size,
        )
        self.env_single_action_space = spaces.Discrete(self.action_space_size)

        self.single_observation_space, self.obs_dtype = emulate_observation_space(
            self.env_single_observation_space
        )
        self.single_action_space, self.atn_dtype = emulate_action_space(self.env_single_action_space)
        self.num_agents = len(self.possible_agents)
        self.is_obs_emulated = self.single_observation_space is not self.env_single_observation_space
        self.is_atn_emulated = self.single_action_space is not self.env_single_action_space

        super().__init__(buf=buf)

        self.obs_struct = self.observations.view(self.obs_dtype)
        self.initialized = False
        self._all_done = False
        self._seed_tracker = EpisodeSeedTracker()
        self.game: Game | None = None
        self.agents: List[str] = []

    @property
    def done(self) -> bool:
        return len(self.agents) == 0 or self._all_done

    def _pack_observation_row(self, row: int, observation: Dict[str, Any]) -> None:
        if self.is_obs_emulated:
            emulate(self.obs_struct[row], observation)
        else:
            self.observations[row] = observation

    def _next_episode_seed(self, seed: int | None) -> int | None:
        return self._seed_tracker.next_episode_seed(seed, derive_seed)

    def _actor_observation(self, color: Color) -> LocalObservation:
        assert self.game is not None
        full_vector = full_game_to_features(
            self.game,
            self.num_players,
            self.map_type,
            base_color=color,
        )
        numeric = full_vector[self.actor_numeric_indices]
        board = create_board_tensor(self.game, color, channels_first=True).astype(
            np.float32,
            copy=False,
        )
        return {
            "numeric": numeric.astype(np.float32, copy=False),
            "board": board,
        }

    def _action_mask(self, agent: str) -> np.ndarray:
        assert self.game is not None
        mask = np.zeros(self.action_space_size, dtype=np.int8)
        current_color = self.game.state.current_color()
        if self.agents and self.agent_name_to_color[agent] == current_color:
            valid_actions = [
                to_action_space(
                    action,
                    self.num_players,
                    self.map_type,
                    tuple(self.game.state.colors),
                )
                for action in self.game.playable_actions
            ]
            mask[valid_actions] = 1
        else:
            mask[self.end_turn_idx] = 1
        return mask

    def _get_observation(self, agent: str) -> Dict[str, np.ndarray]:
        assert self.game is not None
        color = self.agent_name_to_color[agent]
        return {
            "observation": self._actor_observation(color),
            "action_mask": self._action_mask(agent),
            "critic": full_game_to_features(
                self.game,
                self.num_players,
                self.map_type,
                base_color=color,
            ),
        }

    def _get_info(self, agent: str) -> Dict[str, Any]:
        valid_actions = np.where(self._action_mask(agent) == 1)[0]
        info: Dict[str, Any] = {"valid_actions": valid_actions}
        if self.shared_critic:
            assert self.game is not None
            color = self.agent_name_to_color[agent]
            info["critic_observation"] = full_game_to_features(
                self.game,
                self.num_players,
                self.map_type,
                base_color=color,
            )
        return info

    def _winner_info(self) -> Dict[str, Any]:
        assert self.game is not None
        winning_color = self.game.winning_color()
        winner_agent = self.color_to_agent_name.get(winning_color) if winning_color else None
        winner_agent_index = (
            self.possible_agents.index(winner_agent) if winner_agent in self.possible_agents else -1
        )
        return {
            "winning_color": winning_color.value if winning_color is not None else None,
            "winner_agent": winner_agent,
            "winner_agent_index": winner_agent_index,
            "player_0_won": winner_agent == "player_0",
        }

    def reset(self, seed=None):
        seed = normalize_reset_seed(seed)
        episode_seed = self._next_episode_seed(seed)

        map_seed = None
        game_seed = None
        if episode_seed is not None:
            map_seed, game_seed = derive_map_and_game_seeds(episode_seed)

        catan_map = build_catan_map(self.map_type, seed=map_seed, number_placement="random")
        players = [Player(color) for color in self.colors_order]
        for player in players:
            player.reset_state()

        self.game = Game(
            players=players,
            seed=game_seed,
            discard_limit=self.discard_limit,
            catan_map=catan_map,
            vps_to_win=self.vps_to_win,
        )
        self.agents = self.possible_agents[:]
        self.reward_function.after_reset(self)

        observations = {agent: self._get_observation(agent) for agent in self.agents}
        infos = {agent: self._get_info(agent) for agent in self.agents}

        self.initialized = True
        self._all_done = False
        for idx, agent in enumerate(self.possible_agents):
            self._pack_observation_row(idx, observations[agent])
        self.rewards[:] = 0.0
        self.terminals[:] = False
        self.truncations[:] = False
        self.masks[:] = True

        info_list = [infos[agent] for agent in self.possible_agents]
        return self.observations, info_list

    def step(self, actions: np.ndarray):
        if not self.initialized:
            raise RuntimeError("step() before reset()")
        if self.done:
            observations, infos = self.reset(seed=None)
            return observations, self.rewards, self.terminals, self.truncations, infos

        if isinstance(actions, np.ndarray):
            if len(actions) != self.num_agents:
                raise ValueError(f"Expected {self.num_agents} actions, got {len(actions)}")
            action_dict = {
                agent: int(actions[idx]) for idx, agent in enumerate(self.possible_agents)
            }
        else:
            action_dict = actions

        unpacked: Dict[str, int] = {}
        for agent, action in action_dict.items():
            if agent not in self.possible_agents:
                raise ValueError(f"Unknown agent {agent}")
            if agent not in self.agents:
                continue
            if self.is_atn_emulated:
                unpacked[agent] = nativize(
                    np.asarray(action).reshape(self.actions[0].shape),
                    self.env_single_action_space,
                    self.atn_dtype,
                )
            else:
                unpacked[agent] = int(action)

        assert self.game is not None
        current_color = self.game.state.current_color()
        current_agent = self.color_to_agent_name[current_color]
        current_action = unpacked.get(current_agent)
        if current_action is not None:
            catan_action = from_action_space(
                int(current_action),
                current_color,
                self.num_players,
                self.map_type,
                tuple(self.game.state.colors),
                self.game.playable_actions,
            )
            if catan_action not in self.game.playable_actions:
                raise ValueError(f"Invalid action {current_action} for current state.")
            self.game.execute(catan_action)

        rewards = {agent: 0.0 for agent in self.agents}
        for agent in self.agents:
            rewards[agent] = self.reward_function.reward(
                self,
                self.game,
                self.agent_name_to_color[agent],
            )

        dones = {agent: False for agent in self.agents}
        truncateds = {agent: False for agent in self.agents}

        winning_color = self.game.winning_color()
        terminal_winner_info: Dict[str, Any] = {}
        if winning_color is not None:
            terminal_winner_info = self._winner_info()
            for agent in self.agents:
                dones[agent] = True
            self.agents = []
        elif self.game.state.num_turns >= TURNS_LIMIT:
            for agent in self.agents:
                truncateds[agent] = True
            self.agents = []

        for color in self.colors_order:
            self.reward_function.after_step(self, self.game, color)

        observations: Dict[str, Dict[str, np.ndarray]] = {}
        infos: Dict[str, Dict[str, Any]] = {}
        for agent in self.agents:
            observations[agent] = self._get_observation(agent)
            infos[agent] = self._get_info(agent)

        self.rewards[:] = 0.0
        self.terminals[:] = True
        self.truncations[:] = False
        for idx, agent in enumerate(self.possible_agents):
            if agent not in observations:
                self.observations[idx] = 0
                self.rewards[idx] = float(rewards.get(agent, 0.0))
                self.terminals[idx] = bool(dones.get(agent, True))
                self.truncations[idx] = bool(truncateds.get(agent, False))
                self.masks[idx] = False
                continue

            self._pack_observation_row(idx, observations[agent])
            self.rewards[idx] = float(rewards[agent])
            self.terminals[idx] = bool(dones[agent])
            self.truncations[idx] = bool(truncateds[agent])
            self.masks[idx] = True

        self._all_done = all(dones.values()) or all(truncateds.values())
        info_list: List[Dict[str, Any]] = []
        for agent in self.possible_agents:
            info = infos.get(agent, {})
            if terminal_winner_info:
                info = {**info, **terminal_winner_info}
            info_list.append(info)
        return self.observations, self.rewards, self.terminals, self.truncations, info_list

    def close(self):
        pass


def _make_parallel_env(
    num_players: int,
    map_type: Literal["BASE", "TOURNAMENT", "MINI"],
    vps_to_win: int,
    discard_limit: int,
    shared_critic: bool,
    reward_function: Literal["shaped", "win"],
    actor_observation_level: ActorObservationLevel = "private",
) -> Callable[..., ParallelCatanatronPufferEnv]:
    def _config() -> MultiAgentCatanatronEnvConfig:
        return MultiAgentCatanatronEnvConfig(
            num_players=num_players,
            map_type=map_type,
            vps_to_win=vps_to_win,
            discard_limit=discard_limit,
            shared_critic=shared_critic,
            reward_function=reward_function,
            actor_observation_level=actor_observation_level,
        )

    return lambda **kwargs: ParallelCatanatronPufferEnv(config=_config(), **kwargs)


def make_vectorized_envs(
    num_players: int,
    map_type: Literal["BASE", "TOURNAMENT", "MINI"],
    vps_to_win: int,
    discard_limit: int,
    shared_critic: bool,
    reward_function: Literal["shaped", "win"],
    num_envs: int,
    actor_observation_level: ActorObservationLevel = "private",
):
    return puffer_vector.make(
        _make_parallel_env(
            num_players,
            map_type,
            vps_to_win,
            discard_limit,
            shared_critic,
            reward_function,
            actor_observation_level,
        ),
        num_envs=num_envs,
        backend=puffer_vector.Multiprocessing,
    )


def flatten_marl_observation(observation: Dict[str, np.ndarray]) -> np.ndarray:
    numeric = np.asarray(observation["observation"]["numeric"], dtype=np.float32).reshape(-1)
    board_arr = np.asarray(observation["observation"]["board"], dtype=np.float32)
    if board_arr.ndim != 3:
        raise ValueError(f"Expected 3D board tensor, got shape {board_arr.shape}")

    if board_arr.shape[0] == BOARD_WIDTH and board_arr.shape[1] == BOARD_HEIGHT:
        board_wh_last = board_arr
    elif board_arr.shape[1] == BOARD_WIDTH and board_arr.shape[2] == BOARD_HEIGHT:
        board_wh_last = np.transpose(board_arr, (1, 2, 0))
    else:
        raise ValueError(
            "Unrecognized board tensor layout; expected (W,H,C) or (C,W,H), "
            f"got {board_arr.shape}"
        )
    return np.concatenate([numeric, board_wh_last.reshape(-1)], axis=0)


def get_valid_actions(
    info: Optional[Dict[str, Any]],
    observation: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    if observation is not None and "action_mask" in observation:
        mask = np.asarray(observation["action_mask"], dtype=np.int8).reshape(-1)
        return np.where(mask > 0)[0].astype(np.int64)
    if not info:
        return np.array([], dtype=np.int64)
    return np.asarray(info.get("valid_actions", []), dtype=np.int64)


__all__ = [
    "BOARD_HEIGHT",
    "BOARD_WIDTH",
    "COLOR_ORDER",
    "LocalObservation",
    "MultiAgentCatanatronEnvConfig",
    "ParallelCatanatronPufferEnv",
    "_make_parallel_env",
    "compute_multiagent_input_dim",
    "flatten_marl_observation",
    "get_valid_actions",
    "make_vectorized_envs",
]
