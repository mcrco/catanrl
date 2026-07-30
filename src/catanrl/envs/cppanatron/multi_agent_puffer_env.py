from __future__ import annotations

from typing import Any, Callable, Dict, List, Literal, Optional

import numpy as np
import pufferlib.vector as puffer_vector
from gymnasium import spaces
from pufferlib.emulation import (
    emulate,
    emulate_action_space,
    emulate_observation_space,
    nativize,
)
from pufferlib.environment import PufferEnv

from catanrl.envs.puffer.common import (
    COLOR_ORDER,
    EpisodeSeedTracker,
    MapType,
    build_shared_critic_observation_space,
    compute_multiagent_input_dim,
    normalize_reset_seed,
)
from catanrl.features.catanatron_utils import (
    ActorObservationLevel,
    get_actor_indices_from_full,
    get_full_numeric_feature_names,
)
from catanrl.utils.catanatron_action_space import (
    get_action_space_size,
    get_end_turn_index,
)
from catanrl.utils.seeding import derive_map_and_game_seeds, derive_seed

from .binding import NativeGame
from .features import full_native_features
from .puffer_env import TURNS_LIMIT, _native_production_sum


class ParallelCppanatronPufferEnv(PufferEnv):
    """Multi-agent Puffer environment backed by the native C++ engine."""

    def __init__(self, config: Optional[Dict[str, Any]] = None, buf=None):
        self.config = config or {}
        self.num_players = int(self.config.get("num_players", 2))
        self.map_type: MapType = self.config.get("map_type", "BASE")
        self.vps_to_win = int(self.config.get("vps_to_win", 10))
        self.discard_limit = int(self.config.get("discard_limit", 7))
        self.shared_critic = bool(self.config.get("shared_critic", False))
        self.reward_function: Literal["shaped", "win"] = self.config.get(
            "reward_function", "shaped"
        )
        self.actor_observation_level: ActorObservationLevel = self.config.get(
            "actor_observation_level", "private"
        )
        if not 2 <= self.num_players <= 4:
            raise ValueError("cppanatron supports between two and four players")
        if self.reward_function not in {"shaped", "win"}:
            raise ValueError(f"Unsupported native reward: {self.reward_function}")

        self.possible_agents = [f"player_{player}" for player in range(self.num_players)]
        self.colors_order = list(COLOR_ORDER[: self.num_players])
        self.action_space_size = get_action_space_size(self.num_players, self.map_type)
        self.end_turn_idx = get_end_turn_index(self.num_players, self.map_type)

        actor_dim, board_shape, actor_numeric_dim = compute_multiagent_input_dim(
            self.num_players,
            self.map_type,
            actor_observation_level=self.actor_observation_level,
        )
        self.vector_dim = actor_dim
        self.board_tensor_shape = board_shape
        self.actor_numeric_dim = actor_numeric_dim
        self.numeric_dim = actor_numeric_dim
        self.critic_numeric_dim = len(
            get_full_numeric_feature_names(self.num_players, self.map_type)
        )
        self.critic_vector_dim = self.vector_dim + self.critic_numeric_dim - self.actor_numeric_dim
        self.actor_observation_indices = get_actor_indices_from_full(
            self.num_players,
            self.map_type,
            level=self.actor_observation_level,
        )
        self.actor_numeric_indices = self.actor_observation_indices[: self.actor_numeric_dim]

        self.env_single_observation_space = build_shared_critic_observation_space(
            numeric_dim=self.actor_numeric_dim,
            board_tensor_shape=self.board_tensor_shape,
            critic_dim=self.critic_vector_dim,
            action_space_size=self.action_space_size,
        )
        self.env_single_action_space = spaces.Discrete(self.action_space_size)
        self.single_observation_space, self.obs_dtype = emulate_observation_space(
            self.env_single_observation_space
        )
        self.single_action_space, self.atn_dtype = emulate_action_space(
            self.env_single_action_space
        )
        self.num_agents = self.num_players
        self.is_obs_emulated = (
            self.single_observation_space is not self.env_single_observation_space
        )
        self.is_atn_emulated = self.single_action_space is not self.env_single_action_space

        super().__init__(buf=buf)
        self.obs_struct = self.observations.view(self.obs_dtype)
        self._seed_tracker = EpisodeSeedTracker()
        self.game: NativeGame | None = None
        self.agents: List[str] = []
        self.initialized = False
        self._all_done = True
        self._prev_vps = np.zeros(self.num_players, dtype=np.int32)
        self._prev_production = np.zeros(self.num_players, dtype=np.float64)

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

    def _full_observation(self, player: int) -> np.ndarray:
        assert self.game is not None
        return full_native_features(
            self.game,
            self.map_type,
            base_player=player,
        )

    def _action_mask(self, player: int) -> np.ndarray:
        assert self.game is not None
        if self.agents and player == self.game.current_player:
            return self.game.valid_action_mask().astype(np.int8, copy=False)
        mask = np.zeros(self.action_space_size, dtype=np.int8)
        mask[self.end_turn_idx] = 1
        return mask

    def _get_observation(self, player: int) -> Dict[str, np.ndarray]:
        return {
            "observation": self._full_observation(player),
            "action_mask": self._action_mask(player),
        }

    def _get_info(self, player: int) -> Dict[str, Any]:
        info: Dict[str, Any] = {
            "valid_actions": np.flatnonzero(self._action_mask(player)),
        }
        if self.shared_critic:
            info["critic_observation"] = self._full_observation(player)
        return info

    def _winner_info(self) -> Dict[str, Any]:
        assert self.game is not None
        winner = self.game.winner
        winner_agent = None if winner is None else self.possible_agents[winner]
        return {
            "winning_color": (None if winner is None else self.colors_order[winner].value),
            "winner_agent": winner_agent,
            "winner_agent_index": -1 if winner is None else winner,
            "player_0_won": winner == 0,
        }

    def _reward(self, player: int) -> float:
        assert self.game is not None
        if self.game.winner == player:
            return 1.0
        if self.reward_function == "win":
            return -1.0 if self.game.winner is not None else 0.0

        vps = self.game.player_state(player).actual_victory_points
        production = _native_production_sum(self.game, player)
        reward = 0.01 * ((vps - self._prev_vps[player]) / self.vps_to_win)
        reward += 0.0025 * (production - self._prev_production[player])
        return float(reward)

    def _update_reward_state(self) -> None:
        assert self.game is not None
        for player in range(self.num_players):
            self._prev_vps[player] = self.game.player_state(player).actual_victory_points
            self._prev_production[player] = _native_production_sum(self.game, player)

    def reset(self, seed=None):
        normalized_seed = normalize_reset_seed(seed)
        episode_seed = self._next_episode_seed(normalized_seed)
        if episode_seed is None:
            episode_seed = int(np.random.SeedSequence().generate_state(1)[0])
        map_seed, game_seed = derive_map_and_game_seeds(episode_seed)

        if self.game is None:
            self.game = NativeGame(
                self.num_players,
                self.map_type,
                seed=game_seed,
                map_seed=map_seed,
                number_placement="random",
                discard_limit=self.discard_limit,
                vps_to_win=self.vps_to_win,
            )
        else:
            self.game.reset(game_seed, map_seed=map_seed)

        self.agents = self.possible_agents[:]
        self._prev_vps.fill(0)
        self._prev_production.fill(0.0)
        self.initialized = True
        self._all_done = False

        infos = []
        for player in range(self.num_players):
            self._pack_observation_row(player, self._get_observation(player))
            infos.append(self._get_info(player))
        self.rewards[:] = 0.0
        self.terminals[:] = False
        self.truncations[:] = False
        self.masks[:] = True
        return self.observations, infos

    def step(self, actions: np.ndarray):
        if not self.initialized:
            raise RuntimeError("step() before reset()")
        if self.done:
            observations, infos = self.reset(seed=None)
            return (
                observations,
                self.rewards,
                self.terminals,
                self.truncations,
                infos,
            )
        if not isinstance(actions, np.ndarray) or len(actions) != self.num_agents:
            raise ValueError(f"Expected {self.num_agents} actions")

        unpacked = np.asarray(actions).reshape(self.num_agents)
        if self.is_atn_emulated:
            unpacked = np.asarray(
                [
                    nativize(
                        np.asarray(action).reshape(self.actions[0].shape),
                        self.env_single_action_space,
                        self.atn_dtype,
                    )
                    for action in unpacked
                ],
                dtype=np.int64,
            )

        assert self.game is not None
        current_player = self.game.current_player
        action = int(unpacked[current_player])
        if not self._action_mask(current_player)[action]:
            raise ValueError(f"Invalid action {action} for current state.")
        self.game.step(action)

        rewards = np.asarray(
            [self._reward(player) for player in range(self.num_players)],
            dtype=np.float32,
        )
        self._update_reward_state()
        terminated = self.game.winner is not None
        truncated = self.game.num_turns >= TURNS_LIMIT
        terminal_info = self._winner_info() if terminated else {}
        if terminated or truncated:
            self.agents = []

        self.rewards[:] = rewards
        self.terminals[:] = terminated
        self.truncations[:] = truncated
        infos: List[Dict[str, Any]] = []
        for player in range(self.num_players):
            if self.agents:
                self._pack_observation_row(player, self._get_observation(player))
                self.masks[player] = True
                info = self._get_info(player)
            else:
                self.observations[player] = 0
                self.masks[player] = False
                info = {}
            infos.append({**info, **terminal_info})

        self._all_done = bool(terminated or truncated)
        return (
            self.observations,
            self.rewards,
            self.terminals,
            self.truncations,
            infos,
        )

    def close(self):
        if self.game is not None:
            self.game.close()
            self.game = None


def _make_cppanatron_marl_env(
    num_players: int,
    map_type: MapType,
    vps_to_win: int,
    discard_limit: int,
    shared_critic: bool,
    reward_function: Literal["shaped", "win"],
    actor_observation_level: ActorObservationLevel = "private",
) -> Callable[..., ParallelCppanatronPufferEnv]:
    config = {
        "num_players": num_players,
        "map_type": map_type,
        "vps_to_win": vps_to_win,
        "discard_limit": discard_limit,
        "shared_critic": shared_critic,
        "reward_function": reward_function,
        "actor_observation_level": actor_observation_level,
    }
    return lambda **kwargs: ParallelCppanatronPufferEnv(
        config=config,
        **kwargs,
    )


def make_cppanatron_marl_vectorized_envs(
    num_players: int,
    map_type: MapType,
    vps_to_win: int,
    discard_limit: int,
    shared_critic: bool,
    reward_function: Literal["shaped", "win"],
    num_envs: int,
    actor_observation_level: ActorObservationLevel = "private",
):
    return puffer_vector.make(
        _make_cppanatron_marl_env(
            num_players=num_players,
            map_type=map_type,
            vps_to_win=vps_to_win,
            discard_limit=discard_limit,
            shared_critic=shared_critic,
            reward_function=reward_function,
            actor_observation_level=actor_observation_level,
        ),
        num_envs=num_envs,
        backend=puffer_vector.Multiprocessing,
    )


__all__ = [
    "ParallelCppanatronPufferEnv",
    "_make_cppanatron_marl_env",
    "make_cppanatron_marl_vectorized_envs",
]
