from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

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
    BOARD_HEIGHT,
    BOARD_WIDTH,
    EpisodeSeedTracker,
    MapType,
    build_shared_critic_observation_space,
    compute_single_agent_dims,
    normalize_reset_seed,
)
from catanrl.features.catanatron_utils import (
    ActorObservationLevel,
    get_actor_indices_from_full,
)
from catanrl.utils.seeding import derive_map_and_game_seeds, derive_seed

from .binding import NativeGame
from .features import full_native_features

TURNS_LIMIT = 1_000


def _native_production_sum(game: NativeGame) -> float:
    robber = game.robber_coordinate
    buildings = {node: building for node, _color, building in game.buildings()}
    probability = {
        number: (6 - abs(7 - number)) / 36.0 for number in range(2, 13)
    }
    total = 0.0
    for x, y, z, _id, kind, resource, number, _direction, nodes in game.tiles():
        if kind != 0 or resource < 0 or (x, y, z) == robber:
            continue
        for node in nodes:
            building = buildings.get(node)
            if building is not None:
                total += probability[number] * (1 if building == 0 else 2)
    return total


class SingleAgentCppanatronPufferEnv(PufferEnv):
    """Single-agent Puffer environment backed by the native C++ engine."""

    def __init__(self, config: Optional[Dict[str, Any]] = None, buf=None):
        self.config = config or {}
        self.map_type: MapType = self.config.get("map_type", "BASE")
        self.vps_to_win = int(self.config.get("vps_to_win", 15))
        self.discard_limit = int(self.config.get("discard_limit", 9))
        self.opponent_configs: List[str] = list(
            self.config.get("opponent_configs", ["F"])
        )
        self.expert_config: str | None = self.config.get("expert_config")
        self.reward_function = str(self.config.get("reward_function", "shaped"))
        self.nn_seat = str(self.config.get("nn_seat", "random"))
        self.actor_observation_level: ActorObservationLevel = self.config.get(
            "actor_observation_level", "private"
        )
        self.num_players = len(self.opponent_configs) + 1
        if self.num_players > 4:
            raise ValueError("cppanatron supports at most four players")
        if self.expert_config is not None and self.expert_config.upper() not in {
            "F",
            "VALUE",
            "VALUEFUNCTION",
        }:
            raise ValueError("Native expert currently supports only F/ValueFunction")
        supported_opponents = {"F", "VALUE", "VALUEFUNCTION", "R", "RANDOM"}
        unsupported_opponents = [
            config
            for config in self.opponent_configs
            if config.split(":", 1)[0].upper() not in supported_opponents
        ]
        if unsupported_opponents:
            raise ValueError(
                "Native opponents currently support only F/ValueFunction and Random; "
                f"got {unsupported_opponents}"
            )

        dims = compute_single_agent_dims(
            self.num_players,
            self.map_type,
            actor_observation_level=self.actor_observation_level,
        )
        self.actor_numeric_dim = dims["actor_numeric_dim"]
        self.numeric_dim = self.actor_numeric_dim
        self.critic_numeric_dim = dims["critic_numeric_dim"]
        self.board_channels = dims["board_channels"]
        self.board_tensor_shape = (self.board_channels, BOARD_WIDTH, BOARD_HEIGHT)
        self.critic_vector_dim = dims["critic_dim"]
        self.actor_observation_indices = get_actor_indices_from_full(
            self.num_players,
            self.map_type,
            level=self.actor_observation_level,
        )
        self.actor_numeric_indices = self.actor_observation_indices[
            : self.actor_numeric_dim
        ]

        probe = NativeGame(self.num_players, self.map_type, seed=0)
        self.action_space_size = probe.action_space_size
        probe.close()
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
        self.num_agents = 1
        self.is_obs_emulated = (
            self.single_observation_space is not self.env_single_observation_space
        )
        self.is_atn_emulated = (
            self.single_action_space is not self.env_single_action_space
        )

        super().__init__(buf=buf)
        self.obs_struct = self.observations.view(self.obs_dtype)[0]
        self._seed_tracker = EpisodeSeedTracker()
        self._rng = np.random.default_rng()
        self.game: NativeGame | None = None
        self.controlled_player = 0
        self._opponent_configs_by_player: list[str | None] = []
        self.initialized = False
        self._episode_done = True
        self._prev_vps = 0
        self._prev_production = 0.0

    @property
    def done(self) -> bool:
        return self._episode_done

    def _pack_observation(self, observation: Dict[str, Any]) -> None:
        if self.is_obs_emulated:
            emulate(self.obs_struct, observation)
        else:
            self.observations[:] = observation

    def _next_episode_seed(self, seed: int | None) -> int | None:
        return self._seed_tracker.next_episode_seed(seed, derive_seed)

    def _opponent_action(self, player: int) -> int:
        assert self.game is not None
        configured_opponent = self._opponent_configs_by_player[player]
        if configured_opponent is None:
            raise RuntimeError("Requested an opponent action for the controlled player")
        config = configured_opponent.split(":", 1)[0].upper()
        if config in {"F", "VALUE", "VALUEFUNCTION"}:
            return self.game.value_action()
        if config in {"R", "RANDOM"}:
            valid = np.flatnonzero(self.game.valid_action_mask())
            return int(self._rng.choice(valid))
        raise ValueError(f"Unsupported native opponent: {configured_opponent}")

    def _advance_until_controlled_decision(self) -> None:
        assert self.game is not None
        while (
            self.game.winner is None
            and self.game.num_turns < TURNS_LIMIT
            and self.game.current_player != self.controlled_player
        ):
            self.game.step(self._opponent_action(self.game.current_player))

    def _full_observation(self) -> np.ndarray:
        assert self.game is not None
        return full_native_features(
            self.game,
            self.map_type,
            base_player=self.controlled_player,
        )

    def _get_observation(self) -> Dict[str, Any]:
        assert self.game is not None
        return {
            "observation": self._full_observation(),
            "action_mask": self.game.valid_action_mask().astype(np.int8, copy=False),
        }

    def _build_info(self) -> Dict[str, Any]:
        assert self.game is not None
        valid_actions = np.flatnonzero(self.game.valid_action_mask()).tolist()
        is_terminal = (
            self.game.winner is not None or self.game.num_turns >= TURNS_LIMIT
        )
        info: Dict[str, Any] = {
            "valid_actions": valid_actions,
            "nn_won": self.game.winner == self.controlled_player,
        }
        if self.expert_config is not None:
            info["expert_action"] = (
                int(valid_actions[0]) if is_terminal else self.game.value_action()
            )
        return info

    def _reward(self) -> float:
        assert self.game is not None
        if self.game.winner == self.controlled_player:
            return 1.0
        if self.reward_function == "win":
            return -1.0 if self.game.winner is not None else 0.0
        if self.reward_function != "shaped":
            raise ValueError(f"Unsupported native reward: {self.reward_function}")
        vps = self.game.player_state(self.controlled_player).actual_victory_points
        production = _native_production_sum(self.game)
        reward = 0.01 * ((vps - self._prev_vps) / self.vps_to_win)
        reward += 0.0025 * (production - self._prev_production)
        self._prev_vps = vps
        self._prev_production = production
        return float(reward)

    def reset(self, seed=None):
        normalized_seed = normalize_reset_seed(seed)
        episode_seed = self._next_episode_seed(normalized_seed)
        if episode_seed is None:
            episode_seed = int(np.random.SeedSequence().generate_state(1)[0])
        self._rng = np.random.default_rng(episode_seed)
        if self.nn_seat == "first":
            self.controlled_player = 0
        elif self.nn_seat == "second":
            self.controlled_player = 1
        elif self.nn_seat == "random":
            self.controlled_player = int(self._rng.integers(self.num_players))
        else:
            raise ValueError(f"Unknown nn_seat: {self.nn_seat}")

        opponent_configs = list(self.opponent_configs)
        if self.nn_seat == "random":
            self._rng.shuffle(opponent_configs)
        config_iterator = iter(opponent_configs)
        self._opponent_configs_by_player = [
            None if player == self.controlled_player else next(config_iterator)
            for player in range(self.num_players)
        ]

        map_seed, game_seed = derive_map_and_game_seeds(episode_seed)
        if self.game is None:
            self.game = NativeGame(
                self.num_players,
                self.map_type,
                seed=game_seed,
                map_seed=map_seed,
                discard_limit=self.discard_limit,
                vps_to_win=self.vps_to_win,
            )
        else:
            self.game.reset(game_seed, map_seed=map_seed)
        self._advance_until_controlled_decision()
        observation = self._get_observation()
        info = self._build_info()
        self._prev_vps = 0
        self._prev_production = 0.0

        self._pack_observation(observation)
        self.rewards[0] = 0.0
        self.terminals[0] = False
        self.truncations[0] = False
        self.masks[0] = True
        self.initialized = True
        self._episode_done = False
        return self.observations, [info]

    def step(self, actions: np.ndarray):
        if not self.initialized:
            raise RuntimeError("step() before reset()")
        if self._episode_done:
            observations, infos = self.reset(seed=None)
            return (
                observations,
                self.rewards,
                self.terminals,
                self.truncations,
                infos,
            )
        if isinstance(actions, np.ndarray):
            action: Any = int(actions.ravel()[0])
        else:
            action = actions
        if self.is_atn_emulated:
            action = nativize(
                np.asarray(action).reshape(self.actions[0].shape),
                self.env_single_action_space,
                self.atn_dtype,
            )

        assert self.game is not None
        self.game.step(int(action))
        self._advance_until_controlled_decision()
        terminated = self.game.winner is not None
        truncated = self.game.num_turns >= TURNS_LIMIT
        reward = self._reward()
        final_info = self._build_info()
        done = bool(terminated or truncated)
        if done:
            final_reward = reward
            _, reset_infos = self.reset(seed=None)
            info = reset_infos[0]
            info["nn_won"] = final_info["nn_won"]
            info["final_info"] = final_info
            self.rewards[0] = final_reward
            self.terminals[0] = bool(terminated)
            self.truncations[0] = bool(truncated)
            self.masks[0] = True
            return (
                self.observations,
                self.rewards,
                self.terminals,
                self.truncations,
                [info],
            )

        observation = self._get_observation()
        info = self._build_info()
        self._pack_observation(observation)
        self.rewards[0] = reward
        self.terminals[0] = False
        self.truncations[0] = False
        self.masks[0] = True
        self._episode_done = False
        return (
            self.observations,
            self.rewards,
            self.terminals,
            self.truncations,
            [info],
        )

    def close(self):
        if self.game is not None:
            self.game.close()
            self.game = None


def _make_cppanatron_puffer_env(
    reward_function: str,
    map_type: MapType,
    opponent_configs: List[str],
    nn_seat: str = "random",
    vps_to_win: int = 15,
    discard_limit: int = 9,
    expert_config: str | None = None,
    actor_observation_level: ActorObservationLevel = "private",
) -> Callable[..., SingleAgentCppanatronPufferEnv]:
    def _config() -> Dict[str, Any]:
        return {
            "map_type": map_type,
            "vps_to_win": vps_to_win,
            "discard_limit": discard_limit,
            "opponent_configs": opponent_configs,
            "reward_function": reward_function,
            "expert_config": expert_config,
            "actor_observation_level": actor_observation_level,
            "nn_seat": nn_seat,
        }

    return lambda **kwargs: SingleAgentCppanatronPufferEnv(
        config=_config(), **kwargs
    )


def make_cppanatron_vectorized_envs(
    reward_function: str,
    map_type: MapType,
    opponent_configs: List[str],
    num_envs: int,
    nn_seat: str = "random",
    vps_to_win: int = 15,
    discard_limit: int = 9,
    expert_config: str | None = None,
    actor_observation_level: ActorObservationLevel = "private",
):
    return puffer_vector.make(
        _make_cppanatron_puffer_env(
            reward_function,
            map_type,
            opponent_configs,
            nn_seat,
            vps_to_win,
            discard_limit,
            expert_config,
            actor_observation_level,
        ),
        num_envs=num_envs,
        backend=puffer_vector.Multiprocessing,
    )
