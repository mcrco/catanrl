from __future__ import annotations

from typing import Any, Callable, Dict, List, Literal, Optional

import numpy as np
import pufferlib.vector as puffer_vector
from gymnasium import spaces
from pufferlib.emulation import emulate, emulate_action_space, emulate_observation_space, nativize
from pufferlib.environment import PufferEnv

from catanatron.game import Game, TURNS_LIMIT
from catanatron.gym.board_tensor_features import create_board_tensor
from catanatron.models.player import Color, Player, RandomPlayer

from catanrl.envs.gym.rewards import RewardFunction, ShapedReward, WinReward
from catanrl.envs.puffer.common import (
    BOARD_HEIGHT,
    BOARD_WIDTH,
    EpisodeSeedTracker,
    MapType,
    build_shared_critic_observation_space,
    compute_single_agent_dims,
    create_expert,
    create_opponents,
    normalize_reset_seed,
)
from catanrl.features.catanatron_utils import (
    ActorObservationLevel,
    full_game_to_features,
    get_actor_indices_from_full,
)
from catanrl.utils.catanatron_action_space import (
    from_action_space,
    get_action_space_size,
    to_action_space,
)
from catanrl.utils.catanatron_map import build_catan_map
from catanrl.utils.seeding import derive_map_and_game_seeds, derive_seed


def _resolve_reward_function(reward_function: str | RewardFunction | None) -> RewardFunction:
    if isinstance(reward_function, RewardFunction):
        return reward_function
    if reward_function in (None, "shaped"):
        return ShapedReward()
    if reward_function == "win":
        return WinReward()
    raise ValueError(f"Unknown reward function: {reward_function}")


class SingleAgentCatanatronPufferEnv(PufferEnv):
    """Native single-agent Puffer environment backed directly by `Game`."""

    def __init__(self, config: Optional[Dict[str, Any]] = None, buf=None):
        self.config = config or {}
        self.map_type: MapType = self.config.get("map_type", "BASE")
        self.vps_to_win = int(self.config.get("vps_to_win", 15))
        self.discard_limit = int(self.config.get("discard_limit", 9))
        self.enemies: List[Player] = self.config.get(
            "enemies",
            [RandomPlayer(Color.RED)],
        )
        self.shared_critic = bool(self.config.get("shared_critic", True))
        if not self.shared_critic:
            raise ValueError("SingleAgentCatanatronPufferEnv requires shared_critic=True")
        self.actor_observation_level: ActorObservationLevel = self.config.get(
            "actor_observation_level", "private"
        )

        self.reward_function = _resolve_reward_function(self.config.get("reward_function"))
        self.expert_player: Player | None = self.config.get("expert_player")

        assert all(player.color != Color.BLUE for player in self.enemies)
        self.p0 = Player(Color.BLUE)
        self.players = [self.p0] + self.enemies
        self.num_players = len(self.players)
        self.action_space_size = get_action_space_size(self.num_players, self.map_type)

        dims = compute_single_agent_dims(
            self.num_players,
            self.map_type,
            actor_observation_level=self.actor_observation_level,
        )
        self.numeric_dim = dims["numeric_dim"]
        self.full_numeric_dim = dims["full_numeric_dim"]
        self.board_channels = dims["board_channels"]
        self.board_tensor_shape = (self.board_channels, BOARD_WIDTH, BOARD_HEIGHT)
        self.critic_vector_dim = dims["critic_dim"]
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
        self.num_agents = 1
        self.is_obs_emulated = self.single_observation_space is not self.env_single_observation_space
        self.is_atn_emulated = self.single_action_space is not self.env_single_action_space

        super().__init__(buf=buf)

        self.obs_struct = self.observations.view(self.obs_dtype)[0]
        self.initialized = False
        self._episode_done = True
        self._seed_tracker = EpisodeSeedTracker()
        self.game: Game | None = None

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

    def _advance_until_p0_decision(self) -> None:
        assert self.game is not None
        while (
            self.game.winning_color() is None
            and self.game.state.current_color() != self.p0.color
        ):
            self.game.play_tick()

    def get_valid_actions(self) -> List[int]:
        assert self.game is not None
        return [
            to_action_space(
                action,
                self.num_players,
                self.map_type,
                tuple(self.game.state.colors),
            )
            for action in self.game.playable_actions
        ]

    def _actor_observation(self) -> Dict[str, np.ndarray]:
        assert self.game is not None
        full_vector = full_game_to_features(
            self.game,
            self.num_players,
            self.map_type,
            base_color=self.p0.color,
        )
        numeric = full_vector[self.actor_numeric_indices]
        board = create_board_tensor(self.game, self.p0.color, channels_first=True).astype(
            np.float32,
            copy=False,
        )
        return {
            "numeric": numeric.astype(np.float32, copy=False),
            "board": board.astype(np.float32, copy=False),
        }

    def _critic_observation(self) -> np.ndarray:
        assert self.game is not None
        return full_game_to_features(self.game, self.num_players, self.map_type)

    def _action_mask(self) -> np.ndarray:
        mask = np.zeros(self.action_space_size, dtype=np.int8)
        mask[self.get_valid_actions()] = 1
        return mask

    def _get_expert_action(self) -> int:
        if self.expert_player is None:
            raise ValueError("No expert player configured")
        assert self.game is not None
        expert_catan_action = self.expert_player.decide(self.game, self.game.playable_actions)
        return to_action_space(
            expert_catan_action,
            self.num_players,
            self.map_type,
            tuple(self.game.state.colors),
        )

    def _build_info(self) -> Dict[str, Any]:
        assert self.game is not None
        winning_color = self.game.winning_color()
        is_terminal = winning_color is not None or self.game.state.num_turns >= TURNS_LIMIT
        info: Dict[str, Any] = {
            "valid_actions": self.get_valid_actions(),
            "nn_won": bool(winning_color == self.p0.color),
        }
        if self.expert_player is not None:
            valid_actions = info["valid_actions"]
            info["expert_action"] = (
                int(valid_actions[0]) if is_terminal else self._get_expert_action()
            )
        return info

    def _get_observation(self) -> Dict[str, Any]:
        return {
            "observation": self._actor_observation(),
            "action_mask": self._action_mask(),
            "critic": self._critic_observation(),
        }

    def reset(self, seed=None):
        seed = normalize_reset_seed(seed)
        episode_seed = self._next_episode_seed(seed)

        map_seed = None
        game_seed = None
        if episode_seed is not None:
            map_seed, game_seed = derive_map_and_game_seeds(episode_seed)

        catan_map = build_catan_map(self.map_type, seed=map_seed, number_placement="random")
        for player in self.players:
            player.reset_state()
        self.game = Game(
            players=self.players,
            seed=game_seed,
            catan_map=catan_map,
            discard_limit=self.discard_limit,
            vps_to_win=self.vps_to_win,
        )
        if self.expert_player is not None:
            self.expert_player.reset_state()

        self._advance_until_p0_decision()
        observation = self._get_observation()
        info = self._build_info()
        self.reward_function.after_reset(self)

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
            return observations, self.rewards, self.terminals, self.truncations, infos

        if isinstance(actions, np.ndarray):
            action = actions.ravel()
            if isinstance(self.single_action_space, spaces.Discrete):
                action = int(action[0])
        else:
            action = actions

        if self.is_atn_emulated:
            action = nativize(
                np.asarray(action).reshape(self.actions[0].shape),
                self.env_single_action_space,
                self.atn_dtype,
            )

        assert self.game is not None
        catan_action = from_action_space(
            int(action),
            self.p0.color,
            self.num_players,
            self.map_type,
            tuple(self.game.state.colors),
            self.game.playable_actions,
        )
        if catan_action not in self.game.playable_actions:
            raise ValueError(f"Invalid action {action} for current state.")

        self.game.execute(catan_action)
        self._advance_until_p0_decision()

        observation = self._get_observation()
        info = self._build_info()
        winning_color = self.game.winning_color()
        terminated = winning_color is not None
        truncated = self.game.state.num_turns >= TURNS_LIMIT
        reward = self.reward_function.reward(self, self.game, self.p0.color)
        self.reward_function.after_step(self, self.game, self.p0.color)

        self._pack_observation(observation)
        self.rewards[0] = float(reward)
        self.terminals[0] = bool(terminated)
        self.truncations[0] = bool(truncated)
        self.masks[0] = True
        self._episode_done = bool(terminated or truncated)
        return self.observations, self.rewards, self.terminals, self.truncations, [info]

    def close(self):
        pass


def _make_puffer_env(
    reward_function: str,
    map_type: MapType,
    opponent_configs: List[str],
    vps_to_win: int = 15,
    discard_limit: int = 9,
    expert_config: str | None = None,
    actor_observation_level: ActorObservationLevel = "private",
) -> Callable[..., SingleAgentCatanatronPufferEnv]:
    def _config() -> Dict[str, Any]:
        expert_player = create_expert(expert_config) if expert_config else None
        return {
            "map_type": map_type,
            "vps_to_win": vps_to_win,
            "discard_limit": discard_limit,
            "enemies": create_opponents(opponent_configs),
            "reward_function": _resolve_reward_function(reward_function),
            "shared_critic": True,
            "expert_player": expert_player,
            "actor_observation_level": actor_observation_level,
        }

    return lambda **kwargs: SingleAgentCatanatronPufferEnv(config=_config(), **kwargs)


def make_puffer_vectorized_envs(
    reward_function: str,
    map_type: Literal["BASE", "MINI", "TOURNAMENT"],
    opponent_configs: List[str],
    num_envs: int,
    vps_to_win: int = 15,
    discard_limit: int = 9,
    expert_config: str | None = None,
    actor_observation_level: ActorObservationLevel = "private",
):
    return puffer_vector.make(
        _make_puffer_env(
            reward_function,
            map_type,
            opponent_configs,
            vps_to_win,
            discard_limit,
            expert_config,
            actor_observation_level,
        ),
        num_envs=num_envs,
        backend=puffer_vector.Multiprocessing,
    )


__all__ = [
    "BOARD_HEIGHT",
    "BOARD_WIDTH",
    "SingleAgentCatanatronPufferEnv",
    "_make_puffer_env",
    "compute_single_agent_dims",
    "create_expert",
    "create_opponents",
    "make_puffer_vectorized_envs",
]
