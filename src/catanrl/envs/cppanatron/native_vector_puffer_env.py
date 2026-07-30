from __future__ import annotations

from typing import Any, Dict, Literal, Optional

import numpy as np
from gymnasium import spaces
from pufferlib.emulation import emulate_action_space, emulate_observation_space
from pufferlib.environment import PufferEnv

from catanrl.envs.puffer.common import (
    EpisodeSeedTracker,
    MapType,
    build_shared_critic_observation_space,
    compute_multiagent_input_dim,
    normalize_reset_seed,
)
from catanrl.features.catanatron_utils import (
    ActorObservationLevel,
    get_full_numeric_feature_names,
)
from catanrl.utils.catanatron_action_space import get_action_space_size
from catanrl.utils.seeding import derive_map_and_game_seeds, derive_seed

from .batch_binding import NativeGameBatch
from .puffer_env import TURNS_LIMIT


class NativeVectorCppanatronPufferEnv(PufferEnv):
    """Puffer-native vector environment backed by one batched C++ call."""

    def __init__(self, config: Optional[Dict[str, Any]] = None, buf=None):
        self.config = config or {}
        self.num_envs = int(self.config.get("num_envs", 1))
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
        if self.num_envs <= 0:
            raise ValueError("num_envs must be positive")
        if not 2 <= self.num_players <= 4:
            raise ValueError("cppanatron supports between two and four players")
        if self.reward_function not in {"shaped", "win"}:
            raise ValueError(f"Unsupported native reward: {self.reward_function}")

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
        self.action_space_size = get_action_space_size(self.num_players, self.map_type)
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
        self.num_agents = self.num_envs * self.num_players
        self.agents_per_env = [self.num_players] * self.num_envs
        super().__init__(buf=buf)

        self._seed_trackers = [EpisodeSeedTracker() for _ in range(self.num_envs)]
        self._initialized = False
        self._batch = NativeGameBatch(
            num_envs=self.num_envs,
            num_players=self.num_players,
            map_type=self.map_type,
            discard_limit=self.discard_limit,
            vps_to_win=self.vps_to_win,
            reward_function=self.reward_function,
            turns_limit=TURNS_LIMIT,
            observations=self.observations,
            obs_dtype=self.obs_dtype,
            actions=self.actions,
            rewards=self.rewards,
            terminals=self.terminals,
            truncations=self.truncations,
            masks=self.masks,
        )

    def _episode_seed(self, env_index: int, root_seed: int | None) -> int:
        episode_seed = self._seed_trackers[env_index].next_episode_seed(root_seed, derive_seed)
        if episode_seed is None:
            return int(np.random.SeedSequence().generate_state(1)[0])
        return episode_seed

    def _seeds_for_reset(self, seed) -> tuple[np.ndarray, np.ndarray]:
        normalized_seed = normalize_reset_seed(seed)
        map_seeds = np.empty(self.num_envs, dtype=np.uint64)
        game_seeds = np.empty(self.num_envs, dtype=np.uint64)
        for env_index in range(self.num_envs):
            root_seed = (
                None
                if normalized_seed is None
                else derive_seed(normalized_seed, "native_env", env_index)
            )
            episode_seed = self._episode_seed(env_index, root_seed)
            map_seeds[env_index], game_seeds[env_index] = derive_map_and_game_seeds(episode_seed)
        return map_seeds, game_seeds

    def reset(self, seed=None):
        map_seeds, game_seeds = self._seeds_for_reset(seed)
        self._batch.reset_all(map_seeds, game_seeds)
        self._initialized = True
        return self.observations, [{} for _ in range(self.num_agents)]

    def step(self, actions: np.ndarray):
        if not self._initialized:
            raise RuntimeError("step() before reset()")
        actions = np.asarray(actions, dtype=np.int32)
        if actions.shape != self.actions.shape:
            raise ValueError(f"Expected actions with shape {self.actions.shape}")
        self.actions[:] = actions
        self._batch.step()

        done_envs = np.logical_or(
            self.terminals.reshape(self.num_envs, self.num_players).all(axis=1),
            self.truncations.reshape(self.num_envs, self.num_players).all(axis=1),
        )
        for env_index in np.flatnonzero(done_envs):
            episode_seed = self._episode_seed(int(env_index), None)
            map_seed, game_seed = derive_map_and_game_seeds(episode_seed)
            self._batch.reset_at(
                int(env_index),
                map_seed,
                game_seed,
                preserve_transition=True,
            )
        return (
            self.observations,
            self.rewards,
            self.terminals,
            self.truncations,
            [{} for _ in range(self.num_agents)],
        )

    def close(self):
        self._batch.close()


def make_cppanatron_native_marl_vectorized_envs(
    num_players: int,
    map_type: MapType,
    vps_to_win: int,
    discard_limit: int,
    shared_critic: bool,
    reward_function: Literal["shaped", "win"],
    num_envs: int,
    actor_observation_level: ActorObservationLevel = "private",
):
    return NativeVectorCppanatronPufferEnv(
        config={
            "num_envs": num_envs,
            "num_players": num_players,
            "map_type": map_type,
            "vps_to_win": vps_to_win,
            "discard_limit": discard_limit,
            "shared_critic": shared_critic,
            "reward_function": reward_function,
            "actor_observation_level": actor_observation_level,
        }
    )


__all__ = [
    "NativeVectorCppanatronPufferEnv",
    "make_cppanatron_native_marl_vectorized_envs",
]
