"""
Native PufferLib environments for Catan (single-agent and parallel multi-agent).

Uses the same flattened observation layout as GymnasiumPufferEnv / PettingZooPufferEnv
(emulate_observation_space + emulate) so decode_puffer_batch and training loops stay
compatible.

obs_struct: For Dict observation spaces, self.observations has dtype matching the flat
Box; use self.observations.view(self.obs_dtype) for structured emulate() writes.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pufferlib
from pufferlib.emulation import emulate, emulate_action_space, emulate_observation_space, nativize
from pufferlib.environment import PufferEnv

from catanrl.envs.gym.single_env import SingleAgentCatanatronEnv
from catanrl.envs.zoo.multi_env import MultiAgentCatanatronEnvConfig
from catanrl.envs.zoo.parallel_env import ParallelCatanatronEnv


def _normalize_reset_seed(seed: Any) -> Optional[int]:
    if seed is None:
        return None
    if isinstance(seed, (list, tuple, np.ndarray)):
        if len(seed) == 0:
            return None
        return int(seed[0])
    return int(seed)


class SingleAgentCatanatronPufferEnv(PufferEnv):
    """Native PufferEnv wrapping SingleAgentCatanatronEnv (shared_critic / PPO-DAgger)."""

    def __init__(self, config: Optional[Dict[str, Any]] = None, buf=None):
        self._inner = SingleAgentCatanatronEnv(config=config or {})
        if not self._inner.shared_critic:
            raise ValueError("SingleAgentCatanatronPufferEnv requires shared_critic=True")

        self.single_observation_space, self.obs_dtype = emulate_observation_space(
            self._inner.observation_space
        )
        self.single_action_space, self.atn_dtype = emulate_action_space(self._inner.action_space)
        self.num_agents = 1
        self.is_obs_emulated = self.single_observation_space is not self._inner.observation_space
        self.is_atn_emulated = self.single_action_space is not self._inner.action_space

        super().__init__(buf=buf)

        if isinstance(self._inner.observation_space, pufferlib.spaces.Box):
            self.obs_struct = self.observations
        else:
            self.obs_struct = self.observations.view(self.obs_dtype)[0]

        self.initialized = False
        self._episode_done = True

    @property
    def done(self) -> bool:
        """Match GymnasiumPufferEnv so Serial vectorization resets between episodes."""
        return self._episode_done

    def _pack_observation(self, ob: Dict[str, Any]) -> None:
        if self.is_obs_emulated:
            emulate(self.obs_struct, ob)
        else:
            self.observations[:] = ob

    def reset(self, seed=None):
        seed = _normalize_reset_seed(seed)
        ob, info = self._inner.reset(seed=seed)
        self._pack_observation(ob)
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

        # Multiprocessing worker always calls step; Serial calls reset() when done.
        if self._episode_done:
            self.reset(seed=None)

        if isinstance(actions, np.ndarray):
            action = actions.ravel()
            if isinstance(self.single_action_space, pufferlib.spaces.Discrete):
                action = int(action[0])
        else:
            action = actions

        if self.is_atn_emulated:
            action = nativize(
                np.asarray(action).reshape(self.actions[0].shape),
                self._inner.action_space,
                self.atn_dtype,
            )

        ob, reward, terminated, truncated, info = self._inner.step(action)
        self._pack_observation(ob)
        self.rewards[0] = float(reward)
        self.terminals[0] = bool(terminated)
        self.truncations[0] = bool(truncated)
        self.masks[0] = True
        self._episode_done = bool(terminated or truncated)
        return self.observations, self.rewards, self.terminals, self.truncations, [info]

    def close(self):
        pass


class ParallelCatanatronPufferEnv(PufferEnv):
    """Native PufferEnv wrapping ParallelCatanatronEnv (central-critic MARL)."""

    def __init__(self, config: Optional[MultiAgentCatanatronEnvConfig] = None, buf=None):
        self._inner = ParallelCatanatronEnv(config=config)
        single_agent = self._inner.possible_agents[0]
        env_obs_space = self._inner.observation_spaces[single_agent]
        env_atn_space = self._inner.action_spaces[single_agent]

        self.single_observation_space, self.obs_dtype = emulate_observation_space(env_obs_space)
        self.single_action_space, self.atn_dtype = emulate_action_space(env_atn_space)
        self.num_agents = len(self._inner.possible_agents)
        self.is_obs_emulated = self.single_observation_space is not env_obs_space
        self.is_atn_emulated = self.single_action_space is not env_atn_space

        super().__init__(buf=buf)

        if isinstance(env_obs_space, pufferlib.spaces.Box):
            self.obs_struct = self.observations
        else:
            self.obs_struct = self.observations.view(self.obs_dtype)

        self.possible_agents = list(self._inner.possible_agents)
        self.initialized = False
        self._all_done = False

    @property
    def done(self) -> bool:
        return len(self._inner.agents) == 0 or self._all_done

    def _pack_observation_row(self, row: int, ob: Dict[str, Any]) -> None:
        if self.is_obs_emulated:
            emulate(self.obs_struct[row], ob)
        else:
            self.observations[row] = ob

    def reset(self, seed=None):
        seed = _normalize_reset_seed(seed)
        obs, infos = self._inner.reset(seed=seed)
        self.initialized = True
        self._all_done = False

        for i, agent in enumerate(self.possible_agents):
            self._pack_observation_row(i, obs[agent])

        self.rewards[:] = 0.0
        self.terminals[:] = False
        self.truncations[:] = False
        self.masks[:] = True

        info_list = [infos[agent] for agent in self.possible_agents]
        return self.observations, info_list

    def step(self, actions: np.ndarray):
        if not self.initialized:
            raise RuntimeError("step() before reset()")

        # MP worker: always step — start a new episode if the last one finished.
        if not self._inner.agents:
            self.reset(seed=None)
            self._all_done = False

        if isinstance(actions, np.ndarray):
            if len(actions) != self.num_agents:
                raise ValueError(
                    f"Expected {self.num_agents} actions, got {len(actions)}"
                )
            action_dict = {
                agent: int(actions[i]) for i, agent in enumerate(self.possible_agents)
            }
        else:
            action_dict = actions

        unpacked: Dict[str, int] = {}
        for agent, atn in action_dict.items():
            if agent not in self.possible_agents:
                raise ValueError(f"Unknown agent {agent}")
            if agent not in self._inner.agents:
                continue
            if self.is_atn_emulated:
                unpacked[agent] = nativize(
                    np.asarray(atn).reshape(self.actions[0].shape),
                    self._inner.action_spaces[agent],
                    self.atn_dtype,
                )
            else:
                unpacked[agent] = int(atn)

        obs, rewards, dones, truncateds, infos = self._inner.step(unpacked)

        self.rewards[:] = 0.0
        self.terminals[:] = True
        self.truncations[:] = False

        for i, agent in enumerate(self.possible_agents):
            if agent not in obs:
                self.observations[i] = 0
                self.rewards[i] = 0.0
                self.terminals[i] = True
                self.truncations[i] = False
                self.masks[i] = False
                continue

            self._pack_observation_row(i, obs[agent])
            self.rewards[i] = float(rewards[agent])
            self.terminals[i] = bool(dones[agent])
            self.truncations[i] = bool(truncateds[agent])
            self.masks[i] = True

        self._all_done = all(dones.values()) or all(truncateds.values())

        info_list: List[Dict[str, Any]] = []
        for agent in self.possible_agents:
            if agent in infos:
                info_list.append(infos[agent])
            else:
                info_list.append({})

        return self.observations, self.rewards, self.terminals, self.truncations, info_list

    def close(self):
        pass
