"""Sanity checks for DAgger training data plumbing."""

from __future__ import annotations

import numpy as np
from pufferlib.emulation import nativize

from catanrl.algorithms.imitation_learning.dataset import AggregatedDataset
from catanrl.envs.gym.rewards import WinReward
from catanrl.envs.puffer.common import create_expert, create_opponents
from catanrl.envs.puffer.single_agent_env import SingleAgentCatanatronPufferEnv
from catanrl.envs.puffer.rollout_utils import flatten_puffer_observation, get_action_mask_from_obs
from catanrl.features.catanatron_utils import (
    full_game_to_features,
    game_to_features,
    get_actor_indices_from_full,
)
from catanrl.utils.catanatron_action_space import to_action_space


def _make_debug_env(actor_observation_level: str = "private") -> SingleAgentCatanatronPufferEnv:
    return SingleAgentCatanatronPufferEnv(
        config={
            "map_type": "BASE",
            "vps_to_win": 10,
            "discard_limit": 7,
            "enemies": create_opponents(["F"]),
            "reward_function": WinReward(),
            "shared_critic": True,
            "expert_player": create_expert("F"),
            "actor_observation_level": actor_observation_level,
        }
    )


def _decode_observation(env: SingleAgentCatanatronPufferEnv, observation: np.ndarray) -> dict:
    return nativize(observation[0], env.env_single_observation_space, env.obs_dtype)


def _deep_copy_dict(value: dict) -> dict:
    copied: dict = {}
    for key, item in value.items():
        if isinstance(item, dict):
            copied[key] = _deep_copy_dict(item)
        elif isinstance(item, np.ndarray):
            copied[key] = item.copy()
        elif isinstance(item, list):
            copied[key] = list(item)
        else:
            copied[key] = item
    return copied


def _snapshot_state(env: SingleAgentCatanatronPufferEnv, observation: np.ndarray, info: dict) -> dict:
    structured_observation = _decode_observation(env, observation)
    return {
        "observation": _deep_copy_dict(structured_observation),
        "info": _deep_copy_dict(info),
        "game": env.game.copy(),
        "num_players": env.num_players,
        "map_type": env.map_type,
        "p0_color": env.p0.color,
        "action_space_size": env.action_space_size,
        "critic_dim": env.critic_vector_dim,
        "actor_observation_level": env.actor_observation_level,
    }


def _collect_structured_states(
    num_states: int = 4,
) -> tuple[SingleAgentCatanatronPufferEnv, list[dict]]:
    env = _make_debug_env()
    states: list[dict] = []
    try:
        observation, infos = env.reset(seed=123)
        info = infos[0]
        for _ in range(num_states):
            states.append(_snapshot_state(env, observation, info))
            expert_action = int(info["expert_action"])
            observation, _, terminated, truncated, infos = env.step(
                np.asarray([expert_action], dtype=np.int32)
            )
            info = infos[0]
            if bool(terminated[0] or truncated[0]):
                observation, infos = env.reset()
                info = infos[0]
    except Exception:
        env.close()
        raise
    return env, states


def test_actor_and_critic_features_match_runtime_helpers():
    env, states = _collect_structured_states()
    try:
        dataset = AggregatedDataset(
            critic_dim=states[0]["critic_dim"],
            num_players=states[0]["num_players"],
            map_type=states[0]["map_type"],
            max_size=8,
        )

        for state in states:
            observation = state["observation"]
            actor_vec, critic_vec = flatten_puffer_observation(observation)
            expected_actor = game_to_features(
                state["game"],
                state["p0_color"],
                state["num_players"],
                state["map_type"],
            )
            expected_critic = full_game_to_features(
                state["game"],
                state["num_players"],
                state["map_type"],
            )

            assert np.array_equal(actor_vec, expected_actor)
            assert np.array_equal(critic_vec, expected_critic)
            assert np.array_equal(critic_vec[dataset.actor_indices], expected_actor)
    finally:
        env.close()


def test_public_actor_features_derive_from_critic_indices():
    env = _make_debug_env(actor_observation_level="public")
    try:
        observation, infos = env.reset(seed=123)
        state = _snapshot_state(env, observation, infos[0])
        actor_vec, critic_vec = flatten_puffer_observation(state["observation"])
        actor_indices = get_actor_indices_from_full(
            state["num_players"],
            state["map_type"],
            level="public",
        )

        dataset = AggregatedDataset(
            critic_dim=state["critic_dim"],
            num_players=state["num_players"],
            map_type=state["map_type"],
            actor_observation_level="public",
            max_size=8,
        )

        assert np.array_equal(actor_vec, critic_vec[actor_indices])
        assert np.array_equal(dataset.actor_indices, actor_indices)
    finally:
        env.close()


def test_action_mask_matches_valid_actions():
    env, states = _collect_structured_states()
    try:
        for state in states:
            observation = state["observation"]
            info = state["info"]
            mask = get_action_mask_from_obs(observation)
            expected_mask = np.zeros(state["action_space_size"], dtype=np.bool_)
            expected_mask[info["valid_actions"]] = True
            assert np.array_equal(mask, expected_mask)
    finally:
        env.close()


def test_expert_action_index_matches_expert_decision():
    env, states = _collect_structured_states()
    try:
        for state in states:
            info = state["info"]
            game = state["game"]
            expert_player = create_expert("F")
            expert_action = expert_player.decide(game, game.playable_actions)
            expected_idx = to_action_space(
                expert_action,
                state["num_players"],
                state["map_type"],
                tuple(game.state.colors),
            )
            assert int(info["expert_action"]) == expected_idx
    finally:
        env.close()
