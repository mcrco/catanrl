"""Sanity checks for DAgger training data plumbing."""

from __future__ import annotations

import numpy as np

from catanrl.algorithms.imitation_learning.dataset import AggregatedDataset
from catanrl.envs.gym.rewards import WinReward
from catanrl.envs.gym.single_env import SingleAgentCatanatronEnv, create_expert, create_opponents
from catanrl.envs.puffer.rollout_utils import flatten_puffer_observation, get_action_mask_from_obs
from catanrl.features.catanatron_utils import full_game_to_features, game_to_features
from catanrl.utils.catanatron_action_space import to_action_space


def _make_debug_env() -> SingleAgentCatanatronEnv:
    return SingleAgentCatanatronEnv(
        config={
            "map_type": "BASE",
            "vps_to_win": 10,
            "discard_limit": 7,
            "enemies": create_opponents(["F"]),
            "reward_function": WinReward(),
            "shared_critic": True,
            "expert_player": create_expert("F"),
        }
    )


def _snapshot_state(env: SingleAgentCatanatronEnv, observation: dict, info: dict) -> dict:
    return {
        "observation": observation,
        "info": info,
        "game": env.game.copy(),
        "num_players": env.num_players,
        "map_type": env.map_type,
        "p0_color": env.p0.color,
        "action_space_size": env.action_space_size,
        "critic_dim": env.critic_vector_dim,
    }


def _collect_structured_states(num_states: int = 4) -> tuple[SingleAgentCatanatronEnv, list[dict]]:
    env = _make_debug_env()
    states: list[dict] = []
    try:
        observation, info = env.reset(seed=123)
        for _ in range(num_states):
            states.append(_snapshot_state(env, observation, info))
            expert_action = int(info["expert_action"])
            observation, _, terminated, truncated, info = env.step(expert_action)
            if terminated or truncated:
                observation, info = env.reset()
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
