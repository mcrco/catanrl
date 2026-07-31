from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from catanrl.algorithms.alphazero.native_self_play import (
    _play_native_self_play_game,
    generate_native_self_play_data,
)
from catanrl.envs.cppanatron import (
    NativeGame,
    NativeMCTSSearch,
    find_cppanatron_library,
    full_native_features,
)
from catanrl.features.catanatron_utils import get_observation_indices_from_full
from catanrl.models.heads import FlatPolicyHead
from catanrl.models.wrappers import PolicyNetworkWrapper, ValueNetworkWrapper
from nn_mcts_helpers import MockInferenceBackend


@pytest.fixture(scope="module", autouse=True)
def _require_native_library():
    try:
        find_cppanatron_library()
    except FileNotFoundError as exc:
        pytest.skip(str(exc))


def _run_search(seed: int) -> tuple[np.ndarray, np.ndarray]:
    game = NativeGame(
        2,
        "MINI",
        seed=91,
        map_seed=73,
        number_placement="random",
        vps_to_win=6,
    )
    logits = np.linspace(-1.0, 1.0, game.action_space_size, dtype=np.float32)
    leaf_players: list[int] = []
    try:
        with NativeMCTSSearch(game, "MINI", c_puct=1.5, seed=seed) as search:
            root_observation, root_player = search.root_observation()
            assert root_player == game.current_player
            np.testing.assert_array_equal(
                root_observation,
                full_native_features(game, "MINI", game.current_player),
            )
            search.initialize_root(logits)
            search.add_root_dirichlet_noise(0.3, 0.25)
            for _ in range(24):
                leaf = search.select_leaf()
                if leaf is None:
                    continue
                observation, player = leaf
                assert observation.dtype == np.float32
                assert observation.shape == (search.observation_size,)
                assert np.isfinite(observation).all()
                assert 0 <= player < game.num_players
                leaf_players.append(player)
                search.evaluate_leaf(logits, value=0.0)
            return search.root_visits(), np.asarray(leaf_players)
    finally:
        game.close()


def test_native_mcts_search_is_deterministic_and_visits_legal_actions():
    first_visits, first_players = _run_search(1234)
    second_visits, second_players = _run_search(1234)

    np.testing.assert_array_equal(first_visits, second_visits)
    np.testing.assert_array_equal(first_players, second_players)
    assert int(first_visits.sum()) == 24

    game = NativeGame(2, "MINI", seed=91, map_seed=73, number_placement="random")
    try:
        legal = game.valid_action_mask()
        assert np.all(first_visits[~legal] == 0)
    finally:
        game.close()


def test_native_mcts_rejects_wrong_policy_shape():
    game = NativeGame(2, "MINI", seed=5, map_seed=7)
    try:
        with NativeMCTSSearch(game, "MINI") as search:
            with pytest.raises(ValueError, match="Expected policy logits"):
                search.initialize_root(np.zeros(game.action_space_size - 1))
    finally:
        game.close()


def test_native_self_play_game_emits_standard_legal_training_fields():
    action_space_size = 187
    backend = MockInferenceBackend(action_space_size)
    args = {
        "map_type": "MINI",
        "num_players": 2,
        "num_simulations": 2,
        "c_puct": 1.5,
        "actor_observation_level": "private",
        "critic_observation_level": "full",
        "temperature": 1.0,
        "final_temperature": 0.1,
        "target_temperature": 1.0,
        "temperature_drop_move": 30,
        "noise_turns": 20,
        "dirichlet_alpha": 0.3,
        "dirichlet_frac": 0.25,
        "vps_to_win": 10,
        "discard_limit": 7,
        "turns_limit": 2,
    }

    samples, winner = _play_native_self_play_game(
        episode_seed=101,
        args_dict=args,
        inference_backend=backend,
    )

    assert winner is None
    assert samples
    actor_size = len(get_observation_indices_from_full(2, "MINI", "private"))
    critic_size = len(get_observation_indices_from_full(2, "MINI", "full"))
    for actor_state, critic_state, policy, action_mask, player in samples:
        assert actor_state.shape == (actor_size,)
        assert critic_state.shape == (critic_size,)
        assert policy.shape == action_mask.shape == (action_space_size,)
        assert action_mask.dtype == np.bool_
        assert np.count_nonzero(action_mask) > 1
        assert float(policy.sum()) == pytest.approx(1.0)
        assert np.all(policy[~action_mask] == 0.0)
        assert player in (0, 1)


def test_native_parallel_self_play_returns_shared_experience_type():
    torch.manual_seed(0)
    actor_size = len(get_observation_indices_from_full(2, "MINI", "private"))
    critic_size = len(get_observation_indices_from_full(2, "MINI", "full"))
    action_space_size = 187
    policy_model = PolicyNetworkWrapper(
        nn.Identity(),
        FlatPolicyHead(actor_size, action_space_size),
    )
    critic_model = ValueNetworkWrapper(
        nn.Identity(),
        nn.Sequential(nn.Linear(critic_size, 1), nn.Tanh()),
    )

    experiences, stats = generate_native_self_play_data(
        policy_model=policy_model,
        critic_model=critic_model,
        model_type="flat",
        map_type="MINI",
        num_players=2,
        num_games=1,
        num_game_workers=1,
        num_simulations=1,
        c_puct=1.5,
        prunning=False,
        actor_observation_level="private",
        critic_observation_level="full",
        ismcts_determinizations=1,
        inference_batch_size=4,
        inference_wait_ms=1.0,
        temperature=1.0,
        final_temperature=0.1,
        target_temperature=1.0,
        temperature_drop_move=30,
        noise_turns=20,
        dirichlet_alpha=0.3,
        dirichlet_frac=0.25,
        vps_to_win=10,
        discard_limit=7,
        seed=17,
        device="cpu",
        show_tqdm=False,
        turns_limit=1,
    )

    assert stats["games"] == 1
    assert experiences
    assert all(exp.policy.shape == (action_space_size,) for exp in experiences)
    assert all(np.all(exp.policy[~exp.action_mask] == 0.0) for exp in experiences)
