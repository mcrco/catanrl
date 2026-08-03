from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn
from typing import cast

from catanrl.algorithms.alphazero.native_search import (
    run_native_search_policy,
    step_game_and_reconcile_search,
)
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


def _run_search(seed: int):
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
            return search.root_visits(), np.asarray(leaf_players), search.metrics()
    finally:
        game.close()


def test_native_mcts_search_is_deterministic_and_visits_legal_actions():
    first_visits, first_players, first_metrics = _run_search(1234)
    second_visits, second_players, second_metrics = _run_search(1234)

    np.testing.assert_array_equal(first_visits, second_visits)
    np.testing.assert_array_equal(first_players, second_players)
    assert first_metrics == second_metrics
    assert int(first_visits.sum()) == 24
    assert first_metrics.simulations == 24
    assert 1 <= first_metrics.principal_variation_depth <= first_metrics.maximum_depth
    assert 1.0 <= first_metrics.mean_depth <= first_metrics.maximum_depth
    assert first_metrics.root_value == pytest.approx(0.0)

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


def test_native_mcts_reuses_a_deterministic_played_subtree():
    game = NativeGame(2, "MINI", seed=59, map_seed=61)
    logits = np.zeros(game.action_space_size, dtype=np.float32)
    backend = MockInferenceBackend(game.action_space_size, value=0.4)
    actor_indices = get_observation_indices_from_full(2, "MINI", "private")
    critic_indices = get_observation_indices_from_full(2, "MINI", "full")
    try:
        with NativeMCTSSearch(
            game,
            "MINI",
            seed=67,
            canonical_pruning=True,
        ) as search:
            search.initialize_root(logits)
            for _ in range(12):
                leaf = search.select_leaf()
                if leaf is not None:
                    search.evaluate_leaf(logits, 0.2)
            action = int(np.argmax(search.root_visits()))
            retained_search = step_game_and_reconcile_search(
                game=game,
                map_type="MINI",
                action=action,
                search=search,
            )
            assert retained_search is search
            root_observation, root_player = search.root_observation()
            assert root_player == game.current_player
            np.testing.assert_array_equal(
                root_observation,
                full_native_features(game, "MINI", game.current_player),
            )

            result = run_native_search_policy(
                game=game,
                map_type="MINI",
                inference_backend=backend,
                actor_indices=actor_indices,
                critic_indices=critic_indices,
                num_simulations=4,
                c_puct=1.5,
                search_seed=71,
                add_noise=False,
                dirichlet_alpha=0.3,
                dirichlet_frac=0.0,
                action_temperature=0.0,
                target_temperature=1.0,
                rng=np.random.default_rng(73),
                value_scale=0.5,
                canonical_pruning=True,
                search=search,
            )
            assert result.diagnostics.tree_reused == 1.0
            assert result.diagnostics.retained_root_visits > 0
            assert result.diagnostics.simulations == 4
            assert result.diagnostics.backed_up_network_value == pytest.approx(0.2)
    finally:
        game.close()


def test_reused_search_divergence_falls_back_to_fresh_tree() -> None:
    game = NativeGame(2, "MINI", seed=79, map_seed=83)

    class _DivergedSearch:
        closed = False

        def advance(self, _action: int) -> bool:
            return True

        def root_observation(self) -> tuple[np.ndarray, int]:
            observation = full_native_features(game, "MINI", game.current_player)
            return np.zeros_like(observation), game.current_player

        def close(self) -> None:
            self.closed = True

    search = _DivergedSearch()
    action = int(np.flatnonzero(game.valid_action_mask())[0])
    try:
        reconciled = step_game_and_reconcile_search(
            game=game,
            map_type="MINI",
            action=action,
            search=cast(NativeMCTSSearch, search),
        )
    finally:
        game.close()

    assert reconciled is None
    assert search.closed


def test_native_search_policy_reports_depth_and_policy_effect():
    game = NativeGame(2, "MINI", seed=31, map_seed=37)
    action_space_size = game.action_space_size
    backend = MockInferenceBackend(
        action_space_size,
        value=0.2,
        policy_logits=np.linspace(-0.5, 0.5, action_space_size, dtype=np.float32),
    )
    actor_indices = get_observation_indices_from_full(2, "MINI", "private")
    critic_indices = get_observation_indices_from_full(2, "MINI", "full")
    try:
        result = run_native_search_policy(
            game=game,
            map_type="MINI",
            inference_backend=backend,
            actor_indices=actor_indices,
            critic_indices=critic_indices,
            num_simulations=16,
            c_puct=1.5,
            search_seed=41,
            add_noise=False,
            dirichlet_alpha=0.3,
            dirichlet_frac=0.0,
            action_temperature=0.0,
            target_temperature=1.0,
            rng=np.random.default_rng(43),
        )
    finally:
        game.close()

    diagnostics = result.diagnostics
    assert diagnostics.simulations == 16
    assert 1 <= diagnostics.principal_variation_depth <= diagnostics.maximum_depth
    assert 1.0 <= diagnostics.mean_depth <= diagnostics.maximum_depth
    assert diagnostics.legal_actions > 1
    assert diagnostics.top1_agreement in (0.0, 1.0)
    assert diagnostics.policy_kl >= 0.0
    assert diagnostics.policy_js >= 0.0
    assert np.isfinite(diagnostics.value_shift)
    assert diagnostics.elapsed_seconds > 0.0
    assert result.policy.sum() == pytest.approx(1.0)


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
        "tree_reuse": True,
        "canonical_pruning": True,
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
    for sample in samples:
        assert sample.actor_state.shape == (actor_size,)
        assert sample.critic_state.shape == (critic_size,)
        assert sample.policy.shape == sample.action_mask.shape == (action_space_size,)
        assert sample.action_mask.dtype == np.bool_
        assert np.count_nonzero(sample.action_mask) > 1
        assert float(sample.policy.sum()) == pytest.approx(1.0)
        assert np.all(sample.policy[~sample.action_mask] == 0.0)
        assert sample.player in (0, 1)


def test_terminal_search_value_blend_stays_on_win_loss_scale():
    from catanrl.algorithms.alphazero.native_self_play import (
        _blend_terminal_search_value,
    )

    assert _blend_terminal_search_value(1.0, -0.5, 0.0) == 1.0
    assert _blend_terminal_search_value(1.0, -0.5, 0.25) == pytest.approx(0.625)
    assert _blend_terminal_search_value(-1.0, 2.0, 0.25) == pytest.approx(-0.5)


def test_native_playout_cap_keeps_fast_decisions_for_value_training():
    action_space_size = 187
    backend = MockInferenceBackend(action_space_size)
    args = {
        "map_type": "MINI",
        "num_players": 2,
        "num_simulations": 2,
        "fast_simulations": 1,
        "full_search_probability": 0.0,
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
    assert all(not sample.full_search for sample in samples)


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
    assert all(exp.full_search for exp in experiences)
    assert all(exp.policy.shape == (action_space_size,) for exp in experiences)
    assert all(np.all(exp.policy[~exp.action_mask] == 0.0) for exp in experiences)
