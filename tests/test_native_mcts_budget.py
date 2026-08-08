from __future__ import annotations

import numpy as np
import pytest

from catanrl.algorithms.alphazero.native_search import NativeSearchDiagnostics
from catanrl.eval.native_mcts_budget import (
    CriticCalibrationAccumulator,
    NativeBudgetGameResult,
    SearchDiagnosticsAccumulator,
    _game_opponent_action,
    _game_worker_main,
    _search,
    run_native_budget_games,
)


def _diagnostics(*, depth: int, agreement: float) -> NativeSearchDiagnostics:
    return NativeSearchDiagnostics(
        simulations=96,
        principal_variation_depth=depth,
        maximum_depth=depth + 2,
        mean_depth=depth + 0.5,
        legal_actions=8,
        prior_top1_action=3,
        search_top1_action=3 if agreement else 4,
        top1_agreement=agreement,
        policy_kl=0.2,
        policy_js=0.05,
        prior_entropy=1.5,
        search_entropy=1.2,
        network_value=0.1,
        search_value=0.3,
        value_shift=0.2,
        value_scale=1.0,
        backed_up_network_value=0.1,
        retained_root_visits=0,
        tree_reused=0.0,
        pruned_actions=0,
        coalesced_outcomes=0,
        neural_evaluations=97,
        elapsed_seconds=0.4,
    )


def test_search_diagnostics_accumulator_merges_weighted_summaries():
    first = SearchDiagnosticsAccumulator()
    first.add(_diagnostics(depth=2, agreement=1.0))
    second = SearchDiagnosticsAccumulator()
    second.add(_diagnostics(depth=4, agreement=0.0))

    first.merge(second.payload())
    summary = first.summary()

    assert summary["searches"] == 2.0
    assert summary["mean_simulations"] == 96.0
    assert summary["mean_principal_variation_depth"] == 3.0
    assert summary["mean_top1_agreement"] == 0.5
    assert summary["maximum_observed_depth"] == 6.0
    assert summary["simulations_per_second"] == pytest.approx(240.0)


def test_critic_calibration_reports_scale_and_affine_fit():
    calibration = CriticCalibrationAccumulator()
    calibration.add(0.25, 1.0)
    calibration.add(-0.25, -1.0)

    summary = calibration.summary()

    assert summary["mse"] == pytest.approx(0.5625)
    assert summary["correlation"] == pytest.approx(1.0)
    assert summary["least_squares_scale"] == pytest.approx(4.0)
    assert summary["affine_scale"] == pytest.approx(4.0)
    assert summary["affine_bias"] == pytest.approx(0.0)


def test_native_budget_summary_includes_win_rate_interval():
    result = NativeBudgetGameResult(
        game_records=[
            {"win": win, "seat": "first", "vps": 15, "turns": 100, "draw": False}
            for win in (True, True, False, False)
        ]
    )

    summary = result.summary()

    assert summary["win_rate"] == 0.5
    assert summary["win_rate_ci95_low"] < 0.5
    assert summary["win_rate_ci95_high"] > 0.5


class _ValueOpponentGame:
    def __init__(self) -> None:
        self.calls = 0

    def value_action(self) -> int:
        self.calls += 1
        return 17


def test_native_value_opponent_uses_cpp_value_player_without_inference():
    game = _ValueOpponentGame()

    action = _game_opponent_action(
        game,  # type: ignore[arg-type]
        "value",
        "BASE",
        pytest.fail,  # type: ignore[arg-type]
        pytest.fail,  # type: ignore[arg-type]
        pytest.fail,  # type: ignore[arg-type]
    )

    assert action == 17
    assert game.calls == 1


class _RandomOpponentGame:
    def valid_action_mask(self):
        return np.array([False, True, False, True, False], dtype=np.bool_)


def test_native_random_opponent_selects_only_legal_actions_and_is_deterministic():
    game = _RandomOpponentGame()
    first_rng = np.random.default_rng(1977)
    second_rng = np.random.default_rng(1977)

    first = [
        _game_opponent_action(
            game,  # type: ignore[arg-type]
            "random",
            "BASE",
            pytest.fail,  # type: ignore[arg-type]
            pytest.fail,  # type: ignore[arg-type]
            pytest.fail,  # type: ignore[arg-type]
            first_rng,
        )
        for _ in range(100)
    ]
    second = [
        _game_opponent_action(
            game,  # type: ignore[arg-type]
            "random",
            "BASE",
            pytest.fail,  # type: ignore[arg-type]
            pytest.fail,  # type: ignore[arg-type]
            pytest.fail,  # type: ignore[arg-type]
            second_rng,
        )
        for _ in range(100)
    ]

    assert first == second
    assert set(first) == {1, 3}


def test_native_random_opponent_requires_rng():
    with pytest.raises(ValueError, match="random generator"):
        _game_opponent_action(
            _RandomOpponentGame(),  # type: ignore[arg-type]
            "random",
            "BASE",
            pytest.fail,  # type: ignore[arg-type]
            pytest.fail,  # type: ignore[arg-type]
            pytest.fail,  # type: ignore[arg-type]
        )


def test_native_budget_search_can_match_canopy_root_noise(monkeypatch):
    captured = {}
    sentinel = object()

    def _run_native_search_policy(**kwargs):
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(
        "catanrl.eval.native_mcts_budget.run_native_search_policy",
        _run_native_search_policy,
    )

    result = _search(
        game=object(),  # type: ignore[arg-type]
        budget=1600,
        decision_index=7,
        episode_seed=47,
        args_dict={
            "map_type": "BASE",
            "c_puct": 2.5,
            "value_scale": 1.0,
            "canonical_pruning": True,
            "search_selection": "completed-q",
            "c_visit": 50.0,
            "c_scale": 1.0,
            "root_dirichlet_alpha": 0.05,
            "root_dirichlet_fraction": 0.25,
        },
        actor_indices=np.array([0]),
        critic_indices=np.array([0]),
        inference_backend=object(),  # type: ignore[arg-type]
    )

    assert result is sentinel
    assert captured["add_noise"] is True
    assert captured["dirichlet_alpha"] == pytest.approx(0.05)
    assert captured["dirichlet_frac"] == pytest.approx(0.25)


@pytest.mark.parametrize("fraction", [-0.01, 1.01, float("nan")])
def test_native_budget_games_rejects_invalid_root_noise_fraction(fraction):
    with pytest.raises(ValueError, match="root_dirichlet_fraction"):
        run_native_budget_games(
            policy_model=None,
            critic_model=None,
            model_type="flat",
            map_type="BASE",
            actor_observation_level="private",
            critic_observation_level="full",
            budget=1,
            games_per_seat=1,
            num_workers=1,
            inference_batch_size=1,
            inference_wait_ms=1.0,
            c_puct=2.5,
            seed=1,
            vps_to_win=15,
            discard_limit=9,
            device="cpu",
            root_dirichlet_fraction=fraction,
        )


def test_native_budget_worker_streams_each_game(monkeypatch):
    class _Backend:
        closed = False

        def __init__(self, **_kwargs) -> None:
            pass

        def close(self) -> None:
            self.closed = True

    class _Queue:
        def __init__(self) -> None:
            self.messages = []

        def put(self, message) -> None:
            self.messages.append(message)

    def _play_budget_game(**kwargs):
        diagnostics = SearchDiagnosticsAccumulator()
        diagnostics.add(_diagnostics(depth=2, agreement=1.0))
        calibration = CriticCalibrationAccumulator()
        calibration.add(0.25, 1.0)
        return (
            {
                "win": True,
                "seat": "first" if kwargs["mcts_seat"] == 0 else "second",
                "episode_seed": kwargs["episode_seed"],
            },
            diagnostics,
            calibration,
        )

    monkeypatch.setattr(
        "catanrl.eval.native_mcts_budget._RemoteNNMCTSInferenceBackend",
        _Backend,
    )
    monkeypatch.setattr(
        "catanrl.eval.native_mcts_budget._indices",
        lambda _args: (object(), object()),
    )
    monkeypatch.setattr(
        "catanrl.eval.native_mcts_budget._play_budget_game",
        _play_budget_game,
    )
    result_queue = _Queue()

    _game_worker_main(
        3,
        object(),  # type: ignore[arg-type]
        object(),  # type: ignore[arg-type]
        result_queue,  # type: ignore[arg-type]
        [(0, 11), (1, 12)],
        1600,
        {},
    )

    assert [message["done"] for message in result_queue.messages] == [False, False, True]
    assert [message["games"] for message in result_queue.messages] == [1, 1, 0]
    assert [message["result"]["game_records"] for message in result_queue.messages[:2]] == [
        [{"win": True, "seat": "first", "episode_seed": 11}],
        [{"win": True, "seat": "second", "episode_seed": 12}],
    ]
    assert result_queue.messages[-1]["result"]["game_records"] == []
