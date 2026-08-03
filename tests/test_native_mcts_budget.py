from __future__ import annotations

import pytest

from catanrl.algorithms.alphazero.native_search import NativeSearchDiagnostics
from catanrl.eval.native_mcts_budget import (
    CriticCalibrationAccumulator,
    NativeBudgetGameResult,
    SearchDiagnosticsAccumulator,
    _game_opponent_action,
    _game_worker_main,
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
