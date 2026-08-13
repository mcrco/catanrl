from __future__ import annotations

import pytest

from catanrl.eval.canopy_parity_comparison import (
    compare_canopy_to_native,
    paired_win_rate_interval,
)


def _game(seat: str, seed: int, win: bool) -> dict[str, object]:
    return {"seat": seat, "episode_seed": seed, "win": win}


def _payloads() -> tuple[dict, dict]:
    canopy_games = [
        _game("first", 10, True),
        _game("first", 11, False),
        _game("second", 10, False),
        _game("second", 11, False),
    ]
    candidate_games = [
        _game("first", 10, True),
        _game("first", 11, True),
        _game("second", 10, False),
        _game("second", 11, True),
    ]
    canopy = {
        "implementation": "cullback/canopy adapted into Catanatron",
        "game_engine": "Catanatron",
        "map_layout_source": "cppanatron layout imported into Catanatron",
        "opponent": "F",
        "map_type": "BASE",
        "number_placement": "random",
        "num_players": 2,
        "vps_to_win": 15,
        "discard_limit": 9,
        "simulations": 1600,
        "root_noise": 0.0,
        "games_per_seat": 2,
        "seed": 12043,
        "max_actions": 2000,
        "checkpoint_name": "model_iter_315.mpk",
        "game_records": canopy_games,
    }
    native = {
        "config": {
            "experiment": "candidate",
            "checkpoint": "7",
            "game_opponent": "value",
            "authoritative_engine": "catanatron",
            "map_layout_source": "cppanatron layout imported into Catanatron",
            "map_type": "BASE",
            "vps_to_win": 15,
            "discard_limit": 9,
            "games_per_seat": 2,
            "seed": 12043,
            "max_actions": 2000,
            "root_dirichlet_fraction": 0.0,
        },
        "game_sweeps": {"1600": {"game_records": candidate_games}},
    }
    return canopy, native


def test_compare_canopy_to_native_validates_and_pairs_contract() -> None:
    canopy, native = _payloads()

    result = compare_canopy_to_native(canopy, native)

    assert result["candidate"]["wins"] == 3
    assert result["canopy"]["wins"] == 1
    assert result["paired"]["overall"]["a_only_win"] == 2
    assert result["paired"]["overall"]["b_only_win"] == 0
    assert result["win_rate_difference_interval"]["difference"] == pytest.approx(0.5)


def test_compare_canopy_to_native_rejects_contract_drift() -> None:
    canopy, native = _payloads()
    native["config"]["discard_limit"] = 7

    with pytest.raises(ValueError, match="discard threshold"):
        compare_canopy_to_native(canopy, native)


def test_compare_canopy_to_native_rejects_native_authority() -> None:
    canopy, native = _payloads()
    native["config"]["authoritative_engine"] = "native"

    with pytest.raises(ValueError, match="Candidate games were not governed by Catanatron"):
        compare_canopy_to_native(canopy, native)


def test_paired_interval_rejects_unmatched_seeds() -> None:
    with pytest.raises(ValueError, match="Paired game keys differ"):
        paired_win_rate_interval(
            [_game("first", 1, True), _game("first", 2, False)],
            [_game("first", 1, False), _game("first", 3, True)],
        )
