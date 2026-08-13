from __future__ import annotations

import pytest

from catanrl.eval.canopy_checkpoint_selection import (
    rank_direct_canopy_results,
    shortlist_from_search_screen,
)


def _records(wins: list[bool]):
    return [
        {"seat": "first" if index % 2 == 0 else "second", "episode_seed": index, "win": win}
        for index, win in enumerate(wins)
    ]


def test_shortlist_uses_completed_search_ranking() -> None:
    payload = {"ranking": [{"selector": 9}, {"selector": "7"}, {"selector": 3}]}
    assert shortlist_from_search_screen(payload, top_k=2) == ["9", "7"]


def test_shortlist_rejects_incomplete_ranking() -> None:
    with pytest.raises(ValueError, match="only 1"):
        shortlist_from_search_screen({"ranking": [{"selector": 9}]}, top_k=2)


def test_direct_ranking_uses_canopy_wins_then_vps() -> None:
    sweeps = {
        "7": {
            "summary": {"wins": 2, "games": 4, "win_rate": 0.5, "mean_vps": 11.0},
            "game_records": _records([True, False, True, False]),
        },
        "9": {
            "summary": {"wins": 3, "games": 4, "win_rate": 0.75, "mean_vps": 10.0},
            "game_records": _records([True, True, True, False]),
        },
        "10": {
            "summary": {"wins": 2, "games": 4, "win_rate": 0.5, "mean_vps": 12.0},
            "game_records": _records([False, True, True, False]),
        },
    }
    ranking, paired = rank_direct_canopy_results(sweeps)
    assert [row["selector"] for row in ranking] == ["9", "10", "7"]
    assert set(paired) == {"7", "10"}
