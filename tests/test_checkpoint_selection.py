from __future__ import annotations

import pytest

from catanrl.eval.checkpoint_selection import (
    numeric_policy_selectors,
    rank_checkpoint_summaries,
)


class _Registry:
    checkpoints = [
        {"step": 10, "role": "policy", "file": "checkpoints/policy_value_iter_10.pt"},
        {"step": 5, "role": "policy", "file": "checkpoints/policy_value_iter_5.pt"},
        {"step": 10, "role": "critic", "file": "checkpoints/critic_iter_10.pt"},
        {"step": 10, "role": "policy", "file": "checkpoints/policy_value_iter_10.pt"},
    ]


def test_numeric_policy_selectors_are_unique_and_sorted():
    assert numeric_policy_selectors(_Registry()) == [5, 10]


def test_rank_checkpoint_summaries_prefers_wins_then_vps_then_recency():
    sweeps = {
        "5": {"summary": {"games": 20, "wins": 12, "mean_vps": 11.0}},
        "10": {"summary": {"games": 20, "wins": 12, "mean_vps": 11.2}},
        "15": {"summary": {"games": 20, "wins": 12, "mean_vps": 11.2}},
        "20": {"summary": {"games": 20, "wins": 11, "mean_vps": 12.0}},
    }

    ranking = rank_checkpoint_summaries(sweeps, top_k=3)

    assert [row["selector"] for row in ranking] == ["15", "10", "5"]


def test_rank_checkpoint_summaries_rejects_invalid_counts():
    with pytest.raises(ValueError, match="invalid game counts"):
        rank_checkpoint_summaries(
            {"5": {"summary": {"games": 10, "wins": 11}}},
            top_k=1,
        )
