"""Agent specs and stats of the paired native search matchups."""

from __future__ import annotations

import pytest

from catanrl.eval.search_matchups import AgentStats, parse_agent


def test_agent_specs_parse_kinds_and_parameters() -> None:
    assert parse_agent("F").kind == "value"
    assert parse_agent("raw").params == {}
    spec = parse_agent("mcts:sims=64, c_puct=2.5,dets=8")
    assert spec.kind == "mcts"
    assert spec.params == {"sims": "64", "c_puct": "2.5", "dets": "8"}
    assert spec.label == "mcts:sims=64, c_puct=2.5,dets=8"


@pytest.mark.parametrize("text", ["alphabeta", "mcts:sims"])
def test_agent_specs_reject_unknown_kinds_and_bare_parameters(text: str) -> None:
    with pytest.raises(ValueError):
        parse_agent(text)


def test_agent_stats_sum_counts_and_keep_the_deepest_search() -> None:
    total = AgentStats(decisions=3, searched=2, evaluations=100, seconds=1.0, max_depth=7)
    total.add(AgentStats(decisions=1, searched=2, evaluations=60, max_depth=4))
    assert (total.decisions, total.searched, total.evaluations) == (4, 4, 160)
    assert total.max_depth == 7
    summary = total.summary("a")
    assert summary["a/evals_per_search"] == pytest.approx(40.0)
    assert summary["a/max_depth"] == 7.0
    assert "a/raw_disagreement" not in summary
