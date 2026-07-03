import pytest

from catanrl.eval.paired_comparison import compare_paired_payloads, exact_mcnemar


def _games(wins):
    return [{"seat": "first", "episode_seed": index, "win": win} for index, win in enumerate(wins)]


def test_exact_mcnemar_uses_only_discordant_pairs():
    games_a = _games([True] * 10 + [True] * 8 + [False] * 2 + [False] * 5)
    games_b = _games([True] * 10 + [False] * 8 + [True] * 2 + [False] * 5)

    result = exact_mcnemar(games_a, games_b)

    assert result["both_win"] == 10
    assert result["a_only_win"] == 8
    assert result["b_only_win"] == 2
    assert result["both_loss"] == 5
    assert result["p_value_exact_two_sided"] == pytest.approx(0.109375)


def test_exact_mcnemar_rejects_unmatched_game_keys():
    with pytest.raises(ValueError, match="Paired game keys differ"):
        exact_mcnemar(_games([True, False]), _games([True]))


def test_paired_payloads_require_same_scenario():
    payload_a = {"scenario": {"opponent": "F"}, "games": _games([True])}
    payload_b = {"scenario": {"opponent": "R"}, "games": _games([False])}

    with pytest.raises(ValueError, match="scenarios differ"):
        compare_paired_payloads(payload_a, payload_b)
