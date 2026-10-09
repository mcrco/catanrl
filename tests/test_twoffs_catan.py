"""2FFS on cppanatron positions: chance outcomes, rollouts, budgets."""

from __future__ import annotations

import numpy as np
import pytest

import cppanatron as cpp
from catanrl.algorithms.twoffs.catan import (
    CatanOracle,
    CatanTwoFFSConfig,
    chance_outcomes,
    sample_chance,
    search_catan,
)
from catanrl.beliefs.native_dev_card_belief import development_deck_counts
from catanrl.envs.cppanatron import FlatGame
from catanrl.eval.search_matchups import parse_agent, twoffs_config_from_params


class _UniformEvaluator:
    """Uniform policy, value favouring whoever has more victory points."""

    def __init__(self) -> None:
        self.rows = 0

    def evaluate(self, observations: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self.rows += len(observations)
        logits = np.zeros((len(observations), 313), dtype=np.float32)
        return logits, np.zeros(len(observations), dtype=np.float32)


def _random_game(seed: int = 7) -> FlatGame:
    return FlatGame(2, "BASE", seed=seed, map_seed=seed, vps_to_win=15, discard_limit=9)


def _advance_until(game: FlatGame, predicate, rng: np.random.Generator, limit: int = 20_000):
    for _ in range(limit):
        if game.winner is not None:
            return None
        for action in game.game.playable_actions:
            if predicate(game, action):
                return action
        legal = np.flatnonzero(game.action_mask())
        game.step(int(rng.choice(legal)))
    return None


def test_roll_outcomes_follow_dice_totals() -> None:
    game = _random_game()
    rng = np.random.default_rng(0)
    roll = _advance_until(game, lambda g, a: a.action_type == cpp.ActionType.ROLL, rng)
    outcomes = chance_outcomes(game.game, roll)
    assert len(outcomes) == 11
    assert sum(p for _, p in outcomes) == pytest.approx(1.0)
    totals = {sum(kwargs["dice"]): p for kwargs, p in outcomes}
    assert totals[7] == pytest.approx(6 / 36)
    assert totals[2] == pytest.approx(1 / 36)


def test_development_draws_follow_the_remaining_deck() -> None:
    game = _random_game(11)
    rng = np.random.default_rng(1)
    buy = _advance_until(
        game, lambda g, a: a.action_type == cpp.ActionType.BUY_DEVELOPMENT_CARD, rng
    )
    assert buy is not None
    counts = development_deck_counts(game.game)
    assert sum(counts) == game.game.development_cards_remaining
    outcomes = chance_outcomes(game.game, buy)
    assert sum(p for _, p in outcomes) == pytest.approx(1.0)
    for kwargs, probability in outcomes:
        index = list(cpp.DevelopmentCard.__members__.values()).index(kwargs["development_card"])
        assert probability == pytest.approx(counts[index] / sum(counts))


def test_robber_steals_follow_the_victims_hand() -> None:
    game = _random_game(3)
    rng = np.random.default_rng(2)

    def stealing(g, action):
        if action.action_type != cpp.ActionType.MOVE_ROBBER or action.value[1] is None:
            return False
        return sum(g.game.player(action.value[1]).resources) > 1

    steal = _advance_until(game, stealing, rng)
    assert steal is not None
    hand = game.game.player(steal.value[1]).resources
    outcomes = chance_outcomes(game.game, steal)
    assert sum(p for _, p in outcomes) == pytest.approx(1.0)
    assert len(outcomes) == sum(count > 0 for count in hand)
    for kwargs, probability in outcomes:
        index = list(cpp.Resource.__members__.values()).index(kwargs["stolen_resource"])
        assert probability == pytest.approx(hand[index] / sum(hand))


def test_sampled_rolls_do_not_replay_the_engine_rng() -> None:
    game = _random_game()
    rng = np.random.default_rng(0)
    roll = _advance_until(game, lambda g, a: a.action_type == cpp.ActionType.ROLL, rng)
    totals = set()
    for seed in range(30):
        totals.add(sum(sample_chance(game.game, roll, np.random.default_rng(seed))["dice"]))
    assert len(totals) > 3


def test_rollouts_reach_terminal_values() -> None:
    game = _random_game(5)
    rng = np.random.default_rng(4)
    _advance_until(game, lambda g, a: False, rng, limit=150)
    oracle = CatanOracle(
        game.game.copy(), game.space, _UniformEvaluator(), CatanTwoFFSConfig(), np.random.default_rng(0)
    )
    values = oracle.rollout([game.game.copy() for _ in range(4)])
    assert all(value in (-1.0, 0.0, 1.0) for value in values)
    assert oracle.network_evaluations > 0


def test_search_returns_a_legal_action_within_budget() -> None:
    game = _random_game(9)
    rng = np.random.default_rng(5)
    for _ in range(400):
        legal = np.flatnonzero(game.action_mask())
        if game.num_turns > 20 and legal.size >= 3:
            break
        game.step(int(rng.choice(legal)))
    legal = np.flatnonzero(game.action_mask())
    assert legal.size >= 2
    evaluator = _UniformEvaluator()
    config = CatanTwoFFSConfig(depth=3, budget=200, fast_bias_leaf=0.2, fast_bias_slope=0.1)
    before = game.game.copy()
    outcome = search_catan(game.game, game.space, evaluator, config, np.random.default_rng(0))
    assert outcome.action in set(legal.tolist())
    assert game.game.search_equivalent(before)
    # The root expansion is unconditional; later expansions respect the cap.
    assert outcome.result.total_cost <= max(200, outcome.result.num_fast_queries)
    assert outcome.network_evaluations == evaluator.rows


def test_matchup_specs_build_twoffs_configs() -> None:
    spec = parse_agent("2ffs:budget=512,depth=2,leaf=0.3,slope=0.13,batch=4,fallback=prior")
    assert spec.kind == "2ffs"
    config = twoffs_config_from_params(spec.params)
    assert (config.budget, config.depth, config.slow_batch) == (512.0, 2, 4)
    assert (config.fast_bias_leaf, config.fast_bias_slope) == (0.3, 0.13)
    assert config.fallback == "prior"
    assert config.rollout_decisions is None
    assert twoffs_config_from_params({"rollout": "40"}).rollout_decisions == 40
    with pytest.raises(ValueError):
        twoffs_config_from_params({"width": "3"})
