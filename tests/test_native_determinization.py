"""Exact 1v1 dev-card belief and determinization on cppanatron games."""

from __future__ import annotations

import numpy as np
import pytest

from catanrl.beliefs.native_dev_card_belief import (
    determinize,
    development_deck_counts,
    opponent_belief,
    sample_hypotheses,
    unseen_pool,
)
from catanrl.envs.cppanatron import FlatGame
from catanrl.eval.batched_native_mcts import run_batched_native_games


def _position_with_hidden_cards(seed: int = 11, hand: int = 2) -> FlatGame:
    """Random play until the player not to move holds at least ``hand`` dev cards."""
    for attempt in range(20):
        game = FlatGame(2, "BASE", seed=seed + attempt, map_seed=seed + attempt, vps_to_win=15, discard_limit=9)
        rng = np.random.default_rng(seed + attempt)
        for _ in range(4000):
            if game.winner is not None:
                break
            legal = np.flatnonzero(game.action_mask())
            opponent = 1 - game.current_player
            if legal.size > 1 and sum(game.player(opponent).development_cards) >= hand:
                return game
            game.step(int(rng.choice(legal)))
    raise AssertionError("no position with hidden development cards found")


def test_belief_is_normalized_and_covers_the_true_hand() -> None:
    game = _position_with_hidden_cards()
    me = game.current_player
    them = game.player(1 - me)
    hypotheses = opponent_belief(game.game, me)
    pool = unseen_pool(game.game, me)
    assert sum(h.weight for h in hypotheses) == pytest.approx(1.0)
    for hypothesis in hypotheses:
        assert sum(hypothesis.cards) == sum(them.development_cards)
        assert all(take <= available for take, available in zip(hypothesis.cards, pool))
        assert them.victory_points + hypothesis.cards[4] < game.game.vps_to_win
    assert tuple(them.development_cards) in {h.cards for h in hypotheses}


def test_determinized_games_stay_consistent() -> None:
    game = _position_with_hidden_cards()
    me = game.current_player
    opponent = 1 - me
    before_me = game.player(me)
    public_vps = game.player(opponent).victory_points
    pool = unseen_pool(game.game, me)
    remaining = game.game.development_cards_remaining
    for hypothesis in opponent_belief(game.game, me)[:10]:
        world = determinize(game.game, me, hypothesis.cards, seed=3)
        them = world.player(opponent)
        assert tuple(them.development_cards) == hypothesis.cards
        assert them.actual_victory_points == public_vps + hypothesis.cards[4]
        assert world.development_cards_remaining == remaining
        # Deck = unseen pool minus the hypothesized hand, so draws in search
        # carry no trace of the true hand.
        assert development_deck_counts(world) == [p - c for p, c in zip(pool, hypothesis.cards)]
        assert world.player(me).development_cards == before_me.development_cards
        assert world.player(me).resources == before_me.resources


def test_determinization_rejects_inconsistent_hands() -> None:
    game = _position_with_hidden_cards()
    me = game.current_player
    size = sum(game.player(1 - me).development_cards)
    with pytest.raises(ValueError):
        game.game.copy().determinize_development_cards(1 - me, [size + 1, 0, 0, 0, 0], 0)


def test_samples_merge_into_multiplicities() -> None:
    game = _position_with_hidden_cards()
    draws = sample_hypotheses(game.game, game.current_player, 16, np.random.default_rng(0))
    assert sum(count for _, count in draws) == 16
    assert len({cards for cards, _ in draws}) == len(draws)


def test_batched_driver_runs_determinized_searches() -> None:
    def evaluate(observations: np.ndarray):
        return np.zeros((len(observations), 313), dtype=np.float32), np.zeros(len(observations))

    settings = {
        "map_type": "BASE",
        "number_placement": "official_spiral",
        "vps_to_win": 15,
        "discard_limit": 9,
        "turns_limit": 80,
    }
    records, stats, _ = run_batched_native_games(
        evaluate=evaluate,
        scenarios=[(0, 1), (1, 2)],
        simulations=8,
        c_puct=1.5,
        settings=settings,
        concurrency=2,
        determinizations=4,
    )
    assert len(records) == 2
    assert stats.searched > 0
    assert stats.evaluations >= stats.searched
