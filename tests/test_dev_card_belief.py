from __future__ import annotations

import math
import random
from collections import Counter

import numpy as np
import pytest

from catanatron.game import Game
from catanatron.models.enums import (
    KNIGHT,
    MONOPOLY,
    ROAD_BUILDING,
    VICTORY_POINT,
    YEAR_OF_PLENTY,
)
from catanatron.models.player import Color, RandomPlayer
from catanatron.state_functions import player_key

from catanrl.beliefs import DevCardBelief, determinize_game, patch_full_features
from catanrl.beliefs.dev_card_belief import _multivariate_hypergeometric_weight
from catanrl.features.catanatron_utils import (
    full_game_to_features,
    get_full_numeric_feature_names,
)
from catanrl.utils.catanatron_action_space import get_player_colors
from catanrl.utils.catanatron_game import force_player_order
from catanrl.utils.catanatron_map import build_catan_map


def make_1v1_game() -> Game:
    players = [RandomPlayer(Color.BLUE), RandomPlayer(Color.RED)]
    game = Game(players, catan_map=build_catan_map("BASE"))
    force_player_order(game, players)
    return game


def make_3p_game() -> Game:
    colors = list(get_player_colors(3))
    players = [RandomPlayer(c) for c in colors]
    game = Game(players, catan_map=build_catan_map("BASE"))
    force_player_order(game, players)
    return game


def set_player_dev_hand(game: Game, color: Color, **counts: int) -> None:
    key = player_key(game.state, color)
    for dev in (KNIGHT, YEAR_OF_PLENTY, MONOPOLY, ROAD_BUILDING, VICTORY_POINT):
        game.state.player_state[f"{key}_{dev}_IN_HAND"] = int(counts.get(dev, 0))


def set_played(game: Game, color: Color, dev: str, n: int) -> None:
    key = player_key(game.state, color)
    game.state.player_state[f"{key}_PLAYED_{dev}"] = int(n)


def test_weights_sum_to_one():
    game = make_1v1_game()
    set_player_dev_hand(game, Color.BLUE, KNIGHT=1)
    set_player_dev_hand(game, Color.RED, KNIGHT=2)  # opponent has 2 hidden
    belief = DevCardBelief(game, Color.BLUE)
    hyps = belief.enumerate_hypotheses()
    assert hyps
    assert math.isclose(sum(h.weight for h in hyps), 1.0, rel_tol=1e-9)


def test_zero_dev_cards_single_empty_hypothesis():
    game = make_1v1_game()
    belief = DevCardBelief(game, Color.BLUE)
    hyps = belief.enumerate_hypotheses()
    assert len(hyps) == 1
    assert hyps[0].weight == pytest.approx(1.0)
    assert sum(hyps[0].opponent_dev_hands[Color.RED].values()) == 0


def test_hypergeometric_matches_hand_computation():
    # Pool: 14 knights and 5 VP cards remaining, opponent drew 1 card.
    pool = Counter({KNIGHT: 14, VICTORY_POINT: 5})
    total = 19
    p_knight = _multivariate_hypergeometric_weight(pool, Counter({KNIGHT: 1}), 1)
    p_vp = _multivariate_hypergeometric_weight(pool, Counter({VICTORY_POINT: 1}), 1)
    assert p_knight == pytest.approx(14 / total)
    assert p_vp == pytest.approx(5 / total)
    assert p_knight + p_vp == pytest.approx(1.0)


def test_not_won_pruning_excludes_winning_vp_hands():
    game = make_1v1_game()
    game.vps_to_win = 10
    # Opponent already has 9 public VPs and 1 hidden dev card. A VP card would
    # make them win, so that hypothesis must be pruned.
    key = player_key(game.state, Color.RED)
    game.state.player_state[f"{key}_VICTORY_POINTS"] = 9
    set_player_dev_hand(game, Color.RED, KNIGHT=1)  # truth is a knight
    belief = DevCardBelief(game, Color.BLUE)
    hyps = belief.enumerate_hypotheses()
    for h in hyps:
        assert h.opponent_dev_hands[Color.RED].get(VICTORY_POINT, 0) == 0
    assert math.isclose(sum(h.weight for h in hyps), 1.0, rel_tol=1e-9)


def test_belief_does_not_depend_on_opponent_truth():
    game_a = make_1v1_game()
    set_player_dev_hand(game_a, Color.RED, KNIGHT=2)
    belief_a = DevCardBelief(game_a, Color.BLUE)
    hyps_a = {
        tuple(sorted(h.opponent_dev_hands[Color.RED].items())): h.weight
        for h in belief_a.enumerate_hypotheses()
    }

    game_b = make_1v1_game()
    # Same public count (2 dev cards) but a different true composition.
    set_player_dev_hand(game_b, Color.RED, MONOPOLY=1, VICTORY_POINT=1)
    belief_b = DevCardBelief(game_b, Color.BLUE)
    hyps_b = {
        tuple(sorted(h.opponent_dev_hands[Color.RED].items())): h.weight
        for h in belief_b.enumerate_hypotheses()
    }
    assert hyps_a == hyps_b


def test_played_cards_removed_from_pool():
    game = make_1v1_game()
    # All 14 knights are accounted for publicly: opponent can't hold a knight.
    set_played(game, Color.BLUE, KNIGHT, 7)
    set_played(game, Color.RED, KNIGHT, 7)
    set_player_dev_hand(game, Color.RED, MONOPOLY=1)
    belief = DevCardBelief(game, Color.BLUE)
    assert belief.unseen_pool.get(KNIGHT, 0) == 0
    for h in belief.enumerate_hypotheses():
        assert h.opponent_dev_hands[Color.RED].get(KNIGHT, 0) == 0


def test_enumeration_matches_monte_carlo():
    game = make_1v1_game()
    set_player_dev_hand(game, Color.BLUE, KNIGHT=1, VICTORY_POINT=1)
    set_player_dev_hand(game, Color.RED, KNIGHT=2)  # 2 hidden cards
    belief = DevCardBelief(game, Color.BLUE)
    exact = {
        tuple(sorted(h.opponent_dev_hands[Color.RED].items())): h.weight
        for h in belief.enumerate_hypotheses()
    }

    rng = random.Random(0)
    counts: Counter = Counter()
    n = 40000
    for h in belief.sample_hypotheses(n, rng):
        counts[tuple(sorted(h.opponent_dev_hands[Color.RED].items()))] += 1
    for key, weight in exact.items():
        assert counts[key] / n == pytest.approx(weight, abs=0.02)


def test_sample_hypotheses_respects_not_already_won_constraint():
    # 3-player game: the Monte-Carlo path must enforce the same "opponent hasn't
    # already won" pruning as the exact 1v1 enumeration. RED sits one VP short of
    # winning, so no sampled hand may hand them a victory-point card.
    game = make_3p_game()
    game.vps_to_win = 10
    perspective, opponent, third = game.state.colors

    red_key = player_key(game.state, opponent)
    game.state.player_state[f"{red_key}_VICTORY_POINTS"] = 9
    set_player_dev_hand(game, opponent, KNIGHT=1)  # 1 hidden card, publicly known count
    set_player_dev_hand(game, third, KNIGHT=2)

    belief = DevCardBelief(game, perspective)
    assert belief.unseen_pool.get(VICTORY_POINT, 0) > 0  # VP cards are available to draw

    samples = belief.sample_hypotheses(500, random.Random(0))
    assert len(samples) == 500
    for h in samples:
        assert h.opponent_dev_hands[opponent].get(VICTORY_POINT, 0) == 0
        # Public counts must still be respected for every opponent.
        assert sum(h.opponent_dev_hands[opponent].values()) == 1
        assert sum(h.opponent_dev_hands[third].values()) == 2


def test_sample_hypotheses_marginals_match_exact_enumeration_1v1():
    # In 1v1 the sampler and the exact enumeration describe the same belief.
    game = make_1v1_game()
    set_player_dev_hand(game, Color.RED, KNIGHT=2)
    belief = DevCardBelief(game, Color.BLUE)
    exact = {
        tuple(sorted(h.opponent_dev_hands[Color.RED].items())): h.weight
        for h in belief.enumerate_hypotheses()
    }

    counts: Counter = Counter()
    n = 40000
    for h in belief.sample_hypotheses(n, random.Random(1)):
        counts[tuple(sorted(h.opponent_dev_hands[Color.RED].items()))] += 1
    for key, weight in exact.items():
        assert counts[key] / n == pytest.approx(weight, abs=0.02)


def test_determinize_game_consistency():
    game = make_1v1_game()
    set_player_dev_hand(game, Color.BLUE, KNIGHT=1)
    set_player_dev_hand(game, Color.RED, KNIGHT=3)
    key = player_key(game.state, Color.RED)
    game.state.player_state[f"{key}_VICTORY_POINTS"] = 4
    belief = DevCardBelief(game, Color.BLUE)

    hyp = next(
        h
        for h in belief.enumerate_hypotheses()
        if h.opponent_dev_hands[Color.RED].get(VICTORY_POINT, 0) >= 1
    )
    rng = random.Random(1)
    det = determinize_game(belief, hyp, rng=rng)

    hand = hyp.opponent_dev_hands[Color.RED]
    det_key = player_key(det.state, Color.RED)
    for dev in (KNIGHT, YEAR_OF_PLENTY, MONOPOLY, ROAD_BUILDING, VICTORY_POINT):
        assert det.state.player_state[f"{det_key}_{dev}_IN_HAND"] == hand.get(dev, 0)
    # ACTUAL VPs = public VPs + hidden VP cards.
    assert det.state.player_state[f"{det_key}_ACTUAL_VICTORY_POINTS"] == 4 + hand.get(
        VICTORY_POINT, 0
    )
    # Deck + all hands == unseen pool.
    deck_counter = Counter(det.state.development_listdeck)
    assert deck_counter + hand == belief.unseen_pool
    # Source game untouched.
    assert game.state.player_state[f"{key}_KNIGHT_IN_HAND"] == 3


def test_patch_full_features_matches_recomputed_full_features():
    game = make_1v1_game()
    set_player_dev_hand(game, Color.BLUE, KNIGHT=1)
    set_player_dev_hand(game, Color.RED, KNIGHT=2)
    belief = DevCardBelief(game, Color.BLUE)
    colors = tuple(game.state.colors)
    names = get_full_numeric_feature_names(2, "BASE")
    numeric_index = {n: i for i, n in enumerate(names)}

    base_full = full_game_to_features(game, 2, "BASE", base_color=Color.BLUE)

    for hyp in belief.enumerate_hypotheses():
        patched = patch_full_features(
            base_full,
            hyp,
            base_color=Color.BLUE,
            colors=colors,
            numeric_name_to_index=numeric_index,
        )
        # Build the same world via a full Game clone and recompute features.
        det = determinize_game(belief, hyp, rng=random.Random(0), reshuffle_deck=False)
        recomputed = full_game_to_features(det, 2, "BASE", base_color=Color.BLUE)
        assert np.allclose(patched, recomputed, atol=1e-5)
