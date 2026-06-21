"""Belief over opponents' hidden development cards.

In a 1v1 Catan game the *only* hidden information is the composition of the
opponent's development-card hand (resource hands are reconstructible from public
history, and in 1v1 the perspective player witnesses every robber steal). This
module builds an explicit belief over that hidden composition.

For 1v1 the belief is computed *exactly* by enumerating every consistent
opponent hand and weighting it by the multivariate hypergeometric probability of
drawing that multiset from the unseen development-card pool. For games with more
players the belief is estimated by Monte-Carlo sampling instead (the number of
joint hypotheses explodes).

The belief is built purely from public information plus the perspective player's
own hand. It never reads opponents' true hidden cards, so it is an honest model
of what the perspective player can know.
"""

from __future__ import annotations

import math
import random
from collections import Counter
from dataclasses import dataclass, field
from itertools import product
from typing import Callable, Dict, List, Optional

from catanatron.game import Game
from catanatron.models.decks import starting_devcard_bank
from catanatron.models.enums import (
    DEVELOPMENT_CARDS,
    VICTORY_POINT,
)
from catanatron.models.player import Color
from catanatron.state_functions import (
    player_key,
    player_num_dev_cards,
)


@dataclass(frozen=True)
class Hypothesis:
    """A single possible assignment of hidden dev cards to opponents.

    Attributes:
        weight: Probability mass of this hypothesis under the belief. Across all
            hypotheses returned together the weights sum to 1.
        opponent_dev_hands: Maps each opponent color to a Counter of dev-card type
            -> count for that opponent's hidden hand.
    """

    weight: float
    opponent_dev_hands: Dict[Color, Counter] = field(default_factory=dict)


def _starting_dev_counter() -> Counter:
    return Counter(starting_devcard_bank())


def _multivariate_hypergeometric_weight(
    pool: Counter, hand: Counter, draws: int
) -> float:
    """P(drawing exactly ``hand`` when taking ``draws`` cards from ``pool``).

    Uses the multivariate hypergeometric pmf:
        prod_i C(pool_i, hand_i) / C(N, draws)
    where N = sum(pool). Returns 0.0 if the hand is infeasible.
    """
    total = sum(pool.values())
    if draws < 0 or draws > total:
        return 0.0
    denominator = math.comb(total, draws)
    if denominator == 0:
        return 0.0
    numerator = 1
    for card, count in hand.items():
        available = pool.get(card, 0)
        if count > available:
            return 0.0
        numerator *= math.comb(available, count)
    return numerator / denominator


class DevCardBelief:
    """Belief over opponents' hidden development-card hands.

    Construct from a game and the perspective player's color, then call
    :meth:`enumerate_hypotheses` (1v1, exact) or :meth:`sample_hypotheses`
    (n-player, Monte-Carlo).
    """

    def __init__(self, game: Game, perspective_color: Color):
        self.game = game
        self.perspective_color = perspective_color
        state = game.state
        self.colors = tuple(state.colors)
        self.num_players = len(self.colors)
        if perspective_color not in self.colors:
            raise ValueError(
                f"perspective_color {perspective_color} not in game colors {self.colors}"
            )
        self.opponents = tuple(c for c in self.colors if c != perspective_color)
        self.vps_to_win = int(game.vps_to_win)

        # Cards that the perspective player cannot account for. Everything the
        # perspective player can see is removed: their own hand and every dev card
        # that has been played publicly (by anyone). VP cards are never played, so
        # they only leave the pool by being held.
        known = Counter()
        own_key = player_key(state, perspective_color)
        for dev_card in DEVELOPMENT_CARDS:
            known[dev_card] += int(state.player_state[f"{own_key}_{dev_card}_IN_HAND"])
        for color in self.colors:
            key = player_key(state, color)
            for dev_card in DEVELOPMENT_CARDS:
                known[dev_card] += int(state.player_state[f"{key}_PLAYED_{dev_card}"])
        self.unseen_pool: Counter = _starting_dev_counter() - known

        # Public per-opponent info.
        self.opponent_num_devs: Dict[Color, int] = {}
        self.opponent_public_vps: Dict[Color, int] = {}
        for color in self.opponents:
            key = player_key(state, color)
            self.opponent_num_devs[color] = int(player_num_dev_cards(state, color))
            self.opponent_public_vps[color] = int(
                state.player_state[f"{key}_VICTORY_POINTS"]
            )

    def _hand_violates_not_won(self, color: Color, hand: Counter) -> bool:
        """True if holding ``hand`` would already have won the game for ``color``.

        The opponent has not won (the game is still going), so any hypothesis that
        would put them at or above the winning threshold is impossible.
        """
        vp_cards = int(hand.get(VICTORY_POINT, 0))
        return self.opponent_public_vps[color] + vp_cards >= self.vps_to_win

    def _enumerate_hands(self, draws: int) -> List[Counter]:
        """All multisets of size ``draws`` drawable from the unseen pool."""
        cards = [c for c in DEVELOPMENT_CARDS if self.unseen_pool.get(c, 0) > 0]
        caps = [min(self.unseen_pool[c], draws) for c in cards]
        hands: List[Counter] = []
        for combo in product(*(range(cap + 1) for cap in caps)):
            if sum(combo) != draws:
                continue
            hand = Counter(
                {card: n for card, n in zip(cards, combo) if n > 0}
            )
            hands.append(hand)
        return hands

    def enumerate_hypotheses(self) -> List[Hypothesis]:
        """Exact belief for 1v1: every consistent opponent hand with its weight.

        Weights use the multivariate hypergeometric distribution and are pruned by
        the "opponent has not already won" constraint, then renormalized to sum to
        1. Raises if called for games with more than two players.
        """
        if self.num_players != 2:
            raise ValueError(
                "enumerate_hypotheses() supports only 1v1; use sample_hypotheses()."
            )
        opponent = self.opponents[0]
        draws = self.opponent_num_devs[opponent]

        raw: List[tuple[Counter, float]] = []
        for hand in self._enumerate_hands(draws):
            if self._hand_violates_not_won(opponent, hand):
                continue
            weight = _multivariate_hypergeometric_weight(
                self.unseen_pool, hand, draws
            )
            if weight > 0.0:
                raw.append((hand, weight))

        if not raw:
            # Defensive: pruning removed everything (should not happen since the
            # true hand is always feasible and non-winning). Fall back to unpruned.
            for hand in self._enumerate_hands(draws):
                weight = _multivariate_hypergeometric_weight(
                    self.unseen_pool, hand, draws
                )
                if weight > 0.0:
                    raw.append((hand, weight))

        total = sum(w for _, w in raw)
        if total <= 0.0:
            return [Hypothesis(weight=1.0, opponent_dev_hands={opponent: Counter()})]
        return [
            Hypothesis(weight=w / total, opponent_dev_hands={opponent: hand})
            for hand, w in raw
        ]

    def sample_hypotheses(
        self, num_samples: int, rng: Optional[random.Random] = None
    ) -> List[Hypothesis]:
        """Monte-Carlo belief for n-player (or 1v1) games.

        Each sample sequentially deals each opponent's hidden hand from the unseen
        pool without replacement, respecting public ``NUM_DEVS_IN_HAND`` counts and
        the "not already won" constraint. Samples are equally weighted.

        Note: this does not model hidden inter-opponent resource steals (3-4p);
        that is out of scope for the initial implementation.
        """
        if num_samples <= 0:
            raise ValueError("num_samples must be positive")
        rng = rng or random.Random()

        hypotheses: List[Hypothesis] = []
        weight = 1.0 / num_samples
        for _ in range(num_samples):
            pool = [
                card
                for card, count in self.unseen_pool.items()
                for _ in range(int(count))
            ]
            rng.shuffle(pool)
            cursor = 0
            hands: Dict[Color, Counter] = {}
            for color in self.opponents:
                draws = self.opponent_num_devs[color]
                hand = Counter(pool[cursor : cursor + draws])
                cursor += draws
                hands[color] = hand
            hypotheses.append(Hypothesis(weight=weight, opponent_dev_hands=hands))
        return hypotheses

    def marginalize(
        self,
        fn: Callable[[Hypothesis], float],
        *,
        hypotheses: Optional[List[Hypothesis]] = None,
        normalize: bool = True,
    ) -> float:
        """Belief-weighted aggregate of ``fn`` over hypotheses.

        Uses exact enumeration for 1v1 by default; pass ``hypotheses`` to reuse a
        precomputed (e.g. sampled) set.
        """
        if hypotheses is None:
            hypotheses = (
                self.enumerate_hypotheses()
                if self.num_players == 2
                else self.sample_hypotheses(1)
            )
        total_weight = sum(h.weight for h in hypotheses)
        if total_weight <= 0.0:
            return 0.0
        acc = sum(h.weight * fn(h) for h in hypotheses)
        return acc / total_weight if normalize else acc
