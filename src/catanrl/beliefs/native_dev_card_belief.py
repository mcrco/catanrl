"""Exact 1v1 belief over the opponent's hidden development cards on cppanatron games.

In 1v1 the opponent's development-card hand is the only hidden information:
its size is public, and the cards it could be are the full deck minus every
card the perspective player holds or has played and every card the opponent
has played. Each composition is weighted by the multivariate hypergeometric
probability of drawing it from that unseen pool. Compositions whose victory
point cards would already give the opponent a win are excluded, since the
game would have ended. Only public information and the perspective player's
own hand are read, never the opponent's actual cards.

``determinize`` turns a composition into a consistent perfect-information game
through ``Game.determinize_development_cards``, which also rebuilds the deck.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import product

import cppanatron as cpp
import numpy as np

# Catanatron's deck in DevelopmentCard order: knight, year of plenty,
# monopoly, road building, victory point.
DEVELOPMENT_DECK = (14, 2, 2, 2, 5)
_VICTORY_POINT = 4


@dataclass(frozen=True)
class Hypothesis:
    cards: tuple[int, int, int, int, int]
    weight: float


def unseen_pool(game: cpp.Game, perspective: int) -> list[int]:
    """Cards the opponent's hidden hand and the deck are drawn from."""
    if game.num_players != 2:
        raise ValueError("the exact development-card belief is for 1v1 games")
    me = game.player(perspective)
    them = game.player(1 - perspective)
    pool = [
        DEVELOPMENT_DECK[index]
        - me.development_cards[index]
        - me.played_development_cards[index]
        - them.played_development_cards[index]
        for index in range(5)
    ]
    if min(pool) < 0:
        raise RuntimeError(f"negative unseen development-card pool {pool}")
    return pool


def development_deck_counts(game: cpp.Game) -> list[int]:
    """Remaining deck composition: the full deck minus every held or played card."""
    counts = list(DEVELOPMENT_DECK)
    for player in range(game.num_players):
        state = game.player(player)
        for index, (held, played) in enumerate(
            zip(state.development_cards, state.played_development_cards)
        ):
            counts[index] -= held + played
    if sum(counts) != game.development_cards_remaining or min(counts) < 0:
        raise RuntimeError(
            f"development deck {counts} disagrees with "
            f"{game.development_cards_remaining} remaining cards"
        )
    return counts


def opponent_belief(game: cpp.Game, perspective: int) -> list[Hypothesis]:
    """Every composition of the opponent's hidden hand with its probability."""
    pool = unseen_pool(game, perspective)
    them = game.player(1 - perspective)
    hand_size = sum(them.development_cards)
    max_victory_cards = game.vps_to_win - 1 - them.victory_points
    hypotheses: list[tuple[tuple[int, ...], float]] = []
    for head in product(*(range(min(count, hand_size) + 1) for count in pool[:4])):
        victory_cards = hand_size - sum(head)
        if victory_cards < 0 or victory_cards > pool[_VICTORY_POINT]:
            continue
        if victory_cards > max_victory_cards:
            continue
        cards = (*head, victory_cards)
        weight = math.prod(math.comb(available, take) for available, take in zip(pool, cards))
        if weight > 0:
            hypotheses.append((cards, float(weight)))
    total = sum(weight for _, weight in hypotheses)
    if total <= 0:
        raise RuntimeError("no opponent development hand is consistent with the game")
    return [Hypothesis(cards, weight / total) for cards, weight in hypotheses]  # type: ignore[arg-type]


def sample_hypotheses(
    game: cpp.Game, perspective: int, count: int, rng: np.random.Generator
) -> list[tuple[tuple[int, ...], int]]:
    """``count`` belief samples merged into ``(cards, multiplicity)`` pairs."""
    hypotheses = opponent_belief(game, perspective)
    if len(hypotheses) == 1:
        return [(hypotheses[0].cards, count)]
    draws = rng.choice(len(hypotheses), size=count, p=[h.weight for h in hypotheses])
    indices, multiplicities = np.unique(draws, return_counts=True)
    return [(hypotheses[int(i)].cards, int(m)) for i, m in zip(indices, multiplicities)]


def determinize(
    game: cpp.Game, perspective: int, cards: tuple[int, ...], seed: int
) -> cpp.Game:
    """A copy of ``game`` where the opponent holds ``cards`` and the deck matches."""
    world = game.copy()
    world.determinize_development_cards(1 - perspective, list(cards), seed)
    return world


__all__ = [
    "DEVELOPMENT_DECK",
    "Hypothesis",
    "determinize",
    "development_deck_counts",
    "opponent_belief",
    "sample_hypotheses",
    "unseen_pool",
]
