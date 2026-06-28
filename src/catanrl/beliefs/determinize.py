"""Turn a :class:`Hypothesis` into a concrete world.

Two paths are provided:

* :func:`determinize_game` builds a full perfect-information :class:`Game` clone
  whose opponent dev hands match the hypothesis. Use this for IS-MCTS rollouts
  where the engine must simulate opponent moves.
* :func:`patch_full_features` cheaply rewrites the opponent dev-card slots of an
  already-computed full feature vector. Use this for test-time policy
  marginalization, where building a whole Game per hypothesis is wasteful.
"""

from __future__ import annotations

import random
from typing import Dict, Literal, Optional

import numpy as np

from catanatron.game import Game
from catanatron.models.enums import (
    DEVELOPMENT_CARDS,
    KNIGHT,
    MONOPOLY,
    ROAD_BUILDING,
    VICTORY_POINT,
    YEAR_OF_PLENTY,
)
from catanatron.models.player import Color
from catanatron.state_functions import player_key

from catanrl.beliefs.dev_card_belief import DevCardBelief, Hypothesis

OwnedAtStartMode = Literal["held"]

_PLAYABLE_DEV_CARDS = (KNIGHT, MONOPOLY, YEAR_OF_PLENTY, ROAD_BUILDING)


def determinize_game(
    belief: DevCardBelief,
    hypothesis: Hypothesis,
    *,
    rng: Optional[random.Random] = None,
    owned_at_start: OwnedAtStartMode = "held",
    reshuffle_deck: bool = True,
) -> Game:
    """Return a Game clone consistent with ``hypothesis``.

    Opponent dev-card hands are overwritten to match the hypothesis, their
    ``ACTUAL_VICTORY_POINTS`` are recomputed from public VPs plus held VP cards,
    and (optionally) the remaining development deck is rebuilt from the unseen
    pool minus all hypothesized hands and reshuffled so future draws stay
    consistent with the belief.

    The perspective player's own state is untouched. Resources, buildings, and
    the board are identical to the source game (in 1v1 they are fully known).

    Args:
        owned_at_start: How to set opponents' ``*_OWNED_AT_START`` playability
            flags. "held" marks any held dev type as playable, which is exactly
            correct at the start of the perspective player's own turn.
    """
    if owned_at_start != "held":
        raise ValueError(f"Unsupported owned_at_start mode: {owned_at_start!r}")

    game = belief.game.copy()
    state = game.state

    for color, hand in hypothesis.opponent_dev_hands.items():
        key = player_key(state, color)
        for dev_card in DEVELOPMENT_CARDS:
            state.player_state[f"{key}_{dev_card}_IN_HAND"] = int(hand.get(dev_card, 0))
        vp_cards = int(hand.get(VICTORY_POINT, 0))
        state.player_state[f"{key}_ACTUAL_VICTORY_POINTS"] = (
            int(state.player_state[f"{key}_VICTORY_POINTS"]) + vp_cards
        )
        for dev_card in _PLAYABLE_DEV_CARDS:
            state.player_state[f"{key}_{dev_card}_OWNED_AT_START"] = (
                int(hand.get(dev_card, 0)) > 0
            )

    if reshuffle_deck:
        remaining = belief.unseen_pool.copy()
        for hand in hypothesis.opponent_dev_hands.values():
            remaining -= hand
        deck = [
            card for card, count in remaining.items() for _ in range(max(0, int(count)))
        ]
        (rng or random).shuffle(deck)
        state.development_listdeck = deck

    return game


def _perspective_offset(
    colors: tuple[Color, ...], base_color: Color, color: Color
) -> int:
    base_index = colors.index(base_color)
    return (colors.index(color) - base_index) % len(colors)


def patch_full_features(
    full_features: np.ndarray,
    hypothesis: Hypothesis,
    *,
    base_color: Color,
    colors: tuple[Color, ...],
    numeric_name_to_index: Dict[str, int],
) -> np.ndarray:
    """Return a copy of ``full_features`` with opponent dev slots set per hypothesis.

    Only opponent numeric slots that depend on the hypothesized hand are
    modified: ``P{offset}_{DEV}_IN_HAND``, ``P{offset}_{DEV}_PLAYABLE``, and
    ``P{offset}_ACTUAL_VPS``. The board tail and all other features are shared
    from the perfect-information vector (legal in 1v1 where everything else is
    known). ``full_features`` must be a full-info vector built with
    ``base_color`` as perspective.
    """
    patched = full_features.astype(np.float32, copy=True)
    for color, hand in hypothesis.opponent_dev_hands.items():
        offset = _perspective_offset(colors, base_color, color)
        vp_index = numeric_name_to_index.get(f"P{offset}_ACTUAL_VPS")
        public_vp_index = numeric_name_to_index.get(f"P{offset}_PUBLIC_VPS")
        has_played_index = numeric_name_to_index.get(
            f"P{offset}_HAS_PLAYED_DEVELOPMENT_CARD_IN_TURN"
        )
        has_played_dev = (
            bool(patched[has_played_index]) if has_played_index is not None else False
        )
        new_vp = float(hand.get(VICTORY_POINT, 0))
        for dev_card in DEVELOPMENT_CARDS:
            idx = numeric_name_to_index.get(f"P{offset}_{dev_card}_IN_HAND")
            if idx is not None:
                patched[idx] = float(hand.get(dev_card, 0))
        for dev_card in _PLAYABLE_DEV_CARDS:
            idx = numeric_name_to_index.get(f"P{offset}_{dev_card}_PLAYABLE")
            if idx is not None:
                patched[idx] = float(
                    hand.get(dev_card, 0) > 0 and not has_played_dev
                )
        if vp_index is not None and public_vp_index is not None:
            patched[vp_index] = patched[public_vp_index] + new_vp
    return patched
