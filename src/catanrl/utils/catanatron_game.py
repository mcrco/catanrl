"""Shared Catanatron game setup helpers used across training, eval, and envs."""

from __future__ import annotations

from typing import Literal, Sequence, Tuple

from catanatron.game import Game
from catanatron.models.actions import generate_playable_actions
from catanatron.models.player import Color, Player

# Default seating for features, self-play, and most eval scripts.
COLOR_ORDER: Tuple[Color, ...] = (
    Color.RED,
    Color.BLUE,
    Color.ORANGE,
    Color.WHITE,
)

# Puffer/action-space seating: controlled player (Blue) is always index 0.
PLAYER_COLOR_ORDER: Tuple[Color, ...] = (
    Color.BLUE,
    Color.RED,
    Color.WHITE,
    Color.ORANGE,
)

SeatOption = Literal["random", "first", "second"]


def get_player_colors(num_players: int) -> Tuple[Color, ...]:
    if not 1 <= num_players <= len(PLAYER_COLOR_ORDER):
        raise ValueError(
            f"num_players must be in [1, {len(PLAYER_COLOR_ORDER)}], got {num_players}"
        )
    return PLAYER_COLOR_ORDER[:num_players]


def force_player_order(game: Game, players: Sequence[Player]) -> None:
    """Seat ``players`` in the given order regardless of catanatron shuffling."""
    colors = tuple(player.color for player in players)
    game.state.players = list(players)
    game.state.colors = colors
    game.state.color_to_index = {color: idx for idx, color in enumerate(colors)}
    game.state.current_player_index = 0
    game.state.current_turn_index = 0
    game.playable_actions = generate_playable_actions(game.state)


def build_players_for_seat(
    lead_player: Player,
    opponents: Sequence[Player],
    nn_seat: SeatOption,
) -> tuple[list[Player], bool]:
    """Return desired player order plus whether to let upstream shuffle seating."""
    opponents_list = list(opponents)
    if nn_seat == "random":
        return [lead_player] + opponents_list, True
    if nn_seat == "first":
        return [lead_player] + opponents_list, False
    if nn_seat == "second":
        if not opponents_list:
            raise ValueError("Cannot place the NN player second without at least one opponent.")
        return [opponents_list[0], lead_player] + opponents_list[1:], False
    raise ValueError(f"Unknown nn_seat '{nn_seat}'")


def color_label(color: Color) -> str:
    return getattr(color, "value", str(color))
