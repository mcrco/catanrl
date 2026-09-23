"""CatanRL's flat action space and observations on the native engine.

Engine rules are covered by cppanatron's own suite (``cppanatron/tests/python``).
These tests replay random Catanatron games on a :class:`FlatGame` and check what
CatanRL layers on top: flat legality masks agree with ``to_action_space`` and
native observations agree with ``full_game_to_features`` at every step.
"""

from __future__ import annotations

import numpy as np
import pytest
import cppanatron as cpp
from catanatron.models.enums import ActionType
from cppanatron.catanatron import (
    from_catanatron_card,
    from_catanatron_resource,
    reference_game,
)

from catanrl.envs.cppanatron import FlatGame
from catanrl.features.catanatron_utils import full_game_to_features
from catanrl.utils.catanatron_action_space import from_action_space, to_action_space


def test_setup_plays_through_flat_indices():
    game = FlatGame(2, "TOURNAMENT", seed=42)
    assert game.action_space_size == 313
    played = []
    while game.game.is_initial_build_phase:
        action = int(np.flatnonzero(game.action_mask())[0])
        played.append(action)
        game.step(action)
    assert played == [126, 54, 128, 59, 130, 61, 132, 64]
    assert game.current_player == 0 and game.num_turns == 2
    assert game.player(0).resources == (0, 1, 0, 1, 1)


def test_unplayable_flat_actions_raise():
    game = FlatGame(2, "BASE", seed=3)
    with pytest.raises(ValueError, match="not currently playable"):
        game.step(game.action_space_size - 1)


def _python_mask(python_game, num_players: int, map_type: str) -> np.ndarray:
    mask = np.zeros(len(cpp.FlatActionSpace(num_players, cpp.MapType[map_type])), dtype=np.bool_)
    colors = tuple(python_game.state.colors)
    for action in python_game.playable_actions:
        mask[to_action_space(action, num_players, map_type, colors)] = True
    return mask


def _native_outcome(action, result) -> dict:
    if action.action_type == ActionType.ROLL:
        return {"dice": result}
    if action.action_type == ActionType.BUY_DEVELOPMENT_CARD:
        return {"development_card": from_catanatron_card(result)}
    if action.action_type == ActionType.MOVE_ROBBER and result is not None:
        return {"stolen_resource": from_catanatron_resource(result)}
    return {}


@pytest.mark.parametrize("map_type", ["MINI", "BASE", "TOURNAMENT"])
@pytest.mark.parametrize("num_players", [2, 3, 4])
def test_replayed_games_share_masks_and_observations(map_type, num_players):
    game = FlatGame(num_players, map_type, seed=777, vps_to_win=6)
    python_game = reference_game(game.game, cpp.MapType[map_type], seed=777)
    colors = tuple(python_game.state.colors)
    rng = np.random.default_rng(991)

    for step in range(2_000):
        mask = game.action_mask()
        np.testing.assert_array_equal(
            mask,
            _python_mask(python_game, num_players, map_type),
            err_msg=f"legal-action mismatch before step {step}",
        )
        for player in range(num_players):
            np.testing.assert_allclose(
                game.observation(player),
                full_game_to_features(
                    python_game, num_players, map_type, base_color=colors[player]
                ),
                rtol=0,
                atol=1e-7,
                err_msg=f"observation mismatch for seat {player} before step {step}",
            )
        if python_game.winning_color() is not None or game.num_turns >= 1_000:
            break

        index = int(rng.choice(np.flatnonzero(mask)))
        action = from_action_space(
            index,
            python_game.state.current_color(),
            num_players,
            map_type,
            colors,
            python_game.playable_actions,
        )
        result = python_game.execute(action).result
        game.step(index, **_native_outcome(action, result))

    assert game.winner == (
        None if python_game.winning_color() is None else colors.index(python_game.winning_color())
    )

