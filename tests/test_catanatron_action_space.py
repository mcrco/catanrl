"""Round-trip tests for the catanrl flat action encoding.

The flat action <-> Catanatron Action mapping is the contract every agent relies
on: a single off-by-one or a non-bijective entry silently mislabels training
targets and corrupts both PPO and MCTS.
"""

import pytest
from catanatron.models.actions import Action
from catanatron.models.enums import ActionType

from catanrl.utils.catanatron_action_space import (
    from_action_space,
    get_action_array,
    get_action_space_size,
    get_player_colors,
    to_action_space,
)


@pytest.mark.parametrize("num_players", [2, 3, 4])
@pytest.mark.parametrize("map_type", ["BASE", "MINI"])
def test_every_action_index_round_trips(num_players, map_type):
    """from_action_space(i) -> to_action_space(...) == i for the whole array."""
    game_colors = tuple(get_player_colors(num_players))
    actor = game_colors[0]
    size = get_action_space_size(num_players, map_type)
    assert size == len(get_action_array(num_players, map_type))

    seen = set()
    for idx in range(size):
        action = from_action_space(idx, actor, num_players, map_type, game_colors)
        assert action.color == actor
        round_tripped = to_action_space(action, num_players, map_type, game_colors)
        assert round_tripped == idx, (
            f"index {idx} decoded to {action} which re-encoded to {round_tripped}"
        )
        seen.add(idx)
    assert len(seen) == size


@pytest.mark.parametrize("num_players", [2, 3, 4])
def test_action_array_entries_are_unique(num_players):
    """The action array must be a bijection (no duplicate semantic entries)."""
    array = get_action_array(num_players, "BASE")
    assert len(set(array)) == len(array)


def test_move_robber_round_trip_relative_seating():
    # Seating order differs from canonical PLAYER_COLOR_ORDER
    base = tuple(get_player_colors(4))
    game_colors = (base[2], base[0], base[3], base[1])

    actor = game_colors[0]
    # P1 in features = next in seating
    victim_p1 = game_colors[1]
    coord = (-1, 0, 1)
    action = Action(actor, ActionType.MOVE_ROBBER, (coord, victim_p1))
    idx = to_action_space(action, len(game_colors), "BASE", game_colors)
    back = from_action_space(idx, actor, len(game_colors), "BASE", game_colors)
    assert back == action


def test_move_robber_1v1_keeps_distinct_steal_vs_no_steal_actions():
    game_colors = tuple(get_player_colors(2))
    actor = game_colors[0]
    coord = (0, 0, 0)
    steal_action = Action(actor, ActionType.MOVE_ROBBER, (coord, game_colors[1]))
    no_steal_action = Action(actor, ActionType.MOVE_ROBBER, (coord, None))
    idx_steal = to_action_space(steal_action, len(game_colors), "BASE", game_colors)
    idx_no_steal = to_action_space(no_steal_action, len(game_colors), "BASE", game_colors)
    assert idx_steal != idx_no_steal
    assert from_action_space(idx_steal, actor, len(game_colors), "BASE", game_colors) == steal_action
    assert (
        from_action_space(idx_no_steal, actor, len(game_colors), "BASE", game_colors)
        == no_steal_action
    )
