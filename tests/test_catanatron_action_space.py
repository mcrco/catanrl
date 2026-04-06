"""Round-trip tests for catanrl relative robber action encoding."""

from catanatron.models.actions import Action
from catanatron.models.enums import ActionType

from catanrl.utils.catanatron_action_space import (
    from_action_space,
    get_action_array,
    get_old_master_1v1_action_array,
    get_old_master_1v1_to_current_index_groups,
    get_player_colors,
    to_action_space,
)


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


def test_move_robber_1v1_collapses_victim_choice_to_tile():
    game_colors = tuple(get_player_colors(2))
    actor = game_colors[0]
    coord = (0, 0, 0)
    steal_action = Action(actor, ActionType.MOVE_ROBBER, (coord, game_colors[1]))
    no_steal_action = Action(actor, ActionType.MOVE_ROBBER, (coord, None))
    idx_steal = to_action_space(steal_action, len(game_colors), "BASE", game_colors)
    idx_no_steal = to_action_space(no_steal_action, len(game_colors), "BASE", game_colors)
    assert idx_steal == idx_no_steal
    back = from_action_space(
        idx_steal,
        actor,
        len(game_colors),
        "BASE",
        game_colors,
        playable_actions=[steal_action],
    )
    assert back == steal_action


def test_old_master_1v1_mapping_covers_current_action_space():
    old_action_array = get_old_master_1v1_action_array("BASE")
    current_action_array = get_action_array(2, "BASE")
    index_groups = get_old_master_1v1_to_current_index_groups("BASE")

    assert len(index_groups) == len(old_action_array)
    flattened = sorted(idx for group in index_groups for idx in group)
    assert flattened == list(range(len(current_action_array)))
