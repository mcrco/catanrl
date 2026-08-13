from __future__ import annotations

import json
import random

from catanatron.game import Game
from catanatron.models.enums import RESOURCES, SETTLEMENT, Action, ActionPrompt, ActionType
from catanatron.models.player import Color, RandomPlayer

from catanrl.eval.canopy_catanatron_bridge import (
    CANOPY_LAND_HEXES,
    CanopyTopologyMapping,
    _update_tested_non_knight_after_action,
    _update_tested_non_knight_before_action,
    canopy_action_to_catanatron,
    canopy_tested_non_knight,
    catanatron_action_to_canopy,
    game_to_canopy_snapshot,
)


def _game(seed: int = 7) -> Game:
    return Game(
        [RandomPlayer(Color.RED), RandomPlayer(Color.BLUE)],
        seed=seed,
        discard_limit=9,
        vps_to_win=15,
    )


def test_canopy_snapshot_preserves_base_topology_contract() -> None:
    game = _game()
    mapping = CanopyTopologyMapping.from_game(game)
    snapshot = game_to_canopy_snapshot(game, tested_non_knight=(0, 0))

    assert len(mapping.nodes) == 54
    assert len(mapping.edges) == 72
    assert len(set(mapping.nodes)) == 54
    assert len(set(mapping.edges)) == 72
    assert len(snapshot["terrains"]) == 19
    assert len(snapshot["numbers"]) == 19
    assert len(snapshot["ports"]) == 9
    assert len(snapshot["tile_nodes"]) == 19
    assert len(snapshot["tile_edges"]) == 19
    assert all(len(nodes) == 6 for nodes in snapshot["tile_nodes"])
    assert all(len(edges) == 6 for edges in snapshot["tile_edges"])
    assert snapshot["vp_limit"] == 15
    assert snapshot["discard_threshold"] == 9
    assert snapshot["phase"] == {"kind": "place_settlement"}
    json.dumps(snapshot)


def test_every_canopy_action_id_has_an_exact_catanatron_mapping() -> None:
    game = _game()
    mapping = CanopyTopologyMapping.from_game(game)
    color = game.state.current_color()
    opponent = next(candidate for candidate in game.state.colors if candidate != color)

    for action_id in range(249):
        if action_id < 54:
            expected = Action(color, ActionType.BUILD_SETTLEMENT, mapping.nodes[action_id])
        elif action_id < 126:
            expected = Action(color, ActionType.BUILD_ROAD, mapping.edges[action_id - 54])
        elif action_id < 180:
            expected = Action(color, ActionType.BUILD_CITY, mapping.nodes[action_id - 126])
        elif action_id == 180:
            expected = Action(color, ActionType.ROLL, None)
        elif action_id == 181:
            expected = Action(color, ActionType.END_TURN, None)
        elif action_id == 182:
            expected = Action(color, ActionType.BUY_DEVELOPMENT_CARD, None)
        elif action_id == 183:
            expected = Action(color, ActionType.PLAY_KNIGHT_CARD, None)
        elif action_id == 184:
            expected = Action(color, ActionType.PLAY_ROAD_BUILDING, None)
        elif action_id < 200:
            pairs = [
                (first, second)
                for index, first in enumerate(RESOURCES)
                for second in RESOURCES[index:]
            ]
            expected = Action(
                color,
                ActionType.PLAY_YEAR_OF_PLENTY,
                pairs[action_id - 185],
            )
        elif action_id < 205:
            expected = Action(
                color,
                ActionType.PLAY_MONOPOLY,
                RESOURCES[action_id - 200],
            )
        elif action_id < 224:
            q, r = CANOPY_LAND_HEXES[action_id - 205]
            expected = Action(
                color,
                ActionType.MOVE_ROBBER,
                ((q, -q - r, r), opponent),
            )
        elif action_id < 229:
            expected = Action(
                color,
                ActionType.DISCARD_RESOURCE,
                RESOURCES[action_id - 224],
            )
        else:
            pair_index = action_id - 229
            give_index = pair_index // 4
            receive_index = pair_index % 4
            if receive_index >= give_index:
                receive_index += 1
            give = RESOURCES[give_index]
            receive = RESOURCES[receive_index]
            expected = Action(
                color,
                ActionType.MARITIME_TRADE,
                (give, give, give, give, receive),
            )

        assert canopy_action_to_catanatron(action_id, game, [expected]) == expected
        assert catanatron_action_to_canopy(expected, mapping) == action_id


def test_snapshot_serializes_each_decision_during_a_live_game() -> None:
    game = _game(seed=19)
    for _ in range(100):
        if game.winning_color() is not None:
            break
        snapshot = game_to_canopy_snapshot(game)
        assert snapshot["current_player"] == game.state.current_player_index
        assert len(snapshot["players"]) == 2
        json.dumps(snapshot)
        game.execute(game.playable_actions[0])


def test_snapshot_uses_prompt_and_turn_owner_for_robber_phase_state() -> None:
    game = _game()
    state = game.state
    state.is_initial_build_phase = False
    state.current_prompt = ActionPrompt.DISCARD
    state.is_discarding = True
    state.current_turn_index = 0
    state.current_player_index = 1
    state.discard_counts = [0, 2]
    state.player_state["P0_HAS_ROLLED"] = True

    snapshot = game_to_canopy_snapshot(game, tested_non_knight=(0, 0))
    assert snapshot["phase"] == {"kind": "discard", "remaining": 2, "roller": 0}
    assert snapshot["pre_roll"] is False

    # Upstream Catanatron leaves this auxiliary flag stale after moving the
    # robber; current_prompt remains the authoritative phase signal.
    state.current_player_index = 0
    state.current_prompt = ActionPrompt.PLAY_TURN
    state.is_discarding = False
    state.is_moving_knight = True
    snapshot = game_to_canopy_snapshot(game, tested_non_knight=(0, 0))
    assert snapshot["phase"] == {"kind": "main"}


def test_tested_non_knight_matches_canopy_transition_contract() -> None:
    game = _game()
    state = game.state
    color = state.colors[0]
    index = state.color_to_index[color]
    key = f"P{index}"
    coordinate, tile = next(
        (coordinate, tile)
        for coordinate, tile in state.board.map.land_tiles.items()
        if tile.resource is not None
    )
    state.board.robber_coordinate = coordinate
    state.buildings_by_color[color][SETTLEMENT].append(next(iter(tile.nodes.values())))
    state.player_state[f"{key}_KNIGHT_IN_HAND"] = 2
    tested = [0, 0]

    _update_tested_non_knight_before_action(
        game,
        Action(color, ActionType.END_TURN, None),
        tested,
    )
    assert tested[index] == 2

    state.player_state[f"{key}_KNIGHT_IN_HAND"] = 0
    state.player_state[f"{key}_YEAR_OF_PLENTY_IN_HAND"] = 1
    _update_tested_non_knight_after_action(
        game,
        Action(color, ActionType.PLAY_YEAR_OF_PLENTY, ("WOOD", "WOOD")),
        tested,
    )
    assert tested[index] == 1

    _update_tested_non_knight_after_action(
        game,
        Action(color, ActionType.PLAY_KNIGHT_CARD, None),
        tested,
    )
    assert tested[index] == 0


def test_history_derivation_does_not_perturb_catanatron_randomness() -> None:
    game = _game(seed=29)
    for _ in range(30):
        game.execute(game.playable_actions[0])

    random.seed(8675309)
    before = random.getstate()
    tested = canopy_tested_non_knight(game)

    assert random.getstate() == before
    assert len(tested) == 2
    assert all(value >= 0 for value in tested)
