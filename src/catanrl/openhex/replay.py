"""Replay OpenHex event streams into a catanatron Game."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

from catanatron.apply_action import apply_action
from catanatron.game import Game
from catanatron.models.actions import generate_playable_actions
from catanatron.models.enums import Action, ActionRecord, ActionType
from catanatron.models.map import CatanMap
from catanatron.models.player import Color, RandomPlayer

from catanrl.openhex.geometry import (
    hex_dict_to_coordinate,
    vertex_cube_to_node_id,
    vertex_dict_to_tuple,
)

OH_DEV_TO_FAST = {
    "knight": "KNIGHT",
    "victoryPoint": "VICTORY_POINT",
    "monopoly": "MONOPOLY",
    "roadBuilding": "ROAD_BUILDING",
    "yearOfPlenty": "YEAR_OF_PLENTY",
}


def _vertex_to_node(m: CatanMap, v: Any) -> int:
    t = vertex_dict_to_tuple(v)
    return vertex_cube_to_node_id(m, t)


def _edge_to_tuple(m: CatanMap, edge: Any) -> Tuple[int, int]:
    a = vertex_dict_to_tuple(edge[0])
    b = vertex_dict_to_tuple(edge[1])
    return tuple(sorted((vertex_cube_to_node_id(m, a), vertex_cube_to_node_id(m, b))))  # type: ignore[return-value]


def build_game_from_openhex(
    turn_order: Sequence[str],
    catan_map: CatanMap,
    policy_player,
    bot_color: Color,
) -> Game:
    """
    Create a Game with 1v1 OpenHex rules and seating order matching `turn_order`
    (e.g. ["red", "blue"]). Uses RandomPlayer for opponents.
    """
    colors = tuple(Color[c.upper()] for c in turn_order)
    seed = find_seed_for_player_order(colors, catan_map)
    players = []
    for c in colors:
        if c == bot_color:
            players.append(policy_player)
        else:
            players.append(RandomPlayer(c))
    return Game(
        players,
        seed=seed,
        catan_map=catan_map,
        vps_to_win=15,
        discard_limit=9,
        friendly_robber=True,
    )


def find_seed_for_player_order(
    colors: Tuple[Color, ...], catan_map: CatanMap, *, max_seed: int = 500_000
) -> int:
    """Find Game RNG seed so state.colors matches `colors` (OpenHex turn order)."""
    for seed in range(max_seed):
        players = [RandomPlayer(c) for c in colors]
        g = Game(
            players,
            seed=seed,
            catan_map=catan_map,
            vps_to_win=15,
            discard_limit=9,
            friendly_robber=True,
        )
        if g.state.colors == colors:
            return seed
    raise RuntimeError(f"no seed found for colors {colors} within {max_seed}")


def apply_openhex_event(
    game: Game,
    catan_map: CatanMap,
    event: Dict[str, Any],
    *,
    dev_card_reveal: str | None = None,
) -> None:
    """
    Apply one OpenHex event to `game` via apply_action.
    Pass dev_card_reveal when event is devCardBought for your bot (card known).
    """
    et = event.get("type")
    if et in (
        "resourcesProduced",
        "setupResourcesCollected",
        "longestRoadClaimed",
        "largestArmyClaimed",
        "gameWon",
    ):
        return

    if et == "diceRolled":
        color = Color[event["player"].upper()]
        die1, die2 = int(event["die1"]), int(event["die2"])
        action = Action(color, ActionType.ROLL, None)
        apply_action(game.state, action, ActionRecord(action=action, result=(die1, die2)))
        game.playable_actions = generate_playable_actions(game.state)
        return

    if et == "settlementBuilt":
        color = Color[event["player"].upper()]
        node = _vertex_to_node(catan_map, event["vertex"])
        action = Action(color, ActionType.BUILD_SETTLEMENT, node)
        apply_action(game.state, action)
        game.playable_actions = generate_playable_actions(game.state)
        return

    if et == "roadBuilt":
        color = Color[event["player"].upper()]
        edge = _edge_to_tuple(catan_map, event["edge"])
        action = Action(color, ActionType.BUILD_ROAD, edge)
        apply_action(game.state, action)
        game.playable_actions = generate_playable_actions(game.state)
        return

    if et == "cityBuilt":
        color = Color[event["player"].upper()]
        node = _vertex_to_node(catan_map, event["vertex"])
        action = Action(color, ActionType.BUILD_CITY, node)
        apply_action(game.state, action)
        game.playable_actions = generate_playable_actions(game.state)
        return

    if et == "turnEnded":
        color = Color[event["player"].upper()]
        action = Action(color, ActionType.END_TURN, None)
        apply_action(game.state, action)
        game.playable_actions = generate_playable_actions(game.state)
        return

    if et == "devCardBought":
        color = Color[event["player"].upper()]
        card = event.get("card") or dev_card_reveal
        if card is None:
            raise ValueError(
                "devCardBought without visible card (opponent purchase); "
                "cannot replay the dev deck to match the server"
            )
        fast = OH_DEV_TO_FAST.get(card, card.upper())
        action = Action(color, ActionType.BUY_DEVELOPMENT_CARD, None)
        apply_action(game.state, action, ActionRecord(action=action, result=fast))
        game.playable_actions = generate_playable_actions(game.state)
        return

    if et == "knightPlayed":
        color = Color[event["player"].upper()]
        action = Action(color, ActionType.PLAY_KNIGHT_CARD, None)
        apply_action(game.state, action)
        game.playable_actions = generate_playable_actions(game.state)
        return

    if et == "robberMoved":
        color = Color[event["player"].upper()]
        hx = hex_dict_to_coordinate(event["hex"])
        action = Action(color, ActionType.MOVE_ROBBER, (hx, None))
        apply_action(game.state, action)
        game.playable_actions = generate_playable_actions(game.state)
        return

    if et == "resourceStolen":
        stealer = Color[event["player"].upper()]
        target = Color[event["target"].upper()]
        res = event.get("resource")
        if res is None:
            raise ValueError("resourceStolen needs resource for replay")
        coord = game.state.board.robber_coordinate
        fast = str(res).upper()
        action = Action(stealer, ActionType.MOVE_ROBBER, (coord, target))
        apply_action(game.state, action, ActionRecord(action=action, result=fast))
        game.playable_actions = generate_playable_actions(game.state)
        return

    if et == "resourcesDiscarded":
        color = Color[event["player"].upper()]
        resources = event["resources"]
        for res_name, count in resources.items():
            fast = str(res_name).upper()
            for _ in range(int(count)):
                action = Action(color, ActionType.DISCARD_RESOURCE, fast)
                apply_action(game.state, action)
        game.playable_actions = generate_playable_actions(game.state)
        return

    if et == "bankTradeExecuted":
        color = Color[event["player"].upper()]
        offering = event["offering"]
        requesting = event["requesting"]
        give_res = next(iter(offering.keys()))
        take_res = next(iter(requesting.keys()))
        give_n = int(offering[give_res])
        take_n = int(requesting[take_res])
        give_res_u = str(give_res).upper()
        take_res_u = str(take_res).upper()
        if give_n == 4 and take_n == 1:
            tup = (give_res_u, give_res_u, give_res_u, give_res_u, take_res_u)
        elif give_n == 3 and take_n == 1:
            tup = (give_res_u, give_res_u, give_res_u, None, take_res_u)
        elif give_n == 2 and take_n == 1:
            tup = (give_res_u, give_res_u, None, None, take_res_u)
        else:
            raise ValueError(f"unsupported bank trade {event}")
        action = Action(color, ActionType.MARITIME_TRADE, tup)
        apply_action(game.state, action)
        game.playable_actions = generate_playable_actions(game.state)
        return

    if et == "monopolyPlayed":
        color = Color[event["player"].upper()]
        res = str(event["resource"]).upper()
        action = Action(color, ActionType.PLAY_MONOPOLY, res)
        apply_action(game.state, action)
        game.playable_actions = generate_playable_actions(game.state)
        return

    if et == "yearOfPlentyPlayed":
        color = Color[event["player"].upper()]
        r1, r2 = event["resources"]
        action = Action(
            color,
            ActionType.PLAY_YEAR_OF_PLENTY,
            (str(r1).upper(), str(r2).upper()),
        )
        apply_action(game.state, action)
        game.playable_actions = generate_playable_actions(game.state)
        return

    if et == "roadBuildingPlayed":
        color = Color[event["player"].upper()]
        action = Action(color, ActionType.PLAY_ROAD_BUILDING, None)
        apply_action(game.state, action)
        game.playable_actions = generate_playable_actions(game.state)
        return

    raise NotImplementedError(f"OpenHex event replay not implemented: {et}")


def replay_openhex_events(
    game: Game,
    catan_map: CatanMap,
    events: Sequence[Dict[str, Any]],
    *,
    dev_card_hints: Dict[int, str] | None = None,
) -> None:
    """Apply events in order (each item is the inner `event` object, not the wrapper)."""
    dev_card_hints = dev_card_hints or {}
    events_list = list(events)
    i = 0
    while i < len(events_list):
        ev = events_list[i]
        nxt = events_list[i + 1] if i + 1 < len(events_list) else None
        if (
            ev.get("type") == "robberMoved"
            and nxt is not None
            and nxt.get("type") == "resourceStolen"
            and ev.get("player") == nxt.get("player")
        ):
            color = Color[ev["player"].upper()]
            hx = hex_dict_to_coordinate(ev["hex"])
            target = Color[nxt["target"].upper()]
            res = nxt.get("resource")
            if res is None:
                raise ValueError("resourceStolen needs resource for replay")
            fast = str(res).upper()
            action = Action(color, ActionType.MOVE_ROBBER, (hx, target))
            apply_action(game.state, action, ActionRecord(action=action, result=fast))
            game.playable_actions = generate_playable_actions(game.state)
            i += 2
            continue

        hint = dev_card_hints.get(i)
        apply_openhex_event(game, catan_map, ev, dev_card_reveal=hint)
        i += 1
