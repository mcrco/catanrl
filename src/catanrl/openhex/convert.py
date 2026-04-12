"""Map catanatron Actions to OpenHex POST JSON and helpers."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from catanatron.models.enums import Action, ActionType
from catanatron.models.map import CatanMap

from catanrl.openhex.geometry import NODE_ID_TO_VERTEX_CUBE, hex_dict_to_coordinate


def _vdict(v: Tuple[int, int, int]) -> Dict[str, int]:
    return {"x": v[0], "y": v[1], "z": v[2]}


def color_to_oh(c) -> str:
    return c.value.lower()


def oh_to_color(s: str):
    from catanatron.models.player import Color

    return Color[s.upper()]


def _maritime_trade_to_openhex(value: Tuple) -> Dict[str, Any]:
    """MARITIME_TRADE 5-tuple from catanatron_action_space (4:1, 3:1 port, 2:1 port)."""
    j = value[-1]
    giving = [x for x in value[:-1] if x is not None]
    if not giving or j is None:
        raise ValueError(f"invalid maritime trade tuple {value}")
    give_res = giving[0]
    if not all(x == give_res for x in giving):
        raise ValueError(f"maritime giving resources must match {value}")
    n = len(giving)
    if n not in (2, 3, 4):
        raise ValueError(f"unexpected maritime offer length {n}")
    return {
        "type": "tradeWithBank",
        "offering": {str(give_res).lower(): n},
        "requesting": {str(j).lower(): 1},
    }


def action_to_openhex_action_object(
    action: Action, catan_map: CatanMap, *, is_initial_build_phase: bool = False
) -> Dict[str, Any]:
    """Single OpenHex `action` object (nested under POST `action`)."""
    at = action.action_type
    val = action.value

    if at == ActionType.ROLL:
        return {"type": "rollDice"}
    if at == ActionType.END_TURN:
        return {"type": "endTurn"}
    if at == ActionType.BUY_DEVELOPMENT_CARD:
        return {"type": "buyDevCard"}

    if at == ActionType.BUILD_SETTLEMENT:
        v = NODE_ID_TO_VERTEX_CUBE[int(val)]
        if is_initial_build_phase:
            return {"type": "placeInitialSettlement", "vertex": _vdict(v)}
        return {"type": "buildSettlement", "vertex": _vdict(v)}
    if at == ActionType.BUILD_CITY:
        v = NODE_ID_TO_VERTEX_CUBE[int(val)]
        return {"type": "buildCity", "vertex": _vdict(v)}
    if at == ActionType.BUILD_ROAD:
        e0, e1 = val
        v0 = NODE_ID_TO_VERTEX_CUBE[int(e0)]
        v1 = NODE_ID_TO_VERTEX_CUBE[int(e1)]
        edge = [_vdict(v0), _vdict(v1)]
        if is_initial_build_phase:
            return {"type": "placeInitialRoad", "edge": edge}
        return {"type": "buildRoad", "edge": edge}

    if at == ActionType.MOVE_ROBBER:
        coord, victim = val
        hx = hex_dict_to_coordinate(coord)
        body: Dict[str, Any] = {"type": "moveRobber", "hex": _vdict((hx[0], hx[1], hx[2]))}
        return body

    if at == ActionType.DISCARD_RESOURCE:
        res = str(val).lower()
        return {"type": "discardResources", "resources": {res: 1}}

    if at == ActionType.PLAY_KNIGHT_CARD:
        return {"type": "playKnight"}
    if at == ActionType.PLAY_ROAD_BUILDING:
        return {"type": "playRoadBuilding"}

    if at == ActionType.PLAY_YEAR_OF_PLENTY:
        r1, r2 = val
        return {
            "type": "playYearOfPlenty",
            "resources": [str(r1).lower(), str(r2).lower()],
        }
    if at == ActionType.PLAY_MONOPOLY:
        return {"type": "playMonopoly", "resource": str(val).lower()}

    if at == ActionType.MARITIME_TRADE:
        return _maritime_trade_to_openhex(val)

    raise NotImplementedError(f"OpenHex export not implemented for {at}")


def openhex_post_bodies_for_action(
    action: Action,
    catan_map: CatanMap,
    duration_ms: int | None = None,
    *,
    is_initial_build_phase: bool = False,
) -> List[Dict[str, Any]]:
    """
    One or more HTTP POST bodies for this catanatron action.
    MOVE_ROBBER with a victim becomes moveRobber then stealFromPlayer (OpenHex guide).
    """
    objs: List[Dict[str, Any]] = []
    if action.action_type == ActionType.MOVE_ROBBER:
        coord, victim = action.value  # type: ignore[misc]
        hx = hex_dict_to_coordinate(coord)
        objs.append({"type": "moveRobber", "hex": _vdict((hx[0], hx[1], hx[2]))})
        if victim is not None:
            objs.append({"type": "stealFromPlayer", "target": color_to_oh(victim)})
    else:
        objs.append(
            action_to_openhex_action_object(
                action, catan_map, is_initial_build_phase=is_initial_build_phase
            )
        )

    out: List[Dict[str, Any]] = []
    for o in objs:
        body: Dict[str, Any] = {"action": o}
        if duration_ms is not None:
            body["duration_ms"] = duration_ms
        out.append(body)
    return out
