"""Tests for OpenHex bridge."""

from catanatron.models.enums import Action, ActionType
from catanatron.models.map import build_map

from catanrl.openhex.board import build_catan_map_from_openhex_board
from catanrl.openhex.convert import action_to_openhex_action_object, openhex_post_bodies_for_action
from catanrl.openhex.geometry import NODE_ID_TO_VERTEX_CUBE, vertex_cube_to_node_id


def test_vertex_roundtrip_base_map():
    m = build_map("BASE")
    for n, v in NODE_ID_TO_VERTEX_CUBE.items():
        assert vertex_cube_to_node_id(m, v) == n


def test_board_roundtrip():
    from catanatron.models.tiles import Port

    RESOURCE_TO_TERRAIN = {
        "WOOD": "forest",
        "BRICK": "hills",
        "SHEEP": "pasture",
        "WHEAT": "fields",
        "ORE": "mountains",
    }
    from catanrl.openhex.geometry import VERTEX_CUBE_TO_NODE

    NODE_TO_V = {n: v for v, n in VERTEX_CUBE_TO_NODE.items()}

    m = build_map("BASE")
    tiles = []
    for coord, tile in m.land_tiles.items():
        x, y, z = coord
        if tile.resource is None:
            tiles.append({"hex": {"x": x, "y": y, "z": z}, "terrain": "desert"})
        else:
            tiles.append(
                {
                    "hex": {"x": x, "y": y, "z": z},
                    "terrain": RESOURCE_TO_TERRAIN[tile.resource],
                    "number": tile.number,
                }
            )
    ports = []
    for coord, tile in m.tiles.items():
        if isinstance(tile, Port):
            for e in tile.edges.values():
                n1, n2 = e
                if max(n1, n2) <= 53:
                    v1, v2 = NODE_TO_V[n1], NODE_TO_V[n2]

                    def vd(t):
                        return {"x": t[0], "y": t[1], "z": t[2]}

                    if tile.resource is None:
                        ports.append({"edge": [vd(v1), vd(v2)], "rate": 3})
                    else:
                        ports.append(
                            {"edge": [vd(v1), vd(v2)], "rate": 2, "resource": tile.resource.lower()}
                        )
                    break
    m2 = build_catan_map_from_openhex_board({"tiles": tiles, "ports": ports})
    for c in m.land_tiles:
        assert m.land_tiles[c].resource == m2.land_tiles[c].resource
        assert m.land_tiles[c].number == m2.land_tiles[c].number


def test_convert_roll():
    from catanatron.models.player import Color

    m = build_map("BASE")
    a = Action(Color.RED, ActionType.ROLL, None)
    o = action_to_openhex_action_object(a, m)
    assert o == {"type": "rollDice"}


def test_convert_move_robber_two_posts():
    from catanatron.models.player import Color

    m = build_map("BASE")
    a = Action(Color.RED, ActionType.MOVE_ROBBER, ((0, 0, 0), Color.BLUE))
    bodies = openhex_post_bodies_for_action(a, m)
    assert len(bodies) == 2
    assert bodies[0]["action"]["type"] == "moveRobber"
    assert bodies[1]["action"]["type"] == "stealFromPlayer"
