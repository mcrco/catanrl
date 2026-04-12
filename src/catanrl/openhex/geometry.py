"""
OpenHex cube geometry aligned with https://open-hex.web.app/#/coordinates.

Verification (see scripts/verify_openhex_indices.py):
- OpenHex vertex *indices* 0..53 do NOT match catanatron NodeId assignment (only one collision).
- OpenHex hex *indices* 0..18 follow spatial order, not catanatron tile ids.
- OpenHex edge *indices* 0..71 follow JS construction order, not catanatron get_edges() order.

Therefore we register API bots with coordinateFormat \"coordinates\" and convert
cube triples to catanatron node ids / edges via this module.
"""

from __future__ import annotations

from typing import Dict, FrozenSet, Tuple

from catanatron.models.coordinate_system import Coordinate
from catanatron.models.map import CatanMap, build_map

Coordinate3 = Tuple[int, int, int]

# From OpenHex app.js renderCoordinates() — must stay in sync with the live site.
OPENHEX_HEXES: Tuple[Coordinate3, ...] = (
    (0, 0, 0),
    (1, -1, 0),
    (1, 0, -1),
    (0, 1, -1),
    (-1, 1, 0),
    (-1, 0, 1),
    (0, -1, 1),
    (2, -2, 0),
    (2, -1, -1),
    (2, 0, -2),
    (1, 1, -2),
    (0, 2, -2),
    (-1, 2, -1),
    (-2, 2, 0),
    (-2, 1, 1),
    (-2, 0, 2),
    (-1, -1, 2),
    (0, -2, 2),
    (1, -2, 1),
)

OPENHEX_VERTICES: Tuple[Coordinate3, ...] = (
    (-2, 0, 3),
    (-2, 1, 2),
    (-2, 2, 1),
    (-2, 3, 0),
    (-1, -1, 3),
    (-1, 0, 2),
    (-1, 1, 1),
    (-1, 2, 0),
    (-1, 3, -1),
    (0, -2, 3),
    (0, -1, 2),
    (0, 0, 1),
    (0, 1, 0),
    (0, 2, -1),
    (0, 3, -2),
    (1, -2, 2),
    (1, -1, 1),
    (1, 0, 0),
    (1, 1, -1),
    (1, 2, -2),
    (2, -2, 1),
    (2, -1, 0),
    (2, 0, -1),
    (2, 1, -2),
    (3, -2, 0),
    (3, -1, -1),
    (3, 0, -2),
    (-2, 1, 3),
    (-2, 2, 2),
    (-2, 3, 1),
    (-1, 0, 3),
    (-1, 1, 2),
    (-1, 2, 1),
    (-1, 3, 0),
    (0, -1, 3),
    (0, 0, 2),
    (0, 1, 1),
    (0, 2, 0),
    (0, 3, -1),
    (1, -2, 3),
    (1, -1, 2),
    (1, 0, 1),
    (1, 1, 0),
    (1, 2, -1),
    (1, 3, -2),
    (2, -2, 2),
    (2, -1, 1),
    (2, 0, 0),
    (2, 1, -1),
    (2, 2, -2),
    (3, -2, 1),
    (3, -1, 0),
    (3, 0, -1),
    (3, 1, -2),
)


def _hexes_for_vertex(v: Coordinate3) -> Tuple[Coordinate3, Coordinate3, Coordinate3]:
    vx, vy, vz = v
    s = vx + vy + vz
    if s == 1:
        return ((vx, vy - 1, vz), (vx - 1, vy, vz), (vx, vy, vz - 1))
    if s == 2:
        return ((vx - 1, vy, vz - 1), (vx, vy - 1, vz - 1), (vx - 1, vy - 1, vz))
    raise ValueError(f"vertex triple must have sum 1 or 2, got {s}")


def vertex_cube_to_node_id(m: CatanMap, v: Coordinate3) -> int:
    """Map an OpenHex vertex cube coordinate to catanatron node id."""
    trip = _hexes_for_vertex(v)
    nodes_sets: list[set[int]] = []
    for h in trip:
        if h not in m.tiles:
            raise KeyError(f"hex {h} not on map for vertex {v}")
        nodes_sets.append(set(m.tiles[h].nodes.values()))
    inter: FrozenSet[int] = nodes_sets[0] & nodes_sets[1] & nodes_sets[2]
    if len(inter) != 1:
        raise ValueError(f"vertex {v} does not map to a unique node: {inter}")
    return next(iter(inter))


def edge_vertices_to_edge_tuple(
    m: CatanMap, a: Coordinate3, b: Coordinate3
) -> Tuple[int, int]:
    """Two vertex cube coords that form an edge -> sorted catanatron edge (node, node)."""
    n1 = vertex_cube_to_node_id(m, a)
    n2 = vertex_cube_to_node_id(m, b)
    return tuple(sorted((n1, n2)))  # type: ignore[return-value]


def hex_dict_to_coordinate(h: Dict[str, int] | Coordinate3) -> Coordinate:
    if isinstance(h, tuple):
        return h
    return (int(h["x"]), int(h["y"]), int(h["z"]))


def vertex_dict_to_tuple(v: Dict[str, int] | Coordinate3) -> Coordinate3:
    if isinstance(v, tuple):
        return v
    return (int(v["x"]), int(v["y"]), int(v["z"]))


def build_vertex_cube_to_node_cache(m: CatanMap) -> Dict[Coordinate3, int]:
    """Precompute all 54 vertices for fast lookup."""
    out: Dict[Coordinate3, int] = {}
    for v in OPENHEX_VERTICES:
        out[v] = vertex_cube_to_node_id(m, v)
    return out


# Default BASE map cache (topology matches OpenHex standard board)
_BASE_MAP = build_map("BASE")
VERTEX_CUBE_TO_NODE: Dict[Coordinate3, int] = build_vertex_cube_to_node_cache(_BASE_MAP)
NODE_ID_TO_VERTEX_CUBE: Dict[int, Coordinate3] = {n: v for v, n in VERTEX_CUBE_TO_NODE.items()}
