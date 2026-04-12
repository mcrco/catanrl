#!/usr/bin/env python3
"""
Verify OpenHex index conventions vs catanatron BASE map.

Conclusion (printed): OpenHex vertex indices 0..53, hex indices 0..18, and edge
indices 0..71 do not match catanatron NodeId / tile id / get_edges() ordering.
Register bots with coordinateFormat \"coordinates\" and use catanrl.openhex.geometry.
"""

from __future__ import annotations

from catanatron.models.board import get_edges
from catanatron.models.map import build_map

from catanrl.openhex.geometry import OPENHEX_HEXES, OPENHEX_VERTICES, vertex_cube_to_node_id


def main() -> None:
    m = build_map("BASE")

    # Vertices: OpenHex index -> catanatron node id
    matches_v = sum(
        1
        for i, v in enumerate(OPENHEX_VERTICES)
        if vertex_cube_to_node_id(m, v) == i
    )
    print(f"Vertex index == NodeId for {matches_v} / {len(OPENHEX_VERTICES)} vertices")

    # Hexes: OpenHex index -> tile id at same cube coord
    oh_hex_coords = [tuple(h) for h in OPENHEX_HEXES]
    matches_h = 0
    for i, coord in enumerate(oh_hex_coords):
        if coord in m.land_tiles and m.land_tiles[coord].id == i:
            matches_h += 1
    print(f"Hex index == catanatron LandTile.id for {matches_h} / {len(oh_hex_coords)} land hexes")

    # Edges: OpenHex JS construction order vs get_edges order
    vertex_set = {tuple(v) for v in OPENHEX_VERTICES}
    oh_edges = []
    for (x, y, z) in OPENHEX_VERTICES:
        if x + y + z != 1:
            continue
        for dx, dy, dz in [(1, 0, 0), (0, 1, 0), (0, 0, 1)]:
            nb = (x + dx, y + dy, z + dz)
            if nb in vertex_set:
                a, b = (x, y, z), nb
                oh_edges.append(tuple(sorted((a, b))))
    unique_edges = []
    seen = set()
    for e in oh_edges:
        if e not in seen:
            seen.add(e)
            unique_edges.append(e)

    ce = list(get_edges(m.land_nodes))

    def to_cat(e):
        n1 = vertex_cube_to_node_id(m, e[0])
        n2 = vertex_cube_to_node_id(m, e[1])
        return tuple(sorted((n1, n2)))

    oh_cat = [to_cat(e) for e in unique_edges]
    matches_e = sum(1 for i in range(len(oh_cat)) if i < len(ce) and oh_cat[i] == ce[i])
    print(f"Edge index order matches get_edges() for {matches_e} / {len(ce)} edges")

    print(
        "\nRecommendation: use coordinateFormat \"coordinates\" with "
        "catanrl.openhex.geometry / convert.py (see module docstrings)."
    )


if __name__ == "__main__":
    main()
