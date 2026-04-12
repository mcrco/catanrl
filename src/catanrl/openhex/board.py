"""Build a CatanMap from OpenHex board JSON (tiles + ports)."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from catanatron.models.enums import FastResource
from catanatron.models.map import (
    BASE_MAP_TEMPLATE,
    LandTile,
    Port,
    initialize_tiles,
    build_map,
)
from catanatron.models.tiles import Water

from catanrl.openhex.geometry import edge_vertices_to_edge_tuple, hex_dict_to_coordinate

TERRAIN_TO_RESOURCE: Dict[str, FastResource | None] = {
    "forest": "WOOD",
    "hills": "BRICK",
    "pasture": "SHEEP",
    "fields": "WHEAT",
    "mountains": "ORE",
    "desert": None,
}


def _port_resource_from_openhex(port: Dict[str, Any]) -> FastResource | None:
    if port.get("rate") == 3:
        return None
    res = port.get("resource")
    if res is None:
        return None
    return str(res).upper()  # type: ignore[return-value]


def build_catan_map_from_openhex_board(board: Dict[str, Any]) -> CatanMap:
    """
    Reconstruct the server's board layout as a CatanMap.

    `board` is `state.board` from OpenHex (setup phase includes tiles + ports).
    """
    tiles_list: List[Dict[str, Any]] = board.get("tiles") or []
    by_coord: Dict[Tuple[int, int, int], Dict[str, Any]] = {}
    for t in tiles_list:
        c = hex_dict_to_coordinate(t["hex"])
        by_coord[c] = t

    shuffled_tile_resources: List[FastResource | None] = []
    shuffled_numbers: List[int] = []
    for coord, tile_type in BASE_MAP_TEMPLATE.topology.items():
        if tile_type is LandTile:
            info = by_coord.get(coord)
            if info is None:
                raise KeyError(f"missing land tile at {coord}")
            terrain = info["terrain"]
            res = TERRAIN_TO_RESOURCE.get(terrain)
            if res is None and terrain != "desert":
                raise ValueError(f"unknown terrain {terrain}")
            shuffled_tile_resources.append(res)
            if res is not None:
                shuffled_numbers.append(int(info["number"]))
        elif isinstance(tile_type, tuple) and tile_type[0] is Port:
            pass
        elif tile_type is Water:
            pass

    ports_list: List[Dict[str, Any]] = board.get("ports") or []
    tmp_map = build_map("BASE")
    oh_ports: List[Tuple[Tuple[int, int], FastResource | None]] = []
    for p in ports_list:
        ev = p["edge"]
        a = hex_dict_to_coordinate(ev[0])
        b = hex_dict_to_coordinate(ev[1])
        eid = edge_vertices_to_edge_tuple(tmp_map, a, b)
        oh_ports.append((eid, _port_resource_from_openhex(p)))

    shuffled_port_resources: List[FastResource | None] = []
    unused = list(oh_ports)
    for coord, tile_type in BASE_MAP_TEMPLATE.topology.items():
        if isinstance(tile_type, tuple) and tile_type[0] is Port:
            tile = tmp_map.tiles[coord]
            idx: int | None = None
            for i, (eid, _) in enumerate(unused):
                for edge_pair in tile.edges.values():
                    if tuple(sorted(edge_pair)) == tuple(sorted(eid)):
                        idx = i
                        break
                if idx is not None:
                    break
            if idx is None:
                raise RuntimeError(f"could not match OpenHex port at {coord}")
            shuffled_port_resources.append(unused.pop(idx)[1])

    # initialize_tiles pops from the end of each list (see map.initialize_tiles).
    shuffled_tile_resources.reverse()
    shuffled_numbers.reverse()
    shuffled_port_resources.reverse()

    tiles = initialize_tiles(
        BASE_MAP_TEMPLATE,
        shuffled_tile_resources_param=shuffled_tile_resources,
        shuffled_numbers_param=shuffled_numbers,
        shuffled_port_resources_param=shuffled_port_resources,
        number_placement="random",
    )
    return CatanMap.from_tiles(tiles)
