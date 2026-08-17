from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from catanatron.models.enums import RESOURCES
from catanatron.models.map import (
    BASE_MAP_TEMPLATE,
    MINI_MAP_TEMPLATE,
    CatanMap,
    build_map,
    initialize_tiles,
)
from catanatron.models.tiles import LandTile

if TYPE_CHECKING:
    from catanrl.envs.cudanatron.binding import NativeGame

MapType = Literal["BASE", "TOURNAMENT", "MINI"]


def build_catan_map_from_native_game(native_game: NativeGame, map_type: MapType) -> CatanMap:
    """Rebuild a Catanatron map from a cudanatron board.

    Python and CUDA use different RNGs, so the same numeric seed does not imply
    the same shuffled board. Reconstructing the map lets replay tests compare
    identical topology, terrain, numbers, and ports.
    """
    if map_type == "TOURNAMENT":
        return build_map("TOURNAMENT")

    template = MINI_MAP_TEMPLATE if map_type == "MINI" else BASE_MAP_TEMPLATE
    native_tiles = {(tile[0], tile[1], tile[2]): tile[3:] for tile in native_game.tiles()}
    placed_land_resources: list[str | None] = []
    placed_numbers: list[int] = []
    placed_port_resources: list[str | None] = []
    for coordinate, tile_type in template.topology.items():
        _tile_id, kind, resource, number, _direction, _nodes = native_tiles[coordinate]
        if tile_type is LandTile:
            if kind != 0:
                raise ValueError(f"Expected native land tile at {coordinate}, got kind {kind}")
            placed_land_resources.append(None if resource < 0 else RESOURCES[resource])
            if resource >= 0:
                placed_numbers.append(number)
        elif isinstance(tile_type, tuple):
            if kind != 2:
                raise ValueError(f"Expected native port at {coordinate}, got kind {kind}")
            placed_port_resources.append(None if resource < 0 else RESOURCES[resource])

    tiles = initialize_tiles(
        template,
        shuffled_port_resources_param=list(reversed(placed_port_resources)),
        shuffled_tile_resources_param=list(reversed(placed_land_resources)),
        shuffled_numbers_param=list(reversed(placed_numbers)),
        number_placement="random",
    )
    return CatanMap.from_tiles(tiles)
