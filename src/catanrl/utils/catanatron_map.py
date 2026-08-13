from __future__ import annotations

import random
from contextlib import contextmanager
from typing import TYPE_CHECKING, Iterator, Literal

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
    from catanrl.envs.cppanatron import NativeGame

MapType = Literal["BASE", "TOURNAMENT", "MINI"]
NumberPlacement = Literal["official_spiral", "random"]


@contextmanager
def _temporary_random_seed(seed: int | None) -> Iterator[None]:
    if seed is None:
        yield
        return

    state = random.getstate()
    random.seed(seed)
    try:
        yield
    finally:
        random.setstate(state)


def build_catan_map(
    map_type: MapType,
    *,
    seed: int | None = None,
    number_placement: NumberPlacement = "random",
) -> CatanMap:
    with _temporary_random_seed(seed):
        return build_map(map_type, number_placement=number_placement)


def build_catan_map_from_native_game(
    native_game: "NativeGame",
    map_type: MapType,
) -> CatanMap:
    """Reconstruct an exact Catanatron map from a cppanatron game layout.

    Python and C++ use different random-number implementations, so the same
    numeric map seed does not imply the same shuffled board.  This conversion
    lets Catanatron remain the authoritative game while a native search shadow
    starts from literally identical topology, terrain, numbers, and ports.
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
