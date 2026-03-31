from __future__ import annotations

import random
from contextlib import contextmanager
from typing import Iterator, Literal

from catanatron.models.map import CatanMap, build_map

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
