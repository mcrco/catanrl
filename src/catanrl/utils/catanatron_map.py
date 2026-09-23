from __future__ import annotations

import random
from contextlib import contextmanager
from typing import Iterator, Literal

from catanatron.models.map import CatanMap, build_map

MapType = Literal["BASE", "TOURNAMENT", "MINI"]
NumberPlacement = Literal["official_spiral", "random"]
NUMBER_PLACEMENT_CHOICES: tuple[NumberPlacement, ...] = ("official_spiral", "random")
DEFAULT_NUMBER_PLACEMENT: NumberPlacement = "official_spiral"

# Note: number_placement only affects BASE and MINI maps. Catanatron's
# build_map returns a precomputed, fully fixed board for TOURNAMENT (numbers
# and resources alike), silently ignoring the setting.


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
    number_placement: NumberPlacement = DEFAULT_NUMBER_PLACEMENT,
) -> CatanMap:
    with _temporary_random_seed(seed):
        return build_map(map_type, number_placement=number_placement)
