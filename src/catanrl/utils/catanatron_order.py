"""Give Catanatron's playable actions the same order in every process.

Catanatron builds its maritime-trade and year-of-plenty actions from sets, so
their order follows Python's per-process string hash seed and, before Python
3.12, the address of ``None`` (which ASLR moves), and pinning ``PYTHONHASHSEED``
is not enough. F breaks value ties by action order and the random bots draw by
index, so a seeded game against them played out differently in every process.
Sorting what those two builders return makes seeded Python-engine games
reproducible. Importing ``catanrl`` applies this; it only reorders actions.
"""

from catanatron.models import actions
from catanatron.models.enums import RESOURCES

_RESOURCE_RANK = {resource: rank for rank, resource in enumerate(RESOURCES)}
_unordered_maritime_trades = actions.inner_maritime_trade_possibilities
_unordered_year_of_plenty = actions.year_of_plenty_possibilities


def _resources_key(resources) -> list[int]:
    return [-1 if resource is None else _RESOURCE_RANK[resource] for resource in resources]


def inner_maritime_trade_possibilities(hand_freqdeck, bank_freqdeck, port_resources):
    offers = _unordered_maritime_trades(hand_freqdeck, bank_freqdeck, port_resources)
    return sorted(offers, key=_resources_key)


def year_of_plenty_possibilities(color, freqdeck):
    options = _unordered_year_of_plenty(color, freqdeck)
    return sorted(options, key=lambda action: _resources_key(action.value))


actions.inner_maritime_trade_possibilities = inner_maritime_trade_possibilities
actions.year_of_plenty_possibilities = year_of_plenty_possibilities
