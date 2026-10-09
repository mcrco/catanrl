import os
import subprocess
import sys

# Maritime trades and year-of-plenty picks, as Catanatron lists them to players.
PROGRAM = """
import catanrl
from catanatron.models.actions import (
    inner_maritime_trade_possibilities,
    year_of_plenty_possibilities,
)
from catanatron.models.player import Color

print(list(inner_maritime_trade_possibilities([4, 3, 2, 4, 0], [19] * 5, {None, "ORE"})))
print([action.value for action in year_of_plenty_possibilities(Color.RED, [19, 1, 0, 19, 2])])
"""


def _action_order(hash_seed: str) -> str:
    env = dict(os.environ, PYTHONHASHSEED=hash_seed)
    return subprocess.run(
        [sys.executable, "-c", PROGRAM], env=env, capture_output=True, text=True, check=True
    ).stdout


def test_action_order_does_not_depend_on_the_process():
    orders = {_action_order(seed) for seed in ("1", "2", "random", "random")}
    assert len(orders) == 1
