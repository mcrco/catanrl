from typing import List, Literal
import random
import sys

from catanatron.game import Game
from catanatron.models.player import Player
from catanatron.models.map import build_map
from catanatron.players.value import ValueFunctionPlayer
from catanatron.state_functions import get_actual_victory_points

from catanrl.players import create_nn_policy_player, NNPolicyPlayer
from catanrl.features.catanatron_utils import COLOR_ORDER
from catanrl.models import BackboneConfig, MLPBackboneConfig

from tqdm import tqdm


def eval(
    nn_player: NNPolicyPlayer,
    opponents: List[Player],
    map_type: Literal["BASE", "TOURNAMENT", "MINI"] = "BASE",
    num_games: int = 100,
    seed: int = 42,
):
    wins = 0
    vps = []
    total_vps = []
    turns = []

    players = [nn_player] + opponents
    map = build_map(map_type)
    random.seed(seed)
    for _ in range(num_games):
        game = Game(
            players=players,
            catan_map=map,
            seed=random.randint(
                0,
                sys.maxsize,
            ),
        )
        game.play()
        if game.winning_color() == nn_player.color:
            wins += 1

        vps.append(get_actual_victory_points(game.state, nn_player.color))
        total_vps_for_game = sum(
            get_actual_victory_points(game.state, color) for color in game.state.colors
        )
        total_vps.append(total_vps_for_game)
        turns.append(game.state.num_turns)

    return wins, vps, total_vps, turns
