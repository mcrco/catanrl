import random
import sys
from typing import List, Literal
from tqdm import tqdm

from catanatron.game import Game
from catanatron.models.map import build_map
from catanatron.models.player import Player
from catanatron.state_functions import get_actual_victory_points

from catanrl.players import NNPolicyPlayer


def eval(
    nn_player: NNPolicyPlayer,
    opponents: List[Player],
    map_type: Literal["BASE", "TOURNAMENT", "MINI"] = "BASE",
    num_games: int = 100,
    seed: int = 42,
    show_tqdm: bool = False,
):
    wins = 0
    vps = []
    total_vps = []
    turns = []

    players = [nn_player] + opponents
    rng = random.Random(seed)
    for _ in tqdm(range(num_games), disable=not show_tqdm):
        # Ensure each game starts from a clean player state and fresh map.
        # Reusing player internals or a previously-mutated map can skew win rates.
        for player in players:
            player.reset_state()
        game = Game(
            players=players,
            catan_map=build_map(map_type),
            seed=rng.randint(
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
