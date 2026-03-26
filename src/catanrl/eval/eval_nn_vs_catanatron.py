from typing import List, Literal
from tqdm import tqdm

from catanatron.game import Game
from catanatron.models.map import build_map
from catanatron.models.player import Player
from catanatron.state_functions import get_actual_victory_points

from catanrl.utils.seeding import derive_map_and_game_seeds, derive_seed


def eval(
    nn_player: Player,
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
    episode_seeds = [derive_seed(seed, "episode", game_idx) for game_idx in range(num_games)]
    for episode_seed in tqdm(episode_seeds, disable=not show_tqdm):
        map_seed, game_seed = derive_map_and_game_seeds(episode_seed)
        # Ensure each game starts from a clean player state and fresh map.
        # Reusing player internals or a previously-mutated map can skew win rates.
        for player in players:
            player.reset_state()
        game = Game(
            players=players,
            catan_map=build_map(map_type, seed=map_seed),
            seed=game_seed,
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
