from typing import Any, List, Literal
from tqdm import tqdm

from catanatron.game import Game
from catanatron.models.player import Player
from catanatron.state_functions import get_actual_victory_points

from catanrl.utils.catanatron_game import SeatOption, build_players_for_seat, force_player_order
from catanrl.utils.catanatron_map import NumberPlacement, build_catan_map
from catanrl.utils.seeding import derive_map_and_game_seeds, derive_seed


def eval(
    nn_player: Player,
    opponents: List[Player],
    map_type: Literal["BASE", "TOURNAMENT", "MINI"] = "BASE",
    num_games: int = 100,
    seed: int = 67,
    vps_to_win: int = 15,
    discard_limit: int = 9,
    show_tqdm: bool = False,
    nn_seat: SeatOption = "random",
    game_records: list[dict[str, Any]] | None = None,
    number_placement: NumberPlacement = "official_spiral",
):
    wins = 0
    vps = []
    total_vps = []
    turns = []

    players, allow_upstream_shuffle = build_players_for_seat(nn_player, opponents, nn_seat)
    episode_seeds = [derive_seed(seed, "episode", game_idx) for game_idx in range(num_games)]
    for episode_seed in tqdm(episode_seeds, disable=not show_tqdm):
        map_seed, game_seed = derive_map_and_game_seeds(episode_seed)
        # Ensure each game starts from a clean player state and fresh map.
        # Reusing player internals or a previously-mutated map can skew win rates.
        for player in players:
            player.reset_state()
        game = Game(
            players=players,
            catan_map=build_catan_map(
                map_type,
                seed=map_seed,
                number_placement=number_placement,
            ),
            seed=game_seed,
            discard_limit=discard_limit,
            vps_to_win=vps_to_win,
        )
        if not allow_upstream_shuffle:
            force_player_order(game, players)
        game.play()
        won = game.winning_color() == nn_player.color
        if won:
            wins += 1

        nn_vps = get_actual_victory_points(game.state, nn_player.color)
        vps.append(nn_vps)
        total_vps_for_game = sum(
            get_actual_victory_points(game.state, color) for color in game.state.colors
        )
        total_vps.append(total_vps_for_game)
        turns.append(game.state.num_turns)
        if game_records is not None:
            game_records.append(
                {
                    "seat": str(nn_seat),
                    "episode_seed": int(episode_seed),
                    "win": won,
                    "vps": int(nn_vps),
                    "total_vps": int(total_vps_for_game),
                    "turns": int(game.state.num_turns),
                }
            )

    return wins, vps, total_vps, turns
