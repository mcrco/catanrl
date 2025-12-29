import argparse
import random
from typing import List, Literal
import time

from catanatron.game import Game
from catanatron.models.player import RandomPlayer
from catanatron.models.map import build_map

from catanrl.features.catanatron_utils import COLOR_ORDER
from statistics import mean


def bench_game_step(
    map_type: Literal["BASE", "TOURNAMENT", "MINI"],
    steps: int,
    num_players: int,
    turns_limit: int,
    seed: int,
) -> float:
    rng = random.Random(seed)
    players = [RandomPlayer(c) for c in COLOR_ORDER[:num_players]]
    game = Game(players, seed=seed, catan_map=build_map(map_type))
    times: List[float] = []
    for _ in range(steps):
        if game.winning_color() is not None or game.state.num_turns >= turns_limit:
            seed += 1
            game = Game(players, seed=seed, catan_map=build_map(map_type))
        action = rng.choice(game.state.playable_actions)
        t0 = time.perf_counter()
        game.execute(action)
        times.append(time.perf_counter() - t0)
    return float(mean(times))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Profile Catanatron")
    parser.add_argument("--num-steps", type=int, default=100000, help="Number of games to play")
    parser.add_argument(
        "--map-type",
        type=str,
        default="BASE",
        choices=["BASE", "TOURNAMENT", "MINI"],
        help="Map type",
    )
    parser.add_argument("--num-players", type=int, default=2, help="Number of players")
    parser.add_argument("--turns-limit", type=int, default=1000, help="Turns limit")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    average_time = bench_game_step(
        map_type=args.map_type,
        steps=args.num_steps,
        num_players=args.num_players,
        turns_limit=args.turns_limit,
        seed=args.seed,
    )
    print(f"Average time per step: {average_time * 1000:.4f} ms")
