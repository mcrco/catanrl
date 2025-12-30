import argparse
from typing import List, Literal, Dict, Union
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
) -> Dict[str, Union[float, List[int]]]:
    players = [RandomPlayer(c) for c in COLOR_ORDER[:num_players]]
    game = Game(players, seed=seed, catan_map=build_map(map_type))
    times: List[float] = []
    for _ in range(steps):
        if game.winning_color() is not None or game.state.num_turns >= turns_limit:
            seed += 1
            game = Game(players, seed=seed, catan_map=build_map(map_type))
        t0 = time.perf_counter()
        game.play_tick()
        times.append(time.perf_counter() - t0)
    return {
        "min_time": min(times),
        "average_time": float(mean(times)),
        "max_time": max(times),
    }


def bench_games(
    map_type: Literal["BASE", "TOURNAMENT", "MINI"],
    games: int,
    num_players: int,
    seed: int,
) -> Dict[str, Union[float, List[int]]]:
    """Benchmarks the time it takes to initialize and play games."""
    players = [RandomPlayer(c) for c in COLOR_ORDER[:num_players]]
    game = Game(players, seed=seed, catan_map=build_map(map_type))
    times: List[float] = []
    turns: List[int] = []
    for _ in range(games):
        t0 = time.perf_counter()
        game = Game(players, seed=seed, catan_map=build_map(map_type))
        game.play()
        times.append(time.perf_counter() - t0)
        turns.append(game.state.num_turns)
        seed += 1
    return {
        "min_time": min(times),
        "average_time": float(mean(times)),
        "max_time": max(times),
        "turns": turns,
    }


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
    parser.add_argument("--num-games", type=int, default=1000, help="Number of games to play")
    args = parser.parse_args()

    step_results = bench_game_step(
        map_type=args.map_type,
        steps=args.num_steps,
        num_players=args.num_players,
        turns_limit=args.turns_limit,
        seed=args.seed,
    )
    print(f"Average time per step: {step_results['average_time'] * 1000:.4f} ms")
    print(f"Min time per step: {step_results['min_time'] * 1000:.4f} ms")
    print(f"Max time per step: {step_results['max_time'] * 1000:.4f} ms")

    game_results = bench_games(
        map_type=args.map_type,
        games=args.num_games,
        num_players=args.num_players,
        seed=args.seed,
    )
    print(f"Average time per game: {game_results['average_time'] * 1000:.4f} ms")
    print(f"Min time per game: {game_results['min_time'] * 1000:.4f} ms")
    print(f"Max time per game: {game_results['max_time'] * 1000:.4f} ms")
    print(f"Average turns per game: {float(mean(game_results['turns']))}")
    print(f"Min turns per game: {int(min(game_results['turns']))}")
    print(f"Max turns per game: {int(max(game_results['turns']))}")
