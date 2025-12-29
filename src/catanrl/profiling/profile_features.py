import argparse
import random
from typing import List, Callable, Literal
from statistics import mean
import time

from catanatron.game import Game
from catanatron.models.player import Color
from catanatron.models.player import RandomPlayer
from catanatron.models.map import build_map
import numpy as np

from catanrl.features.catanatron_utils import COLOR_ORDER, game_to_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile Neural Network and feature extraction",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--map-type",
        type=Literal["BASE", "MINI", "TOURNAMENT"],
        default="BASE",
        choices=["BASE", "MINI", "TOURNAMENT"],
    )
    parser.add_argument("--hidden-dims", type=str, default="512,512")
    parser.add_argument("--device", type=str, default="cuda", help="cpu, cuda, or cuda:0 style")
    parser.add_argument(
        "--batch-sizes",
        type=str,
        default="1,4,16,32,64,128,256,512,1024,4096,8192,16384,32768",
        help="Comma-separated batch sizes for model forward timing",
    )
    parser.add_argument(
        "--forward-repeats", type=int, default=50, help="Timed repeats per batch size"
    )
    parser.add_argument(
        "--forward-warmup", type=int, default=10, help="Warmup iterations per batch size"
    )
    parser.add_argument(
        "--feature-samples",
        type=int,
        default=10000,
        help="Number of feature extractions to average",
    )
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")
    return parser.parse_args()


def bench_feature_extraction(
    feature_extractor: Callable[
        [Game, Color, int, Literal["BASE", "MINI", "TOURNAMENT"]], np.ndarray
    ],
    map_type: Literal["BASE", "MINI", "TOURNAMENT"],
    num_samples: int,
    seed: int,
) -> float:
    rng = random.Random(seed)
    players = [RandomPlayer(c) for c in COLOR_ORDER]
    game = Game(players, seed=seed, catan_map=build_map(map_type))
    times: List[float] = []
    for _ in range(num_samples):
        if game.winning_color() is not None or not game.state.playable_actions:
            seed += 1
            game = Game(players, seed=seed, catan_map=build_map(map_type))
        color = COLOR_ORDER[rng.randrange(len(COLOR_ORDER))]
        t0 = time.perf_counter()
        _ = feature_extractor(game, color, len(COLOR_ORDER), map_type)
        times.append(time.perf_counter() - t0)
        _ = game.play_tick()
    return float(mean(times))


if __name__ == "__main__":
    args = parse_args()
    average_time = bench_feature_extraction(
        feature_extractor=game_to_features,
        map_type=args.map_type,
        num_samples=args.feature_samples,
        seed=args.seed,
    )
    print(f"Average time per feature extraction: {average_time * 1000:.4f} ms")
