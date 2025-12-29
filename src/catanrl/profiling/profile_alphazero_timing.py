#!/usr/bin/env python3
"""
Benchmark key AlphaZero steps (feature extraction, model forward, game step,
and MCTS rollouts) to identify bottlenecks.
"""

from __future__ import annotations

import argparse
import random
import time
from statistics import mean
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch

from catanatron.game import Game
from catanatron.gym.envs.catanatron_env import ACTION_SPACE_SIZE, to_action_space
from catanatron.models.map import build_map
from catanatron.models.player import Color, RandomPlayer

from catanrl.algorithms.alphazero.mcts import MCTSNode, NeuralMCTS
from catanrl.algorithms.alphazero.trainer import AlphaZeroConfig, AlphaZeroTrainer
from catanrl.features.catanatron_utils import (
    compute_feature_vector_dim,
    game_to_features,
    get_numeric_feature_names,
)
from catanrl.models.models import HierarchicalPolicyValueNetwork


def _maybe_sync(device: str | torch.device) -> None:
    if torch.cuda.is_available() and device is not None and "cuda" in str(device):
        torch.cuda.synchronize()


class ProfilingAlphaZeroTrainer(AlphaZeroTrainer):
    """
    AlphaZero trainer with lightweight timing instrumentation for feature
    extraction and network calls.
    """

    def __init__(
        self,
        config: AlphaZeroConfig,
        model: torch.nn.Module,
        mock_inference: bool = False,
    ):
        super().__init__(config=config, model=model)
        self.feature_times: List[float] = []
        self.model_times: List[float] = []
        self.policy_value_times: List[float] = []
        self._mock_inference = mock_inference

    def reset_timers(self) -> None:
        self.feature_times.clear()
        self.model_times.clear()
        self.policy_value_times.clear()

    def _policy_value(
        self,
        game: Game,
        color: Color,
        cached_features: np.ndarray | None = None,
    ) -> Tuple[np.ndarray, float]:
        if self._mock_inference:
            self.feature_times.append(0.0)
            self.model_times.append(0.0)
            self.policy_value_times.append(0.0)
            probs = np.full(ACTION_SPACE_SIZE, 1.0 / ACTION_SPACE_SIZE, dtype=np.float32)
            value_scalar = 0.0
        else:
            feature_time = 0.0
            if cached_features is None:
                t0 = time.perf_counter()
                features = game_to_features(game, color, len(self.colors), self.config.map_type)
                feature_time = time.perf_counter() - t0
            else:
                features = cached_features
            self.feature_times.append(feature_time)

            states = torch.from_numpy(features).float().unsqueeze(0).to(self.device)
            self.model.eval()
            _maybe_sync(self.device)
            t0 = time.perf_counter()
            with torch.no_grad():
                log_probs, value = self._model_forward(states)
            _maybe_sync(self.device)
            model_time = time.perf_counter() - t0
            self.model_times.append(model_time)
            self.policy_value_times.append(feature_time + model_time)

            probs = torch.exp(log_probs).cpu().numpy().squeeze(0)
            value_scalar = float(value.view(-1)[0].item())

        valid_actions = game.state.playable_actions
        if not valid_actions:
            return np.zeros_like(probs, dtype=np.float32), value_scalar

        mask = np.zeros_like(probs, dtype=np.float32)
        indices = [to_action_space(action) for action in valid_actions]
        mask[indices] = 1.0
        masked = probs * mask
        total = masked.sum()
        if total <= 0:
            masked[indices] = 1.0 / len(indices)
        else:
            masked /= total
        return masked.astype(np.float32), value_scalar


class ProfilingMCTS(NeuralMCTS):
    """Neural-guided MCTS with per-simulation timing and depth stats."""

    def __init__(self, trainer: ProfilingAlphaZeroTrainer):
        super().__init__(trainer)
        self.sim_times: List[float] = []
        self.depths: List[int] = []
        self.search_time: float = 0.0

    def search(
        self,
        game: Game,
        root_features: np.ndarray,
        temperature: float,
        add_noise: bool,
    ) -> Tuple[np.ndarray, object]:
        start_time = time.perf_counter()
        root_game = game.copy()
        root = MCTSNode(
            game=root_game,
            to_play=root_game.state.current_color(),
            parent=None,
            prior=1.0,
        )
        priors, _ = self.trainer._policy_value(
            root.game, root.to_play, cached_features=root_features
        )
        root.expand(priors)
        if add_noise:
            root.add_exploration_noise(
                self.trainer.config.dirichlet_alpha,
                self.trainer.config.dirichlet_frac,
            )

        for _ in range(self.trainer.config.simulations):
            sim_start = time.perf_counter()
            node = root
            search_path = [node]
            while not node.is_terminal():
                if self._is_chance_node(node):
                    node = self._sample_chance_child(node)
                    search_path.append(node)
                    continue

                if not node.expanded():
                    break

                _, node = node.select_child(self.trainer.config.c_puct)
                search_path.append(node)

            value = self._evaluate(node)
            self._backpropagate(search_path, value)
            self.depths.append(len(search_path) - 1)
            self.sim_times.append(time.perf_counter() - sim_start)

        policy = self._build_policy_vector(root, temperature)
        action_idx = self._sample_action(root, policy)
        action = root.action_map[action_idx]
        self.search_time = time.perf_counter() - start_time
        return policy, action


def bench_game_step(
    trainer: ProfilingAlphaZeroTrainer,
    steps: int,
    seed: int,
) -> float:
    rng = random.Random(seed)
    players = [RandomPlayer(c) for c in trainer.colors]
    game = Game(players, seed=seed, catan_map=build_map(trainer.config.map_type))
    times: List[float] = []
    for _ in range(steps):
        if game.winning_color() is not None or not game.state.playable_actions:
            game = Game(
                players, seed=rng.randrange(1 << 30), catan_map=build_map(trainer.config.map_type)
            )
        action = rng.choice(game.state.playable_actions)
        t0 = time.perf_counter()
        game.execute(action)
        times.append(time.perf_counter() - t0)
    return float(mean(times))


def bench_mcts(
    trainer: ProfilingAlphaZeroTrainer,
    sim_counts: Iterable[int],
    runs_per_setting: int,
    seed: int,
) -> List[Dict[str, float]]:
    rng = random.Random(seed)
    results: List[Dict[str, float]] = []
    base_simulations = trainer.config.simulations

    for sims in sim_counts:
        trainer.config.simulations = sims
        search_times: List[float] = []
        per_sim_times: List[float] = []
        depths: List[int] = []
        policy_value_times: List[float] = []
        for _ in range(runs_per_setting):
            trainer.reset_timers()
            mcts = ProfilingMCTS(trainer)
            players = [RandomPlayer(c) for c in trainer.colors]
            game = Game(
                players, seed=rng.randrange(1 << 30), catan_map=build_map(trainer.config.map_type)
            )
            root_feats = game_to_features(
                game, trainer.colors[0], len(trainer.colors), trainer.config.map_type
            )
            mcts.search(game, root_feats, temperature=1.0, add_noise=False)
            search_times.append(mcts.search_time)
            per_sim_times.extend(mcts.sim_times)
            depths.extend(mcts.depths)
            policy_value_times.extend(trainer.policy_value_times)

        results.append(
            {
                "simulations": sims,
                "search_ms": float(mean(search_times) * 1000.0),
                "per_sim_ms": float(mean(per_sim_times) * 1000.0) if per_sim_times else 0.0,
                "avg_depth": float(mean(depths)) if depths else 0.0,
                "max_depth": float(max(depths)) if depths else 0.0,
                "policy_value_ms": float(mean(policy_value_times) * 1000.0)
                if policy_value_times
                else 0.0,
            }
        )

    trainer.config.simulations = base_simulations
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile AlphaZero pipeline timings",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--map-type", type=str, default="BASE", choices=["BASE", "MINI", "TOURNAMENT"]
    )
    parser.add_argument("--num-players", type=int, default=2, choices=[2, 3, 4])
    parser.add_argument("--hidden-dims", type=str, default="2048,2048,1024,1024")
    parser.add_argument("--device", type=str, default="cuda", help="cpu, cuda, or cuda:0 style")
    parser.add_argument(
        "--batch-sizes",
        type=str,
        default="1,4,16,128,1024",
        help="Comma-separated batch sizes for model forward timing",
    )
    parser.add_argument(
        "--mcts-sims",
        type=str,
        default="16,64,128,256",
        help="Comma-separated simulation counts for MCTS timing",
    )
    parser.add_argument(
        "--forward-repeats", type=int, default=50, help="Timed repeats per batch size"
    )
    parser.add_argument(
        "--forward-warmup", type=int, default=10, help="Warmup iterations per batch size"
    )
    parser.add_argument(
        "--feature-samples", type=int, default=64, help="Number of feature extractions to average"
    )
    parser.add_argument(
        "--step-samples", type=int, default=200, help="Number of game steps to time"
    )
    parser.add_argument(
        "--mcts-runs", type=int, default=3, help="Search runs per simulation setting"
    )
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")
    parser.add_argument(
        "--mock-inference",
        action="store_true",
        help="Skip real model forward passes and use uniform policies to isolate env/MCTS speed",
    )
    return parser.parse_args()


def parse_int_list(value: str) -> List[int]:
    return [int(v.strip()) for v in value.split(",") if v.strip()]


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    hidden_dims = parse_int_list(args.hidden_dims)
    batch_sizes = parse_int_list(args.batch_sizes)
    mcts_sims = parse_int_list(args.mcts_sims)

    config = AlphaZeroConfig(
        num_players=args.num_players,
        map_type=args.map_type,
        device=device,
        seed=args.seed,
    )
    input_dim = compute_feature_vector_dim(config.num_players, config.map_type)
    model = HierarchicalPolicyValueNetwork(input_dim=input_dim, hidden_dims=hidden_dims)
    trainer = ProfilingAlphaZeroTrainer(
        config=config,
        model=model,
        mock_inference=args.mock_inference,
    )

    print("== AlphaZero timing profile ==")
    print(f"Device: {trainer.device}")
    print(f"Map: {config.map_type} | Players: {config.num_players}")
    print(f"Input dim: {input_dim} | Hidden dims: {hidden_dims}")

    print("\nModel forward (HierarchicalPolicyValueNetwork):")
    forward_stats = bench_model_forward(
        trainer=trainer,
        input_dim=input_dim,
        batch_sizes=batch_sizes,
        repeats=args.forward_repeats,
        warmup=args.forward_warmup,
    )
    for stat in forward_stats:
        print(
            f"  batch={stat['batch_size']:>4} | "
            f"per_batch={stat['per_batch_ms']:.3f} ms | "
            f"per_item={stat['per_item_ms']:.4f} ms"
        )

    print("\nFeature extraction:")
    feat_ms = (
        bench_feature_extraction(trainer, num_samples=args.feature_samples, seed=args.seed) * 1000.0
    )
    print(f"  avg per state: {feat_ms:.3f} ms over {args.feature_samples} samples")

    print("\nGame step execution:")
    step_ms = bench_game_step(trainer, steps=args.step_samples, seed=args.seed) * 1000.0
    print(f"  avg execute(): {step_ms:.3f} ms over {args.step_samples} steps")

    print("\nMCTS search (includes feature + network calls during evaluation):")
    mcts_stats = bench_mcts(
        trainer=trainer,
        sim_counts=mcts_sims,
        runs_per_setting=args.mcts_runs,
        seed=args.seed,
    )
    for stat in mcts_stats:
        print(
            f"  sims={stat['simulations']:>4} | "
            f"search={stat['search_ms']:.2f} ms | "
            f"per_sim={stat['per_sim_ms']:.3f} ms | "
            f"policy_value={stat['policy_value_ms']:.3f} ms | "
            f"avg_depth={stat['avg_depth']:.2f} (max {stat['max_depth']:.0f})"
        )


if __name__ == "__main__":
    main()
