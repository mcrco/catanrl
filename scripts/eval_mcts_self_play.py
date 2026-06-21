"""
Evaluate a neural-guided MCTS player in self-play.

Loads a policy network and a critic network, uses the critic as the MCTS value
network, and seats a separate NNMCTSPlayer instance at every player color.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import queue
import random
import traceback
from dataclasses import dataclass, field
from typing import Literal, Sequence

import numpy as np
import torch
import wandb
from catanatron.game import Game
from catanatron.models.player import Color
from catanatron.state_functions import get_actual_victory_points
from tqdm import tqdm

from catanrl.envs.puffer.common import compute_single_agent_dims
from catanrl.experiment_store import (
    KIND_POLICY_VALUE,
    backbone_display_type,
    backbone_hidden_dims,
    load_experiment,
)
from catanrl.experiments.common_args import DEFAULT_WANDB_PROJECT
from catanrl.experiments.network_config import resolve_observation_network_args
from catanrl.features.catanatron_utils import (
    ActorObservationLevel,
    COLOR_ORDER,
    CriticObservationLevel,
)
from catanrl.models.backbones import (
    BackboneConfig,
    CrossDimensionalBackboneConfig,
    MLPBackboneConfig,
)
from catanrl.models.models import (
    build_flat_policy_network,
    build_hierarchical_policy_network,
    build_value_network,
)
from catanrl.models.wrappers import PolicyNetworkWrapper, ValueNetworkWrapper
from catanrl.players import NNMCTSPlayer
from catanrl.players.nn_mcts_player import (
    _CentralNNMCTSInferenceServer,
    _RemoteNNMCTSInferenceBackend,
)
from catanrl.utils.catanatron_action_space import get_action_space_size
from catanrl.utils.catanatron_game import color_label, force_player_order
from catanrl.utils.catanatron_map import build_catan_map
from catanrl.utils.seeding import derive_map_and_game_seeds, derive_seed

BOARD_WIDTH = 21
BOARD_HEIGHT = 11


@dataclass
class PlayerStats:
    wins: int = 0
    vps: list[int] = field(default_factory=list)

    @property
    def avg_vps(self) -> float:
        return sum(self.vps) / len(self.vps) if self.vps else 0.0


def build_policy_model(
    backbone_type: str,
    model_type: str,
    hidden_dims: Sequence[int],
    num_players: int,
    map_type: Literal["BASE", "MINI", "TOURNAMENT"],
    actor_observation_level: ActorObservationLevel,
    device: torch.device,
) -> PolicyNetworkWrapper:
    dims = compute_single_agent_dims(
        num_players,
        map_type,
        actor_observation_level=actor_observation_level,
    )
    actor_dim = dims["actor_dim"]
    numeric_dim = dims["numeric_dim"]
    board_channels = dims["board_channels"]

    if backbone_type == "mlp":
        backbone_config = BackboneConfig(
            architecture="mlp",
            args=MLPBackboneConfig(input_dim=actor_dim, hidden_dims=list(hidden_dims)),
        )
    elif backbone_type == "xdim":
        backbone_config = BackboneConfig(
            architecture="cross_dimensional",
            args=CrossDimensionalBackboneConfig(
                board_height=BOARD_HEIGHT,
                board_width=BOARD_WIDTH,
                board_channels=board_channels,
                numeric_dim=numeric_dim,
                cnn_channels=[64, 128, 128],
                cnn_kernel_size=(3, 5),
                numeric_hidden_dims=list(hidden_dims),
                fusion_hidden_dim=hidden_dims[-1] if hidden_dims else 256,
                output_dim=hidden_dims[-1] if hidden_dims else 256,
            ),
        )
    else:
        raise ValueError(f"Unknown backbone_type '{backbone_type}'")

    if model_type == "flat":
        model = build_flat_policy_network(
            backbone_config=backbone_config,
            num_actions=get_action_space_size(num_players, map_type),
        )
    elif model_type == "hierarchical":
        model = build_hierarchical_policy_network(
            backbone_config=backbone_config,
            num_players=num_players,
            map_type=map_type,
        )
    else:
        raise ValueError(f"Unknown model_type '{model_type}'")

    return model.to(device)


def build_critic_model(
    backbone_type: str,
    hidden_dims: Sequence[int],
    num_players: int,
    map_type: Literal["BASE", "MINI", "TOURNAMENT"],
    critic_observation_level: CriticObservationLevel,
    device: torch.device,
) -> ValueNetworkWrapper:
    dims = compute_single_agent_dims(
        num_players,
        map_type,
        critic_observation_level=critic_observation_level,
    )
    critic_dim = dims["critic_dim"]
    critic_numeric_dim = dims["critic_numeric_dim"]
    board_channels = dims["board_channels"]

    if backbone_type == "mlp":
        backbone_config = BackboneConfig(
            architecture="mlp",
            args=MLPBackboneConfig(input_dim=critic_dim, hidden_dims=list(hidden_dims)),
        )
    elif backbone_type == "xdim":
        backbone_config = BackboneConfig(
            architecture="cross_dimensional",
            args=CrossDimensionalBackboneConfig(
                board_height=BOARD_HEIGHT,
                board_width=BOARD_WIDTH,
                board_channels=board_channels,
                numeric_dim=critic_numeric_dim,
                cnn_channels=[64, 128, 128],
                cnn_kernel_size=(3, 5),
                numeric_hidden_dims=list(hidden_dims),
                fusion_hidden_dim=hidden_dims[-1] if hidden_dims else 256,
                output_dim=hidden_dims[-1] if hidden_dims else 256,
            ),
        )
    else:
        raise ValueError(f"Unknown backbone_type '{backbone_type}'")

    model = build_value_network(backbone_config=backbone_config)
    return model.to(device)


def build_self_play_players(
    policy_model: PolicyNetworkWrapper | None,
    critic_model: ValueNetworkWrapper | None,
    model_type: str,
    map_type: Literal["BASE", "MINI", "TOURNAMENT"],
    num_players: int,
    num_simulations: int,
    c_puct: float,
    prunning: bool,
    actor_observation_level: ActorObservationLevel,
    critic_observation_level: CriticObservationLevel,
    ismcts_determinizations: int = 1,
    num_search_workers: int = 1,
    inference_batch_size: int = 32,
    inference_wait_ms: float = 1.0,
    virtual_loss: float = 1.0,
    inference_backend=None,
) -> list[NNMCTSPlayer]:
    return [
        NNMCTSPlayer(
            color=color,
            model_type=model_type,
            policy_model=policy_model,
            critic_model=critic_model,
            map_type=map_type,
            num_simulations=num_simulations,
            ismcts_determinizations=ismcts_determinizations,
            c_puct=c_puct,
            prunning=prunning,
            opponent_policy="self",
            actor_observation_level=actor_observation_level,
            critic_observation_level=critic_observation_level,
            num_search_workers=num_search_workers,
            inference_batch_size=inference_batch_size,
            inference_wait_ms=inference_wait_ms,
            virtual_loss=virtual_loss,
            inference_backend=inference_backend,
        )
        for color in COLOR_ORDER[:num_players]
    ]


def _assign_episode_indices(num_games: int, num_workers: int) -> list[list[int]]:
    assignments: list[list[int]] = [[] for _ in range(max(1, num_workers))]
    for game_idx in range(num_games):
        assignments[game_idx % len(assignments)].append(game_idx)
    return assignments


def _empty_serialized_stats(num_players: int) -> dict[str, dict[str, list[int] | int]]:
    return {color.value: {"wins": 0, "vps": []} for color in COLOR_ORDER[:num_players]}


def _serialize_self_play_result(
    stats: dict[Color, PlayerStats],
    turns: list[int],
) -> dict[str, object]:
    return {
        "stats": {
            color.value: {"wins": player_stats.wins, "vps": list(player_stats.vps)}
            for color, player_stats in stats.items()
        },
        "turns": list(turns),
    }


def _merge_serialized_result(
    aggregate: dict[str, dict[str, list[int] | int]],
    turns: list[int],
    result: dict[str, object],
) -> None:
    result_stats = result["stats"]
    if not isinstance(result_stats, dict):
        raise TypeError("Worker result stats must be a dictionary.")
    for color_label, player_stats in result_stats.items():
        target = aggregate[color_label]
        if not isinstance(player_stats, dict):
            raise TypeError("Worker player stats must be a dictionary.")
        target["wins"] = int(target["wins"]) + int(player_stats["wins"])
        target["vps"].extend(player_stats["vps"])
    turns.extend(result["turns"])


def _deserialize_player_stats(
    aggregate: dict[str, dict[str, list[int] | int]],
    num_players: int,
) -> dict[Color, PlayerStats]:
    stats: dict[Color, PlayerStats] = {}
    for color in COLOR_ORDER[:num_players]:
        serialized = aggregate[color.value]
        stats[color] = PlayerStats(
            wins=int(serialized["wins"]),
            vps=list(serialized["vps"]),
        )
    return stats


def run_self_play_eval(
    players: list[NNMCTSPlayer],
    map_type: Literal["BASE", "MINI", "TOURNAMENT"],
    num_games: int,
    seed: int,
    vps_to_win: int,
    discard_limit: int,
    show_tqdm: bool = True,
) -> tuple[dict[Color, PlayerStats], list[int]]:
    stats = {player.color: PlayerStats() for player in players}
    turns: list[int] = []
    episode_seeds = [derive_seed(seed, "self_play_episode", game_idx) for game_idx in range(num_games)]

    for episode_seed in tqdm(episode_seeds, disable=not show_tqdm):
        map_seed, game_seed = derive_map_and_game_seeds(episode_seed)
        for player in players:
            player.reset_state()

        game = Game(
            players=players,
            catan_map=build_catan_map(map_type, seed=map_seed, number_placement="random"),
            seed=game_seed,
            discard_limit=discard_limit,
            vps_to_win=vps_to_win,
        )
        force_player_order(game, players)
        game.play()

        winning_color = game.winning_color()
        if winning_color in stats:
            stats[winning_color].wins += 1

        for color in game.state.colors:
            stats[color].vps.append(get_actual_victory_points(game.state, color))
        turns.append(game.state.num_turns)

    return stats, turns


def _run_self_play_episode_indices(
    players: list[NNMCTSPlayer],
    map_type: Literal["BASE", "MINI", "TOURNAMENT"],
    episode_indices: Sequence[int],
    seed: int,
    vps_to_win: int,
    discard_limit: int,
) -> tuple[dict[Color, PlayerStats], list[int]]:
    stats = {player.color: PlayerStats() for player in players}
    turns: list[int] = []

    for game_idx in episode_indices:
        episode_seed = derive_seed(seed, "self_play_episode", game_idx)
        random.seed(episode_seed)
        np.random.seed(episode_seed % (2**32))
        torch.manual_seed(episode_seed)
        map_seed, game_seed = derive_map_and_game_seeds(episode_seed)
        for player in players:
            player.reset_state()

        game = Game(
            players=players,
            catan_map=build_catan_map(map_type, seed=map_seed, number_placement="random"),
            seed=game_seed,
            discard_limit=discard_limit,
            vps_to_win=vps_to_win,
        )
        force_player_order(game, players)
        game.play()

        winning_color = game.winning_color()
        if winning_color in stats:
            stats[winning_color].wins += 1

        for color in game.state.colors:
            stats[color].vps.append(get_actual_victory_points(game.state, color))
        turns.append(game.state.num_turns)

    return stats, turns


def _self_play_worker_main(
    worker_id: int,
    episode_indices: Sequence[int],
    args_dict: dict[str, object],
    request_queue: mp.Queue,
    response_queue: mp.Queue,
    result_queue: mp.Queue,
) -> None:
    inference_backend = _RemoteNNMCTSInferenceBackend(
        worker_id=worker_id,
        request_queue=request_queue,
        response_queue=response_queue,
    )
    try:
        players = build_self_play_players(
            policy_model=None,
            critic_model=None,
            model_type=str(args_dict["model_type"]),
            map_type=args_dict["map_type"],
            num_players=int(args_dict["num_players"]),
            num_simulations=int(args_dict["num_simulations"]),
            c_puct=float(args_dict["c_puct"]),
            prunning=bool(args_dict["prunning"]),
            actor_observation_level=args_dict["actor_observation_level"],
            critic_observation_level=args_dict["critic_observation_level"],
            ismcts_determinizations=int(args_dict["ismcts_determinizations"]),
            num_search_workers=int(args_dict["num_search_workers"]),
            inference_batch_size=int(args_dict["inference_batch_size"]),
            inference_wait_ms=float(args_dict["inference_wait_ms"]),
            virtual_loss=float(args_dict["virtual_loss"]),
            inference_backend=inference_backend,
        )
        stats, turns = _run_self_play_episode_indices(
            players=players,
            map_type=args_dict["map_type"],
            episode_indices=episode_indices,
            seed=int(args_dict["seed"]),
            vps_to_win=int(args_dict["vps_to_win"]),
            discard_limit=int(args_dict["discard_limit"]),
        )
        result_queue.put(
            {
                "worker_id": worker_id,
                "games": len(episode_indices),
                "result": _serialize_self_play_result(stats, turns),
            }
        )
    except BaseException:
        result_queue.put(
            {
                "worker_id": worker_id,
                "error": traceback.format_exc(),
            }
        )
    finally:
        inference_backend.close()


def run_parallel_self_play_eval(
    policy_model: PolicyNetworkWrapper,
    critic_model: ValueNetworkWrapper,
    model_type: str,
    map_type: Literal["BASE", "MINI", "TOURNAMENT"],
    num_players: int,
    num_games: int,
    seed: int,
    vps_to_win: int,
    discard_limit: int,
    num_simulations: int,
    num_search_workers: int,
    inference_batch_size: int,
    inference_wait_ms: float,
    virtual_loss: float,
    c_puct: float,
    prunning: bool,
    actor_observation_level: ActorObservationLevel,
    critic_observation_level: CriticObservationLevel,
    num_game_workers: int,
    device: torch.device,
    ismcts_determinizations: int = 1,
    show_tqdm: bool = True,
) -> tuple[dict[Color, PlayerStats], list[int]]:
    if num_games <= 0:
        return _deserialize_player_stats(_empty_serialized_stats(num_players), num_players), []

    assignments = [
        assignment
        for assignment in _assign_episode_indices(num_games, num_game_workers)
        if assignment
    ]
    mp_ctx = mp.get_context("spawn")
    request_queue: mp.Queue = mp_ctx.Queue(maxsize=max(1024, inference_batch_size * len(assignments) * 4))
    response_queues: list[mp.Queue] = [mp_ctx.Queue(maxsize=512) for _ in assignments]
    result_queue: mp.Queue = mp_ctx.Queue()
    inference_server = _CentralNNMCTSInferenceServer(
        policy_model=policy_model,
        critic_model=critic_model,
        model_type=model_type,
        device=device,
        request_queue=request_queue,
        response_queues=response_queues,
        max_batch_size=inference_batch_size,
        max_wait_ms=inference_wait_ms,
    )

    args_dict = {
        "model_type": model_type,
        "map_type": map_type,
        "num_players": num_players,
        "num_simulations": num_simulations,
        "num_search_workers": num_search_workers,
        "inference_batch_size": inference_batch_size,
        "inference_wait_ms": inference_wait_ms,
        "virtual_loss": virtual_loss,
        "c_puct": c_puct,
        "prunning": prunning,
        "actor_observation_level": actor_observation_level,
        "critic_observation_level": critic_observation_level,
        "ismcts_determinizations": ismcts_determinizations,
        "seed": seed,
        "vps_to_win": vps_to_win,
        "discard_limit": discard_limit,
    }

    processes = [
        mp_ctx.Process(
            target=_self_play_worker_main,
            args=(
                worker_id,
                assignment,
                args_dict,
                request_queue,
                response_queues[worker_id],
                result_queue,
            ),
        )
        for worker_id, assignment in enumerate(assignments)
    ]

    aggregate = _empty_serialized_stats(num_players)
    turns: list[int] = []
    pending_workers = set(range(len(processes)))
    progress = tqdm(total=num_games, disable=not show_tqdm)
    first_error: str | None = None

    inference_server.start()
    try:
        for process in processes:
            process.start()

        while pending_workers:
            try:
                message = result_queue.get(timeout=0.5)
            except queue.Empty:
                for worker_id in list(pending_workers):
                    exitcode = processes[worker_id].exitcode
                    if exitcode is not None:
                        first_error = (
                            f"Worker {worker_id} exited with code {exitcode} "
                            "before reporting a result."
                        )
                        pending_workers.remove(worker_id)
                        break
                if first_error is not None:
                    break
                continue

            worker_id = int(message["worker_id"])
            if worker_id not in pending_workers:
                continue
            pending_workers.remove(worker_id)
            if "error" in message:
                first_error = str(message["error"])
                break
            result = message["result"]
            if not isinstance(result, dict):
                raise TypeError("Worker result payload must be a dictionary.")
            _merge_serialized_result(aggregate, turns, result)
            progress.update(int(message.get("games", 0)))
    finally:
        progress.close()
        if first_error is not None:
            for process in processes:
                if process.is_alive():
                    process.terminate()
        for process in processes:
            process.join(timeout=5.0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=1.0)
        inference_server.stop()

    if first_error is not None:
        raise RuntimeError(f"Parallel self-play worker failed:\n{first_error}")

    return _deserialize_player_stats(aggregate, num_players), turns


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a neural-guided MCTS policy/critic in self-play."
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="flat",
        choices=["flat", "hierarchical"],
        help="Policy head type: 'flat' or 'hierarchical'",
    )
    parser.add_argument(
        "--backbone-type",
        type=str,
        default="mlp",
        choices=["mlp", "xdim"],
        help="Backbone architecture for both policy and critic",
    )
    parser.add_argument(
        "--policy-hidden-dims",
        type=int,
        nargs="+",
        default=[512, 512],
        help="Hidden layer sizes for policy backbone",
    )
    parser.add_argument(
        "--critic-hidden-dims",
        type=int,
        nargs="+",
        default=[512, 512],
        help="Hidden layer sizes for critic backbone",
    )
    parser.add_argument(
        "--num-players",
        type=int,
        default=2,
        choices=[2, 3, 4],
        help="Number of self-play players",
    )
    parser.add_argument(
        "--map-type",
        type=str,
        default="BASE",
        choices=["BASE", "MINI", "TOURNAMENT"],
        help="Catan map type",
    )
    parser.add_argument(
        "--actor-observation-level",
        type=str,
        choices=["private", "public", "full"],
        default="private",
        help=(
            "Information level used by the policy network: private, public "
            "(1v1 opponent resources), or full/privileged."
        ),
    )
    parser.add_argument(
        "--critic-observation-level",
        type=str,
        choices=["private", "public", "full"],
        default="full",
        help=(
            "Information level used by the critic network: private, public "
            "(1v1 opponent resources), or full/privileged."
        ),
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default=None,
        help=(
            "Experiment name (under experiments/) or path. When set, both policy "
            "and critic are rebuilt from metadata.json and the architecture flags "
            "above are ignored."
        ),
    )
    parser.add_argument(
        "--which",
        type=str,
        default="best",
        help="Checkpoint selector for --experiment: 'best', 'latest', or a step.",
    )
    parser.add_argument(
        "--policy-weights",
        type=str,
        default=None,
        help="Path to policy model weights (.pt file). Alternative to --experiment.",
    )
    parser.add_argument(
        "--critic-weights",
        type=str,
        default=None,
        help="Path to critic model weights (.pt file). Alternative to --experiment.",
    )
    parser.add_argument(
        "--num-simulations",
        type=int,
        default=128,
        help="MCTS simulations per move",
    )
    parser.add_argument(
        "--ismcts-determinizations",
        "--is-mcts-determinizations",
        dest="ismcts_determinizations",
        type=int,
        default=1,
        help=(
            "Information-Set MCTS: number of belief determinizations of opponents' "
            "hidden dev cards searched per move. 1 disables IS-MCTS (plain search); "
            ">1 requires actor/critic observation public (1v1) or full."
        ),
    )
    parser.add_argument(
        "--num-search-workers",
        type=int,
        default=1,
        help="[experimental, not recommended] Within-tree parallelism: threaded MCTS search "
        "workers per NN player sharing one tree via virtual loss. Profiling shows this is a "
        "net slowdown (GIL-bound) vs across-games parallelism, so keep at 1 and scale "
        "--num-game-workers instead.",
    )
    parser.add_argument(
        "--inference-batch-size",
        type=int,
        default=32,
        help="Maximum leaf evaluations per batched NN inference step.",
    )
    parser.add_argument(
        "--inference-wait-ms",
        type=float,
        default=1.0,
        help="Maximum milliseconds the inference server waits to fill a batch.",
    )
    parser.add_argument(
        "--virtual-loss",
        type=float,
        default=1.0,
        help="Virtual loss applied during threaded tree traversal.",
    )
    parser.add_argument(
        "--c-puct",
        type=float,
        default=1.5,
        help="PUCT exploration constant",
    )
    parser.add_argument(
        "--prunning",
        action="store_true",
        help="Enable catanatron action prunning heuristics",
    )
    parser.add_argument(
        "--num-games",
        type=int,
        default=100,
        help="Number of self-play games to play",
    )
    parser.add_argument(
        "--num-game-workers",
        type=int,
        default=min(16, os.cpu_count() or 4),
        help=(
            "Across-games parallelism (the default strategy): independent self-play game "
            "worker processes feeding a central parent-owned batched NN inference server. "
            "1 runs single-process. Aggregate sims/s saturates near the core count on a "
            "single shared GPU."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--vps-to-win",
        type=int,
        default=15,
        help="Victory points required to win each game",
    )
    parser.add_argument(
        "--discard-limit",
        type=int,
        default=9,
        help="Discard threshold when a 7 is rolled",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda or cpu). Defaults to cuda if available.",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Log evaluation config and summary metrics to Weights & Biases",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default=DEFAULT_WANDB_PROJECT,
        help="Weights & Biases project name",
    )
    parser.add_argument(
        "--wandb-run-name",
        type=str,
        default=None,
        help="Weights & Biases run name",
    )
    parser.add_argument(
        "--wandb-group",
        type=str,
        default=None,
        help="Optional Weights & Biases group name",
    )

    args = parser.parse_args()

    use_experiment = args.experiment is not None
    if use_experiment:
        if args.policy_weights is not None or args.critic_weights is not None:
            parser.error("Do not combine --experiment with --policy-weights/--critic-weights.")
    elif args.policy_weights is None or args.critic_weights is None:
        parser.error("Provide --experiment, or both --policy-weights and --critic-weights.")

    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # Experiment path: rebuild policy + critic from metadata.
    experiment_policy = None
    experiment_critic = None
    if use_experiment:
        exp = load_experiment(args.experiment)
        args.model_type = exp.model_type or args.model_type
        args.map_type = exp.map_type
        args.num_players = exp.num_players
        args.backbone_type = backbone_display_type(exp.policy_spec.backbone)
        args.policy_hidden_dims = backbone_hidden_dims(exp.policy_spec.backbone)
        if exp.policy_spec.observation_level is not None:
            args.actor_observation_level = exp.policy_spec.observation_level
        critic_spec = exp.metadata.networks.get("critic")
        if critic_spec is not None:
            args.critic_hidden_dims = backbone_hidden_dims(critic_spec.backbone)
            if critic_spec.observation_level is not None:
                args.critic_observation_level = critic_spec.observation_level
        if exp.policy_spec.kind == KIND_POLICY_VALUE:
            experiment_policy = exp.build_policy(
                which=args.which, device=device, as_policy_only=False
            )
            experiment_critic = None
        else:
            experiment_policy = exp.build_policy(which=args.which, device=device)
            experiment_critic = exp.build_critic(which=args.which, device=device)

    print(f"\n{'=' * 60}")
    print("Neural MCTS Self-Play Evaluation")
    print(f"{'=' * 60}")
    print(f"Device: {device}")
    print(f"Map type: {args.map_type} | Players: {args.num_players}")
    print(f"Backbone: {args.backbone_type} | Model type: {args.model_type}")
    print(
        f"Actor observation: {args.actor_observation_level} | "
        f"Critic observation: {args.critic_observation_level}"
    )
    print(f"Policy hidden dims: {args.policy_hidden_dims}")
    print(f"Critic hidden dims: {args.critic_hidden_dims}")
    print(f"Policy weights: {args.policy_weights}")
    print(f"Critic weights: {args.critic_weights}")
    print(f"MCTS simulations: {args.num_simulations} | c_puct: {args.c_puct}")
    print(f"IS-MCTS determinizations: {args.ismcts_determinizations}")
    print(
        "MCTS workers: "
        f"{args.num_search_workers} | inference batch: {args.inference_batch_size} "
        f"| wait: {args.inference_wait_ms} ms | virtual loss: {args.virtual_loss}"
    )
    print(f"Prunning: {args.prunning}")
    print(f"Games: {args.num_games} | game workers: {args.num_game_workers}")
    print(f"VPs to win: {args.vps_to_win} | Discard limit: {args.discard_limit}")

    if args.wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            group=args.wandb_group,
            job_type="eval",
            config=vars(args),
        )

    if use_experiment:
        policy_model = experiment_policy
        critic_model = experiment_critic
        if critic_model is None:
            print(
                f"Loaded shared policy-value model from experiment "
                f"'{args.experiment}' ({args.which})"
            )
        else:
            print(f"Loaded policy + critic from experiment '{args.experiment}' ({args.which})")
    else:
        policy_model = build_policy_model(
            backbone_type=args.backbone_type,
            model_type=args.model_type,
            hidden_dims=args.policy_hidden_dims,
            num_players=args.num_players,
            map_type=args.map_type,
            actor_observation_level=args.actor_observation_level,
            device=device,
        )
        critic_model = build_critic_model(
            backbone_type=args.backbone_type,
            hidden_dims=args.critic_hidden_dims,
            num_players=args.num_players,
            map_type=args.map_type,
            critic_observation_level=args.critic_observation_level,
            device=device,
        )

        policy_model.load_state_dict(torch.load(args.policy_weights, map_location=device))
        critic_model.load_state_dict(torch.load(args.critic_weights, map_location=device))
        policy_model.eval()
        critic_model.eval()
        print(f"Loaded policy weights from {args.policy_weights}")
        print(f"Loaded critic weights from {args.critic_weights}")

    resolve_observation_network_args(
        args,
        policy_mode_default=args.actor_observation_level,
        critic_mode_default=args.critic_observation_level,
    )

    print(f"\nRunning {args.num_games} self-play games...")
    if args.num_game_workers <= 1:
        players = build_self_play_players(
            policy_model=policy_model,
            critic_model=critic_model,
            model_type=args.model_type,
            map_type=args.map_type,
            num_players=args.num_players,
            num_simulations=args.num_simulations,
            c_puct=args.c_puct,
            prunning=args.prunning,
            actor_observation_level=args.actor_observation_level,
            critic_observation_level=args.critic_observation_level,
            ismcts_determinizations=args.ismcts_determinizations,
            num_search_workers=args.num_search_workers,
            inference_batch_size=args.inference_batch_size,
            inference_wait_ms=args.inference_wait_ms,
            virtual_loss=args.virtual_loss,
        )
        stats, turns = run_self_play_eval(
            players=players,
            map_type=args.map_type,
            num_games=args.num_games,
            seed=args.seed,
            vps_to_win=args.vps_to_win,
            discard_limit=args.discard_limit,
            show_tqdm=True,
        )
    else:
        stats, turns = run_parallel_self_play_eval(
            policy_model=policy_model,
            critic_model=critic_model,
            model_type=args.model_type,
            map_type=args.map_type,
            num_players=args.num_players,
            num_games=args.num_games,
            seed=args.seed,
            vps_to_win=args.vps_to_win,
            discard_limit=args.discard_limit,
            num_simulations=args.num_simulations,
            num_search_workers=args.num_search_workers,
            inference_batch_size=args.inference_batch_size,
            inference_wait_ms=args.inference_wait_ms,
            virtual_loss=args.virtual_loss,
            c_puct=args.c_puct,
            prunning=args.prunning,
            actor_observation_level=args.actor_observation_level,
            critic_observation_level=args.critic_observation_level,
            ismcts_determinizations=args.ismcts_determinizations,
            num_game_workers=args.num_game_workers,
            device=device,
            show_tqdm=True,
        )

    print(f"\n{'=' * 60}")
    print("Results")
    print(f"{'=' * 60}")
    print(f"{'Player':<12} {'Wins':>8} {'Win Rate':>10} {'Avg VPs':>10}")
    for color in COLOR_ORDER[: args.num_players]:
        player_stats = stats[color]
        win_rate = player_stats.wins / args.num_games if args.num_games else 0.0
        print(
            f"{color_label(color):<12} "
            f"{player_stats.wins:>8} "
            f"{100 * win_rate:>9.1f}% "
            f"{player_stats.avg_vps:>10.2f}"
        )
    avg_turns = sum(turns) / len(turns) if turns else 0.0
    print(f"\nAverage Turns: {avg_turns:.1f}")

    if args.wandb:
        metrics = {"avg_turns": avg_turns}
        for color in COLOR_ORDER[: args.num_players]:
            player_stats = stats[color]
            label = color_label(color).lower()
            metrics[f"{label}/wins"] = player_stats.wins
            metrics[f"{label}/win_rate"] = player_stats.wins / args.num_games if args.num_games else 0.0
            metrics[f"{label}/avg_vps"] = player_stats.avg_vps
        wandb.log(metrics)
        wandb.finish()


if __name__ == "__main__":
    main()
