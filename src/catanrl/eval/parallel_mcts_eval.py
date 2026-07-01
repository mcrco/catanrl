"""Across-games MCTS evaluation with centrally batched neural inference."""

from __future__ import annotations

import multiprocessing as mp
import random
import traceback
from dataclasses import dataclass, field
from typing import Any, Literal, Sequence

import numpy as np
import torch
from catanatron.cli.cli_players import CLI_PLAYERS
from catanatron.game import Game
from catanatron.models.player import Player
from catanatron.state_functions import get_actual_victory_points

from catanrl.algorithms.alphazero.parallel_self_play import run_inference_server_workers
from catanrl.features.catanatron_utils import ActorObservationLevel, CriticObservationLevel
from catanrl.models import PolicyNetworkWrapper, PolicyValueNetworkWrapper, ValueNetworkWrapper
from catanrl.players import NNMCTSPlayer
from catanrl.players.nn_mcts_player import _RemoteNNMCTSInferenceBackend
from catanrl.utils.catanatron_game import COLOR_ORDER, SeatOption, build_players_for_seat, force_player_order
from catanrl.utils.catanatron_map import build_catan_map
from catanrl.utils.seeding import derive_map_and_game_seeds, derive_seed


@dataclass
class ParallelMCTSEvalResult:
    wins: int = 0
    vps: list[int] = field(default_factory=list)
    total_vps: list[int] = field(default_factory=list)
    turns: list[int] = field(default_factory=list)

    @property
    def games(self) -> int:
        return len(self.turns)

    def merge(self, payload: dict[str, Any]) -> None:
        self.wins += int(payload["wins"])
        self.vps.extend(int(value) for value in payload["vps"])
        self.total_vps.extend(int(value) for value in payload["total_vps"])
        self.turns.extend(int(value) for value in payload["turns"])


def _assign_episode_seeds(num_games: int, num_workers: int, seed: int) -> list[list[int]]:
    assignments: list[list[int]] = [[] for _ in range(max(1, num_workers))]
    for game_idx in range(num_games):
        assignments[game_idx % len(assignments)].append(
            derive_seed(seed, "episode", game_idx)
        )
    return assignments


def _build_opponents(opponents_str: str) -> list[Player]:
    player_keys = [key.strip() for key in opponents_str.split(",") if key.strip()]
    cli_players_by_code = {cli_player.code: cli_player for cli_player in CLI_PLAYERS}
    players: list[Player] = []
    for index, key in enumerate(player_keys):
        parts = key.split(":")
        cli_player = cli_players_by_code[parts[0]]
        players.append(cli_player.import_fn(COLOR_ORDER[index + 1], *parts[1:]))
    return players


def _worker_main(
    worker_id: int,
    request_queue: mp.Queue,
    response_queue: mp.Queue,
    result_queue: mp.Queue,
    episode_seeds: Sequence[int],
    args_dict: dict[str, Any],
) -> None:
    inference_backend = _RemoteNNMCTSInferenceBackend(
        worker_id=worker_id,
        request_queue=request_queue,
        response_queue=response_queue,
    )
    try:
        nn_player = NNMCTSPlayer(
            color=COLOR_ORDER[0],
            model_type=str(args_dict["model_type"]),
            policy_model=None,
            critic_model=None,
            map_type=args_dict["map_type"],
            num_simulations=int(args_dict["num_simulations"]),
            ismcts_determinizations=int(args_dict["ismcts_determinizations"]),
            c_puct=float(args_dict["c_puct"]),
            prunning=bool(args_dict["prunning"]),
            opponent_policy=str(args_dict["adversarial_policy"]),
            actor_observation_level=args_dict["actor_observation_level"],
            critic_observation_level=args_dict["critic_observation_level"],
            num_search_workers=1,
            inference_backend=inference_backend,
        )
        opponents = _build_opponents(str(args_dict["opponents"]))
        players, allow_upstream_shuffle = build_players_for_seat(
            nn_player,
            opponents,
            args_dict["nn_seat"],
        )

        result = ParallelMCTSEvalResult()
        for episode_seed in episode_seeds:
            random.seed(episode_seed)
            np.random.seed(episode_seed % (2**32))
            torch.manual_seed(episode_seed)
            map_seed, game_seed = derive_map_and_game_seeds(episode_seed)
            for player in players:
                player.reset_state()

            game = Game(
                players=players,
                catan_map=build_catan_map(
                    args_dict["map_type"], seed=map_seed, number_placement="random"
                ),
                seed=game_seed,
                discard_limit=int(args_dict["discard_limit"]),
                vps_to_win=int(args_dict["vps_to_win"]),
            )
            if not allow_upstream_shuffle:
                force_player_order(game, players)
            game.play()

            if game.winning_color() == nn_player.color:
                result.wins += 1
            result.vps.append(get_actual_victory_points(game.state, nn_player.color))
            result.total_vps.append(
                sum(
                    get_actual_victory_points(game.state, color)
                    for color in game.state.colors
                )
            )
            result.turns.append(game.state.num_turns)

        result_queue.put(
            {
                "worker_id": worker_id,
                "games": result.games,
                "result": {
                    "wins": result.wins,
                    "vps": result.vps,
                    "total_vps": result.total_vps,
                    "turns": result.turns,
                },
            }
        )
    except BaseException:
        result_queue.put({"worker_id": worker_id, "error": traceback.format_exc()})
    finally:
        inference_backend.close()


def run_parallel_mcts_eval(
    *,
    policy_model: PolicyNetworkWrapper | PolicyValueNetworkWrapper,
    critic_model: ValueNetworkWrapper | None,
    model_type: str,
    map_type: Literal["BASE", "MINI", "TOURNAMENT"],
    opponents: str,
    num_games: int,
    num_game_workers: int,
    num_simulations: int,
    ismcts_determinizations: int,
    c_puct: float,
    prunning: bool,
    adversarial_policy: str,
    actor_observation_level: ActorObservationLevel,
    critic_observation_level: CriticObservationLevel,
    inference_batch_size: int,
    inference_wait_ms: float,
    nn_seat: SeatOption,
    seed: int,
    vps_to_win: int,
    discard_limit: int,
    device: str | torch.device,
    show_tqdm: bool = True,
) -> ParallelMCTSEvalResult:
    """Evaluate MCTS games in worker processes sharing one inference server."""
    if num_games <= 0:
        return ParallelMCTSEvalResult()
    if num_game_workers < 1:
        raise ValueError("num_game_workers must be at least 1")

    assignments = [
        assignment
        for assignment in _assign_episode_seeds(num_games, num_game_workers, seed)
        if assignment
    ]
    args_dict: dict[str, Any] = {
        "model_type": model_type,
        "map_type": map_type,
        "opponents": opponents,
        "num_simulations": num_simulations,
        "ismcts_determinizations": ismcts_determinizations,
        "c_puct": c_puct,
        "prunning": prunning,
        "adversarial_policy": adversarial_policy,
        "actor_observation_level": actor_observation_level,
        "critic_observation_level": critic_observation_level,
        "nn_seat": nn_seat,
        "vps_to_win": vps_to_win,
        "discard_limit": discard_limit,
    }
    aggregate = ParallelMCTSEvalResult()

    def handle_result(message: dict) -> None:
        payload = message.get("result")
        if not isinstance(payload, dict):
            raise TypeError("MCTS evaluation worker returned an invalid result payload")
        aggregate.merge(payload)

    first_error = run_inference_server_workers(
        policy_model=policy_model,
        critic_model=critic_model,
        model_type=model_type,
        device=device,
        num_workers=len(assignments),
        inference_batch_size=inference_batch_size,
        inference_wait_ms=inference_wait_ms,
        worker_target=_worker_main,
        worker_args=[(assignment, args_dict) for assignment in assignments],
        handle_result=handle_result,
        total=num_games,
        show_tqdm=show_tqdm,
    )
    if first_error is not None:
        raise RuntimeError(f"Parallel MCTS evaluation worker failed:\n{first_error}")
    if aggregate.games != num_games:
        raise RuntimeError(
            f"Parallel MCTS evaluation returned {aggregate.games} games, expected {num_games}"
        )
    return aggregate


__all__ = ["ParallelMCTSEvalResult", "run_parallel_mcts_eval"]
