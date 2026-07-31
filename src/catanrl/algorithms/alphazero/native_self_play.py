"""AlphaZero self-play driven by the native cppanatron MCTS tree."""

from __future__ import annotations

import multiprocessing as mp
import traceback
from collections import Counter
from collections.abc import Sequence
from typing import Literal

import numpy as np
import torch

from catanrl.envs.cppanatron import NativeGame, NativeMCTSSearch, full_native_features
from catanrl.envs.cppanatron.puffer_env import TURNS_LIMIT
from catanrl.features.catanatron_utils import (
    ActorObservationLevel,
    COLOR_ORDER,
    CriticObservationLevel,
    get_observation_indices_from_full,
)
from catanrl.players.nn_mcts_player import (
    _NNMCTSInferenceBackend,
    _RemoteNNMCTSInferenceBackend,
    _visit_distribution,
)
from catanrl.utils.seeding import derive_map_and_game_seeds, derive_seed

from .parallel_self_play import (
    SelfPlayExperience,
    _assign_episode_seeds,
    run_inference_server_workers,
)

MapType = Literal["BASE", "MINI", "TOURNAMENT"]


def _native_search_policy(
    *,
    game: NativeGame,
    map_type: MapType,
    full_state: np.ndarray,
    inference_backend: _NNMCTSInferenceBackend,
    actor_indices: np.ndarray,
    critic_indices: np.ndarray,
    num_simulations: int,
    c_puct: float,
    search_seed: int,
    add_noise: bool,
    dirichlet_alpha: float,
    dirichlet_frac: float,
    action_temperature: float,
    target_temperature: float | None,
    rng: np.random.Generator,
) -> tuple[np.ndarray, int]:
    root_evaluation = inference_backend.evaluate_leaf(
        full_state[actor_indices],
        full_state[critic_indices],
    )

    with NativeMCTSSearch(
        game,
        map_type,
        c_puct=c_puct,
        seed=search_seed,
    ) as search:
        search.initialize_root(root_evaluation.policy_logits)
        if add_noise and dirichlet_frac > 0.0:
            search.add_root_dirichlet_noise(
                dirichlet_alpha,
                dirichlet_frac,
            )
        for _ in range(num_simulations):
            leaf = search.select_leaf()
            if leaf is None:
                continue
            full_leaf_state, _player = leaf
            evaluation = inference_backend.evaluate_leaf(
                full_leaf_state[actor_indices],
                full_leaf_state[critic_indices],
            )
            search.evaluate_leaf(evaluation.policy_logits, evaluation.value)
        visits = search.root_visits().astype(np.float64)

    valid_indices = np.flatnonzero(game.valid_action_mask())
    valid_visits = visits[valid_indices]
    target_probabilities = _visit_distribution(
        valid_visits,
        action_temperature if target_temperature is None else target_temperature,
    )
    action_probabilities = _visit_distribution(valid_visits, action_temperature)
    policy = np.zeros(game.action_space_size, dtype=np.float32)
    policy[valid_indices] = target_probabilities.astype(np.float32, copy=False)
    action = int(rng.choice(valid_indices, p=action_probabilities))
    return policy, action


def _play_native_self_play_game(
    *,
    episode_seed: int,
    args_dict: dict,
    inference_backend: _NNMCTSInferenceBackend,
) -> tuple[
    list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]],
    int | None,
]:
    map_type: MapType = args_dict["map_type"]
    num_players = int(args_dict["num_players"])
    actor_indices = get_observation_indices_from_full(
        num_players,
        map_type,
        args_dict["actor_observation_level"],
    )
    critic_indices = get_observation_indices_from_full(
        num_players,
        map_type,
        args_dict["critic_observation_level"],
    )
    map_seed, game_seed = derive_map_and_game_seeds(episode_seed)
    rng = np.random.default_rng(derive_seed(episode_seed, "native_self_play_actions"))
    game = NativeGame(
        num_players,
        map_type,
        seed=game_seed,
        map_seed=map_seed,
        number_placement="random",
        discard_limit=int(args_dict["discard_limit"]),
        vps_to_win=int(args_dict["vps_to_win"]),
    )
    samples: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]] = []
    move_number = 0
    turns_limit = int(args_dict.get("turns_limit", TURNS_LIMIT))
    try:
        while game.winner is None and game.num_turns < turns_limit:
            current_player = game.current_player
            action_mask = game.valid_action_mask()
            valid_actions = np.flatnonzero(action_mask)
            if valid_actions.size == 0:
                raise RuntimeError("Native self-play position has no legal action")
            if valid_actions.size == 1:
                action = int(valid_actions[0])
            else:
                full_state = full_native_features(
                    game,
                    map_type,
                    base_player=current_player,
                )
                temperature = (
                    float(args_dict["temperature"])
                    if move_number < int(args_dict["temperature_drop_move"])
                    else float(args_dict["final_temperature"])
                )
                policy, action = _native_search_policy(
                    game=game,
                    map_type=map_type,
                    full_state=full_state,
                    inference_backend=inference_backend,
                    actor_indices=actor_indices,
                    critic_indices=critic_indices,
                    num_simulations=int(args_dict["num_simulations"]),
                    c_puct=float(args_dict["c_puct"]),
                    search_seed=derive_seed(
                        episode_seed,
                        "native_mcts",
                        move_number,
                    ),
                    add_noise=move_number < int(args_dict["noise_turns"]),
                    dirichlet_alpha=float(args_dict["dirichlet_alpha"]),
                    dirichlet_frac=float(args_dict["dirichlet_frac"]),
                    action_temperature=max(temperature, 1e-3),
                    target_temperature=(
                        None
                        if args_dict["target_temperature"] is None
                        else float(args_dict["target_temperature"])
                    ),
                    rng=rng,
                )
                samples.append(
                    (
                        full_state[actor_indices].copy(),
                        full_state[critic_indices].copy(),
                        policy,
                        action_mask.copy(),
                        current_player,
                    )
                )
            game.step(action)
            move_number += 1
        return samples, game.winner
    finally:
        game.close()


def _native_training_worker_main(
    worker_id: int,
    request_queue: mp.Queue,
    response_queue: mp.Queue,
    result_queue: mp.Queue,
    episode_seeds: Sequence[int],
    args_dict: dict,
) -> None:
    inference_backend = _RemoteNNMCTSInferenceBackend(
        worker_id=worker_id,
        request_queue=request_queue,
        response_queue=response_queue,
    )
    try:
        experiences: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]] = []
        stats: Counter[str] = Counter()
        for episode_seed in episode_seeds:
            samples, winner = _play_native_self_play_game(
                episode_seed=episode_seed,
                args_dict=args_dict,
                inference_backend=inference_backend,
            )
            stats["games"] += 1
            if winner is not None:
                stats[f"wins_{COLOR_ORDER[winner].value}"] += 1
            for actor_state, critic_state, policy, action_mask, player in samples:
                value = 0.0 if winner is None else (1.0 if player == winner else -1.0)
                experiences.append((actor_state, critic_state, policy, action_mask, value))
        result_queue.put(
            {
                "worker_id": worker_id,
                "games": len(episode_seeds),
                "experiences": experiences,
                "stats": dict(stats),
            }
        )
    except BaseException:
        result_queue.put({"worker_id": worker_id, "error": traceback.format_exc()})
    finally:
        inference_backend.close()


def generate_native_self_play_data(
    *,
    policy_model,
    critic_model,
    model_type: str,
    map_type: MapType,
    num_players: int,
    num_games: int,
    num_game_workers: int,
    num_simulations: int,
    c_puct: float,
    prunning: bool,
    actor_observation_level: ActorObservationLevel,
    critic_observation_level: CriticObservationLevel,
    ismcts_determinizations: int,
    inference_batch_size: int,
    inference_wait_ms: float,
    temperature: float,
    final_temperature: float,
    target_temperature: float | None,
    temperature_drop_move: int,
    noise_turns: int,
    dirichlet_alpha: float,
    dirichlet_frac: float,
    vps_to_win: int,
    discard_limit: int,
    seed: int,
    device: str | torch.device,
    show_tqdm: bool = True,
    turns_limit: int = TURNS_LIMIT,
) -> tuple[list[SelfPlayExperience], dict[str, int]]:
    """Generate the trainer's standard replay records with native C++ MCTS."""
    if prunning:
        raise ValueError("Native MCTS does not implement Python action pruning")
    if ismcts_determinizations != 1:
        raise ValueError("Native MCTS currently requires --ismcts-determinizations 1")
    if num_games <= 0:
        return [], {}

    assignments = [
        assignment
        for assignment in _assign_episode_seeds(num_games, num_game_workers, seed)
        if assignment
    ]
    args_dict = {
        "map_type": map_type,
        "num_players": num_players,
        "num_simulations": num_simulations,
        "c_puct": c_puct,
        "actor_observation_level": actor_observation_level,
        "critic_observation_level": critic_observation_level,
        "temperature": temperature,
        "final_temperature": final_temperature,
        "target_temperature": target_temperature,
        "temperature_drop_move": temperature_drop_move,
        "noise_turns": noise_turns,
        "dirichlet_alpha": dirichlet_alpha,
        "dirichlet_frac": dirichlet_frac,
        "vps_to_win": vps_to_win,
        "discard_limit": discard_limit,
        "turns_limit": turns_limit,
    }
    experiences: list[SelfPlayExperience] = []
    stats: Counter[str] = Counter()

    def handle_result(message: dict) -> None:
        for actor_state, critic_state, policy, action_mask, value in message["experiences"]:
            experiences.append(
                SelfPlayExperience(
                    actor_state=actor_state,
                    critic_state=critic_state,
                    policy=policy,
                    action_mask=action_mask,
                    value=float(value),
                )
            )
        for key, count in message["stats"].items():
            stats[key] += int(count)

    first_error = run_inference_server_workers(
        policy_model=policy_model,
        critic_model=critic_model,
        model_type=model_type,
        device=device,
        num_workers=len(assignments),
        inference_batch_size=inference_batch_size,
        inference_wait_ms=inference_wait_ms,
        worker_target=_native_training_worker_main,
        worker_args=[(assignment, args_dict) for assignment in assignments],
        handle_result=handle_result,
        total=num_games,
        show_tqdm=show_tqdm,
    )
    if first_error is not None:
        raise RuntimeError(f"Native self-play worker failed:\n{first_error}")
    return experiences, dict(stats)


__all__ = ["generate_native_self_play_data"]
