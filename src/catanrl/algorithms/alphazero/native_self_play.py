"""AlphaZero self-play driven by the native cppanatron MCTS tree."""

from __future__ import annotations

import multiprocessing as mp
import traceback
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch

from catanrl.envs.cppanatron import NativeGame, NativeMCTSSearch
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
)
from catanrl.utils.seeding import derive_map_and_game_seeds, derive_seed

from .parallel_self_play import (
    SelfPlayExperience,
    _assign_episode_seeds,
    run_inference_server_workers,
)
from .native_search import PolicyTarget, run_native_search_policy, step_game_and_reconcile_search

MapType = Literal["BASE", "MINI", "TOURNAMENT"]


@dataclass(frozen=True)
class _NativeSelfPlaySample:
    actor_state: np.ndarray
    critic_state: np.ndarray
    policy: np.ndarray
    action_mask: np.ndarray
    player: int
    search_value: float
    full_search: bool

    def __iter__(self):
        """Preserve the historical five-field unpacking used by diagnostics/tests."""
        return iter(
            (
                self.actor_state,
                self.critic_state,
                self.policy,
                self.action_mask,
                self.player,
            )
        )


def _blend_terminal_search_value(
    terminal_value: float,
    search_value: float,
    search_weight: float,
) -> float:
    """Blend root Q into a terminal win/loss target without changing its scale."""
    if not 0.0 <= search_weight <= 1.0:
        raise ValueError("search_weight must be between 0 and 1")
    terminal = float(np.clip(terminal_value, -1.0, 1.0))
    search = float(np.clip(search_value, -1.0, 1.0))
    return (1.0 - search_weight) * terminal + search_weight * search


def _native_search_policy(
    *,
    game: NativeGame,
    map_type: MapType,
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
    value_scale: float = 1.0,
    canonical_pruning: bool = False,
    search: NativeMCTSSearch | None = None,
    policy_target: PolicyTarget = "visits",
    c_visit: float = 50.0,
    c_scale: float = 1.0,
    search_selection: str = "puct",
) -> tuple[np.ndarray, int, np.ndarray, float]:
    result = run_native_search_policy(
        game=game,
        map_type=map_type,
        inference_backend=inference_backend,
        actor_indices=actor_indices,
        critic_indices=critic_indices,
        num_simulations=num_simulations,
        c_puct=c_puct,
        search_seed=search_seed,
        add_noise=add_noise,
        dirichlet_alpha=dirichlet_alpha,
        dirichlet_frac=dirichlet_frac,
        action_temperature=action_temperature,
        target_temperature=target_temperature,
        rng=rng,
        value_scale=value_scale,
        canonical_pruning=canonical_pruning,
        search=search,
        policy_target=policy_target,
        c_visit=c_visit,
        c_scale=c_scale,
        search_selection=search_selection,
    )
    return (
        result.policy,
        result.action,
        result.full_state,
        result.diagnostics.search_value,
    )


def _play_native_self_play_game(
    *,
    episode_seed: int,
    args_dict: dict,
    inference_backend: _NNMCTSInferenceBackend,
) -> tuple[
    list[_NativeSelfPlaySample],
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
    samples: list[_NativeSelfPlaySample] = []
    move_number = 0
    search: NativeMCTSSearch | None = None
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
                temperature = (
                    float(args_dict["temperature"])
                    if move_number < int(args_dict["temperature_drop_move"])
                    else float(args_dict["final_temperature"])
                )
                if search is None and bool(args_dict.get("tree_reuse", False)):
                    search = NativeMCTSSearch(
                        game,
                        map_type,
                        c_puct=float(args_dict["c_puct"]),
                        seed=derive_seed(episode_seed, "native_mcts", move_number),
                        canonical_pruning=bool(args_dict.get("canonical_pruning", False)),
                        search_selection=args_dict.get("search_selection", "puct"),
                        c_visit=float(args_dict.get("c_visit", 50.0)),
                        c_scale=float(args_dict.get("c_scale", 1.0)),
                    )
                full_search_probability = float(args_dict.get("full_search_probability", 1.0))
                full_search = full_search_probability >= 1.0 or (
                    rng.random() < full_search_probability
                )
                num_simulations = (
                    int(args_dict["num_simulations"])
                    if full_search
                    else int(args_dict.get("fast_simulations", args_dict["num_simulations"]))
                )
                policy, action, full_state, search_value = _native_search_policy(
                    game=game,
                    map_type=map_type,
                    inference_backend=inference_backend,
                    actor_indices=actor_indices,
                    critic_indices=critic_indices,
                    num_simulations=num_simulations,
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
                    value_scale=float(args_dict.get("value_scale", 1.0)),
                    canonical_pruning=bool(args_dict.get("canonical_pruning", False)),
                    search=search,
                    policy_target=args_dict.get("policy_target", "visits"),
                    c_visit=float(args_dict.get("c_visit", 50.0)),
                    c_scale=float(args_dict.get("c_scale", 1.0)),
                    search_selection=args_dict.get("search_selection", "puct"),
                )
                samples.append(
                    _NativeSelfPlaySample(
                        actor_state=full_state[actor_indices].copy(),
                        critic_state=full_state[critic_indices].copy(),
                        policy=policy,
                        action_mask=action_mask.copy(),
                        player=current_player,
                        search_value=search_value,
                        full_search=full_search,
                    )
                )
            search = step_game_and_reconcile_search(
                game=game,
                map_type=map_type,
                action=action,
                search=search,
            )
            move_number += 1
        return samples, game.winner
    finally:
        if search is not None:
            search.close()
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
        experiences: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, bool]] = []
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
            search_value_weight = float(args_dict.get("search_value_weight", 0.0))
            for sample in samples:
                terminal_value = (
                    0.0 if winner is None else (1.0 if sample.player == winner else -1.0)
                )
                value = _blend_terminal_search_value(
                    terminal_value,
                    sample.search_value,
                    search_value_weight,
                )
                experiences.append(
                    (
                        sample.actor_state,
                        sample.critic_state,
                        sample.policy,
                        sample.action_mask,
                        value,
                        sample.full_search,
                    )
                )
            stats["full_search_decisions"] += sum(int(sample.full_search) for sample in samples)
            stats["fast_search_decisions"] += sum(int(not sample.full_search) for sample in samples)
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
    value_scale: float = 1.0,
    tree_reuse: bool = False,
    canonical_pruning: bool = False,
    full_search_probability: float = 1.0,
    fast_simulations: int = 64,
    search_value_weight: float = 0.0,
    policy_target: PolicyTarget = "visits",
    c_visit: float = 50.0,
    c_scale: float = 1.0,
    search_selection: str = "puct",
) -> tuple[list[SelfPlayExperience], dict[str, int]]:
    """Generate the trainer's standard replay records with native C++ MCTS."""
    if prunning:
        raise ValueError("Native MCTS does not implement Python action pruning")
    if ismcts_determinizations != 1:
        raise ValueError("Native MCTS currently requires --ismcts-determinizations 1")
    if not 0.0 < full_search_probability <= 1.0:
        raise ValueError("full_search_probability must be in (0, 1]")
    if fast_simulations < 1 or (
        full_search_probability < 1.0 and fast_simulations > num_simulations
    ):
        raise ValueError("fast_simulations must be between 1 and num_simulations")
    if not 0.0 <= search_value_weight <= 1.0:
        raise ValueError("search_value_weight must be between 0 and 1")
    if policy_target not in ("visits", "completed-q"):
        raise ValueError("policy_target must be 'visits' or 'completed-q'")
    if search_selection not in ("puct", "completed-q"):
        raise ValueError("search_selection must be 'puct' or 'completed-q'")
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
        "value_scale": value_scale,
        "tree_reuse": tree_reuse,
        "canonical_pruning": canonical_pruning,
        "full_search_probability": full_search_probability,
        "fast_simulations": fast_simulations,
        "search_value_weight": search_value_weight,
        "policy_target": policy_target,
        "c_visit": c_visit,
        "c_scale": c_scale,
        "search_selection": search_selection,
    }
    experiences: list[SelfPlayExperience] = []
    stats: Counter[str] = Counter()

    def handle_result(message: dict) -> None:
        for actor_state, critic_state, policy, action_mask, value, full_search in message[
            "experiences"
        ]:
            experiences.append(
                SelfPlayExperience(
                    actor_state=actor_state,
                    critic_state=critic_state,
                    policy=policy,
                    action_mask=action_mask,
                    value=float(value),
                    full_search=bool(full_search),
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
