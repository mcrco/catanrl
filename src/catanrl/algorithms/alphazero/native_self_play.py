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
    COLOR_ORDER,
    ActorObservationLevel,
    CriticObservationLevel,
    get_observation_indices_from_full,
)
from catanrl.players.nn_mcts_player import (
    _NNMCTSInferenceBackend,
    _RemoteNNMCTSInferenceBackend,
)
from catanrl.utils.catanatron_action_space import canopy_action_count_increment
from catanrl.utils.seeding import derive_map_and_game_seeds, derive_seed

from .native_search import PolicyTarget, run_native_search_policy, step_game_and_reconcile_search
from .parallel_self_play import (
    SelfPlayExperience,
    _assign_episode_seeds,
    _put_training_result_chunks,
    run_inference_server_workers,
)

MapType = Literal["BASE", "MINI", "TOURNAMENT"]
TrajectoryActionSelection = Literal["visits", "canopy"]


@dataclass(frozen=True)
class _NativeSelfPlaySample:
    actor_state: np.ndarray
    critic_state: np.ndarray
    policy: np.ndarray
    action_mask: np.ndarray
    player: int
    search_value: float
    search_wdl: np.ndarray
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


def _compute_auxiliary_value_targets(
    samples: Sequence[_NativeSelfPlaySample],
    horizons: Sequence[int],
) -> np.ndarray:
    """Match Canopy's backwards EMA of root-Q in a fixed player perspective."""
    normalized_horizons = tuple(int(horizon) for horizon in horizons)
    if len(set(normalized_horizons)) != len(normalized_horizons) or any(
        horizon <= 0 for horizon in normalized_horizons
    ):
        raise ValueError("Auxiliary value horizons must be distinct and positive")
    targets = np.zeros((len(samples), len(normalized_horizons)), dtype=np.float32)
    if not normalized_horizons:
        return targets
    for sample in samples:
        if sample.player not in (0, 1):
            raise ValueError("Auxiliary value targets currently require exactly two players")
        search_wdl = np.asarray(sample.search_wdl, dtype=np.float32)
        if (
            search_wdl.shape != (3,)
            or not np.isfinite(search_wdl).all()
            or bool((search_wdl < 0.0).any())
            or not np.isclose(search_wdl.sum(), 1.0)
        ):
            raise ValueError("Auxiliary value targets require a normalized three-way search WDL")
    for horizon_index, horizon in enumerate(normalized_horizons):
        alpha = 1.0 - np.exp(-1.0 / float(horizon))
        ema_player_zero = 0.0
        for sample_index in range(len(samples) - 1, -1, -1):
            sample = samples[sample_index]
            player_sign = 1.0 if sample.player == 0 else -1.0
            search_q = float(sample.search_wdl[0] - sample.search_wdl[2])
            q_player_zero = search_q * player_sign
            ema_player_zero = alpha * q_player_zero + (1.0 - alpha) * ema_player_zero
            targets[sample_index, horizon_index] = ema_player_zero * player_sign
    return targets


def _blend_terminal_search_wdl(
    terminal_value: float,
    search_wdl: np.ndarray,
    search_weight: float,
) -> np.ndarray:
    """Blend a one-hot terminal outcome with search-refined WDL probabilities."""
    if not 0.0 <= search_weight <= 1.0:
        raise ValueError("search_weight must be between 0 and 1")
    probabilities = np.asarray(search_wdl, dtype=np.float64)
    if probabilities.shape != (3,):
        raise ValueError(f"Expected search WDL with shape (3,), got {probabilities.shape}")
    if not np.isfinite(probabilities).all() or bool((probabilities < 0.0).any()):
        raise ValueError("search WDL must be finite and non-negative")
    total = float(probabilities.sum())
    if total <= 0.0:
        raise ValueError("search WDL must have positive mass")
    probabilities = probabilities / total
    terminal = np.zeros(3, dtype=np.float64)
    terminal[0 if terminal_value > 0.0 else 2 if terminal_value < 0.0 else 1] = 1.0
    return (1.0 - search_weight) * terminal + search_weight * probabilities


def _choose_trajectory_action(
    *,
    mode: TrajectoryActionSelection,
    search_action: int,
    improved_policy: np.ndarray,
    valid_actions: np.ndarray,
    move_number: int,
    explore_actions: int,
    rng: np.random.Generator,
) -> int:
    """Apply the trajectory rule without changing the stored search target."""
    if mode == "visits" or move_number >= explore_actions:
        return search_action
    if mode != "canopy":
        raise ValueError(f"Unknown trajectory action selection: {mode!r}")
    probabilities = np.asarray(improved_policy[valid_actions], dtype=np.float64)
    probability_sum = float(probabilities.sum())
    if not np.isfinite(probability_sum) or probability_sum <= 0.0:
        probabilities = np.full(valid_actions.size, 1.0 / valid_actions.size)
    else:
        probabilities /= probability_sum
    return int(rng.choice(valid_actions, p=probabilities))


def _trajectory_search_controls(
    *,
    mode: TrajectoryActionSelection,
    move_number: int,
    temperature: float,
    final_temperature: float,
    temperature_drop_move: int,
    noise_turns: int,
) -> tuple[float, bool]:
    """Return action temperature and root-noise use for one search."""
    if mode == "canopy":
        return 1e-3, True
    if mode != "visits":
        raise ValueError(f"Unknown trajectory action selection: {mode!r}")
    action_temperature = temperature if move_number < temperature_drop_move else final_temperature
    return action_temperature, move_number < noise_turns


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
) -> tuple[np.ndarray, int, np.ndarray, float, np.ndarray]:
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
        result.wdl,
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
    shared_observation = np.array_equal(actor_indices, critic_indices)
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
    action_count = 0
    search: NativeMCTSSearch | None = None
    turns_limit = int(args_dict.get("turns_limit", TURNS_LIMIT))
    max_actions = int(args_dict.get("max_actions", 0))
    try:
        while (
            game.winner is None
            and game.num_turns < turns_limit
            and (max_actions <= 0 or action_count < max_actions)
        ):
            current_player = game.current_player
            action_mask = game.valid_action_mask()
            valid_actions = np.flatnonzero(action_mask)
            if valid_actions.size == 0:
                raise RuntimeError("Native self-play position has no legal action")
            if valid_actions.size == 1:
                action = int(valid_actions[0])
            else:
                trajectory_action_selection: TrajectoryActionSelection = args_dict.get(
                    "trajectory_action_selection", "visits"
                )
                temperature, add_noise = _trajectory_search_controls(
                    mode=trajectory_action_selection,
                    move_number=action_count,
                    temperature=float(args_dict["temperature"]),
                    final_temperature=float(args_dict["final_temperature"]),
                    temperature_drop_move=int(args_dict["temperature_drop_move"]),
                    noise_turns=int(args_dict["noise_turns"]),
                )
                if search is None and bool(args_dict.get("tree_reuse", False)):
                    search = NativeMCTSSearch(
                        game,
                        map_type,
                        c_puct=float(args_dict["c_puct"]),
                        seed=derive_seed(episode_seed, "native_mcts", action_count),
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
                policy, action, full_state, search_value, search_wdl = _native_search_policy(
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
                        action_count,
                    ),
                    add_noise=add_noise,
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
                action = _choose_trajectory_action(
                    mode=trajectory_action_selection,
                    search_action=action,
                    improved_policy=policy,
                    valid_actions=valid_actions,
                    move_number=action_count,
                    explore_actions=int(args_dict.get("explore_actions", 24)),
                    rng=rng,
                )
                actor_state = full_state[actor_indices].copy()
                critic_state = (
                    actor_state if shared_observation else full_state[critic_indices].copy()
                )
                samples.append(
                    _NativeSelfPlaySample(
                        actor_state=actor_state,
                        critic_state=critic_state,
                        policy=policy,
                        action_mask=action_mask.copy(),
                        player=current_player,
                        search_value=search_value,
                        search_wdl=search_wdl,
                        full_search=full_search,
                    )
                )
            search = step_game_and_reconcile_search(
                game=game,
                map_type=map_type,
                action=action,
                search=search,
            )
            action_count += canopy_action_count_increment(action, num_players, map_type)
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
        response_timeout_s=float(args_dict["inference_response_timeout_s"]),
    )
    try:
        for episode_seed in episode_seeds:
            experiences: list[
                tuple[
                    np.ndarray,
                    np.ndarray,
                    np.ndarray,
                    np.ndarray,
                    float,
                    bool,
                    np.ndarray,
                    np.ndarray | None,
                ]
            ] = []
            stats: Counter[str] = Counter()
            samples, winner = _play_native_self_play_game(
                episode_seed=episode_seed,
                args_dict=args_dict,
                inference_backend=inference_backend,
            )
            stats["games"] += 1
            if winner is not None:
                stats[f"wins_{COLOR_ORDER[winner].value}"] += 1
            search_value_weight = float(args_dict.get("search_value_weight", 0.0))
            aux_value_horizons = tuple(args_dict.get("aux_value_horizons", ()))
            aux_value_targets = _compute_auxiliary_value_targets(
                samples,
                aux_value_horizons,
            )
            for sample_index, sample in enumerate(samples):
                terminal_value = (
                    0.0 if winner is None else (1.0 if sample.player == winner else -1.0)
                )
                value_wdl = _blend_terminal_search_wdl(
                    terminal_value,
                    sample.search_wdl,
                    search_value_weight,
                )
                value = float(value_wdl[0] - value_wdl[2])
                experiences.append(
                    (
                        sample.actor_state,
                        sample.critic_state,
                        sample.policy,
                        sample.action_mask,
                        value,
                        sample.full_search,
                        value_wdl,
                        (aux_value_targets[sample_index] if aux_value_horizons else None),
                    )
                )
            stats["full_search_decisions"] += sum(int(sample.full_search) for sample in samples)
            stats["fast_search_decisions"] += sum(int(not sample.full_search) for sample in samples)
            _put_training_result_chunks(
                result_queue=result_queue,
                worker_id=worker_id,
                experiences=experiences,
                stats=dict(stats),
                chunk_size=int(args_dict["result_chunk_size"]),
            )
        result_queue.put(
            {
                "worker_id": worker_id,
                "done": True,
                "games": 0,
                "experiences": [],
                "stats": {},
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
    max_actions: int = 0,
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
    trajectory_action_selection: TrajectoryActionSelection = "visits",
    explore_actions: int = 24,
    worker_stall_timeout_s: float = 600.0,
    inference_response_timeout_s: float = 120.0,
    result_chunk_size: int = 64,
    aux_value_horizons: Sequence[int] = (),
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
    if trajectory_action_selection not in ("visits", "canopy"):
        raise ValueError("trajectory_action_selection must be 'visits' or 'canopy'")
    if explore_actions < 0:
        raise ValueError("explore_actions cannot be negative")
    if max_actions < 0:
        raise ValueError("max_actions cannot be negative")
    if worker_stall_timeout_s <= 0.0:
        raise ValueError("worker_stall_timeout_s must be positive")
    if inference_response_timeout_s <= 0.0:
        raise ValueError("inference_response_timeout_s must be positive")
    if result_chunk_size < 1:
        raise ValueError("result_chunk_size must be at least 1")
    normalized_aux_value_horizons = tuple(int(horizon) for horizon in aux_value_horizons)
    if len(set(normalized_aux_value_horizons)) != len(normalized_aux_value_horizons) or any(
        horizon <= 0 for horizon in normalized_aux_value_horizons
    ):
        raise ValueError("aux_value_horizons must contain distinct positive integers")
    if normalized_aux_value_horizons and num_players != 2:
        raise ValueError("Auxiliary value targets currently require exactly two players")
    if trajectory_action_selection == "canopy":
        if policy_target != "completed-q" or search_selection != "completed-q":
            raise ValueError("Canopy trajectory selection requires completed-Q search and targets")
        if target_temperature is None or not np.isclose(target_temperature, 1.0):
            raise ValueError("Canopy trajectory selection requires target_temperature=1.0")
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
        "max_actions": max_actions,
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
        "trajectory_action_selection": trajectory_action_selection,
        "explore_actions": explore_actions,
        "inference_response_timeout_s": inference_response_timeout_s,
        "result_chunk_size": result_chunk_size,
        "aux_value_horizons": normalized_aux_value_horizons,
    }
    experiences: list[SelfPlayExperience] = []
    stats: Counter[str] = Counter()

    def handle_result(message: dict) -> None:
        for (
            actor_state,
            critic_state,
            policy,
            action_mask,
            value,
            full_search,
            value_wdl,
            aux_value_targets,
        ) in message["experiences"]:
            experiences.append(
                SelfPlayExperience(
                    actor_state=actor_state,
                    critic_state=critic_state,
                    policy=policy,
                    action_mask=action_mask,
                    value=float(value),
                    full_search=bool(full_search),
                    value_wdl=np.asarray(value_wdl, dtype=np.float32),
                    aux_value_targets=(
                        None
                        if aux_value_targets is None
                        else np.asarray(aux_value_targets, dtype=np.float32)
                    ),
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
        stall_timeout_s=worker_stall_timeout_s,
    )
    if first_error is not None:
        raise RuntimeError(f"Native self-play worker failed:\n{first_error}")
    return experiences, dict(stats)


__all__ = ["generate_native_self_play_data"]
