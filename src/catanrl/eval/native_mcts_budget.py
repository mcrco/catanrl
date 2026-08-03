"""Native MCTS budget diagnostics and frozen-policy strength evaluation."""

from __future__ import annotations

import multiprocessing as mp
import traceback
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import numpy as np
import torch

from catanrl.algorithms.alphazero.native_search import (
    NativeSearchDiagnostics,
    run_native_search_policy,
    step_game_and_reconcile_search,
)
from catanrl.algorithms.alphazero.parallel_self_play import run_inference_server_workers
from catanrl.envs.cppanatron import NativeGame, NativeMCTSSearch, full_native_features
from catanrl.envs.cppanatron.puffer_env import TURNS_LIMIT
from catanrl.eval.reporting import wilson_interval
from catanrl.features.catanatron_utils import (
    ActorObservationLevel,
    CriticObservationLevel,
    get_observation_indices_from_full,
)
from catanrl.players.nn_mcts_player import _RemoteNNMCTSInferenceBackend
from catanrl.utils.seeding import derive_map_and_game_seeds, derive_seed

MapType = Literal["BASE", "MINI", "TOURNAMENT"]
GameOpponent = Literal["raw", "value"]

_MEAN_DIAGNOSTIC_FIELDS = (
    "simulations",
    "principal_variation_depth",
    "maximum_depth",
    "mean_depth",
    "legal_actions",
    "top1_agreement",
    "policy_kl",
    "policy_js",
    "prior_entropy",
    "search_entropy",
    "network_value",
    "search_value",
    "value_shift",
    "value_scale",
    "backed_up_network_value",
    "retained_root_visits",
    "tree_reused",
    "pruned_actions",
    "coalesced_outcomes",
    "neural_evaluations",
    "elapsed_seconds",
)


@dataclass
class SearchDiagnosticsAccumulator:
    count: int = 0
    maximum_observed_depth: int = 0
    sums: dict[str, float] = field(default_factory=dict)

    def add(self, diagnostics: NativeSearchDiagnostics) -> None:
        self.count += 1
        self.maximum_observed_depth = max(
            self.maximum_observed_depth,
            diagnostics.maximum_depth,
        )
        for name in _MEAN_DIAGNOSTIC_FIELDS:
            self.sums[name] = self.sums.get(name, 0.0) + float(getattr(diagnostics, name))

    def merge(self, payload: dict[str, Any]) -> None:
        self.count += int(payload.get("count", 0))
        self.maximum_observed_depth = max(
            self.maximum_observed_depth,
            int(payload.get("maximum_observed_depth", 0)),
        )
        for name, value in payload.get("sums", {}).items():
            self.sums[name] = self.sums.get(name, 0.0) + float(value)

    def payload(self) -> dict[str, Any]:
        return {
            "count": self.count,
            "maximum_observed_depth": self.maximum_observed_depth,
            "sums": self.sums,
        }

    def summary(self) -> dict[str, float]:
        if self.count == 0:
            return {"searches": 0.0}
        result = {
            "searches": float(self.count),
            "maximum_observed_depth": float(self.maximum_observed_depth),
        }
        result.update({f"mean_{name}": self.sums.get(name, 0.0) / self.count for name in self.sums})
        elapsed = self.sums.get("elapsed_seconds", 0.0)
        if elapsed > 0.0:
            result["simulations_per_second"] = (
                self.sums.get("simulations", 0.0) / elapsed if "simulations" in self.sums else 0.0
            )
        return result


@dataclass
class CriticCalibrationAccumulator:
    count: int = 0
    prediction_sum: float = 0.0
    target_sum: float = 0.0
    prediction_square_sum: float = 0.0
    target_square_sum: float = 0.0
    cross_sum: float = 0.0
    squared_error_sum: float = 0.0
    absolute_error_sum: float = 0.0

    def add(self, prediction: float, target: float) -> None:
        self.count += 1
        self.prediction_sum += prediction
        self.target_sum += target
        self.prediction_square_sum += prediction * prediction
        self.target_square_sum += target * target
        self.cross_sum += prediction * target
        self.squared_error_sum += (prediction - target) ** 2
        self.absolute_error_sum += abs(prediction - target)

    def merge(self, payload: dict[str, Any]) -> None:
        self.count += int(payload.get("count", 0))
        for name in (
            "prediction_sum",
            "target_sum",
            "prediction_square_sum",
            "target_square_sum",
            "cross_sum",
            "squared_error_sum",
            "absolute_error_sum",
        ):
            setattr(self, name, getattr(self, name) + float(payload.get(name, 0.0)))

    def payload(self) -> dict[str, float | int]:
        return asdict(self)

    def summary(self) -> dict[str, float]:
        if self.count == 0:
            return {"samples": 0.0}
        count = float(self.count)
        prediction_mean = self.prediction_sum / count
        target_mean = self.target_sum / count
        prediction_variance = max(0.0, self.prediction_square_sum / count - prediction_mean**2)
        target_variance = max(0.0, self.target_square_sum / count - target_mean**2)
        covariance = self.cross_sum / count - prediction_mean * target_mean
        denominator = np.sqrt(prediction_variance * target_variance)
        least_squares_scale = (
            self.cross_sum / self.prediction_square_sum if self.prediction_square_sum > 0.0 else 0.0
        )
        affine_scale = covariance / prediction_variance if prediction_variance > 0.0 else 0.0
        return {
            "samples": count,
            "mse": self.squared_error_sum / count,
            "mae": self.absolute_error_sum / count,
            "bias": prediction_mean - target_mean,
            "mean_prediction": prediction_mean,
            "std_prediction": float(np.sqrt(prediction_variance)),
            "mean_target": target_mean,
            "correlation": covariance / denominator if denominator > 0.0 else 0.0,
            "least_squares_scale": least_squares_scale,
            "affine_scale": affine_scale,
            "affine_bias": target_mean - affine_scale * prediction_mean,
        }


@dataclass
class NativeBudgetGameResult:
    game_records: list[dict[str, Any]] = field(default_factory=list)
    diagnostics: SearchDiagnosticsAccumulator = field(default_factory=SearchDiagnosticsAccumulator)
    calibration: CriticCalibrationAccumulator = field(default_factory=CriticCalibrationAccumulator)

    def merge(self, payload: dict[str, Any]) -> None:
        self.game_records.extend(dict(record) for record in payload.get("game_records", []))
        self.diagnostics.merge(payload.get("diagnostics", {}))
        self.calibration.merge(payload.get("calibration", {}))

    def summary(self) -> dict[str, float]:
        games = len(self.game_records)
        wins = sum(bool(record["win"]) for record in self.game_records)
        first = [record for record in self.game_records if record["seat"] == "first"]
        second = [record for record in self.game_records if record["seat"] == "second"]
        result = {
            "games": float(games),
            "wins": float(wins),
            "win_rate": wins / games if games else 0.0,
            "first_win_rate": (
                sum(bool(record["win"]) for record in first) / len(first) if first else 0.0
            ),
            "second_win_rate": (
                sum(bool(record["win"]) for record in second) / len(second) if second else 0.0
            ),
            "mean_vps": (
                sum(float(record["vps"]) for record in self.game_records) / games if games else 0.0
            ),
            "mean_turns": (
                sum(float(record["turns"]) for record in self.game_records) / games
                if games
                else 0.0
            ),
            "draw_rate": (
                sum(bool(record["draw"]) for record in self.game_records) / games if games else 0.0
            ),
        }
        if games:
            ci_low, ci_high = wilson_interval(wins, games)
            result["win_rate_ci95_low"] = ci_low
            result["win_rate_ci95_high"] = ci_high
        result.update({f"search/{key}": value for key, value in self.diagnostics.summary().items()})
        result.update({f"critic/{key}": value for key, value in self.calibration.summary().items()})
        return result


@dataclass
class NativeBudgetProbeResult:
    position_records: list[dict[str, Any]] = field(default_factory=list)

    def merge(self, payload: dict[str, Any]) -> None:
        self.position_records.extend(dict(record) for record in payload.get("position_records", []))

    def summaries(self, budgets: Sequence[int]) -> dict[int, dict[str, float]]:
        summaries: dict[int, dict[str, float]] = {}
        for budget in budgets:
            rows = [row for row in self.position_records if int(row["budget"]) == budget]
            if not rows:
                summaries[budget] = {"positions": 0.0}
                continue
            summary = {"positions": float(len(rows))}
            for name in _MEAN_DIAGNOSTIC_FIELDS:
                summary[f"mean_{name}"] = sum(float(row[name]) for row in rows) / len(rows)
            summary["maximum_observed_depth"] = float(
                max(int(row["maximum_depth"]) for row in rows)
            )
            summaries[budget] = summary
        return summaries


def _indices(args_dict: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    return (
        get_observation_indices_from_full(
            2,
            args_dict["map_type"],
            args_dict["actor_observation_level"],
        ),
        get_observation_indices_from_full(
            2,
            args_dict["map_type"],
            args_dict["critic_observation_level"],
        ),
    )


def _raw_policy_action(
    game: NativeGame,
    map_type: MapType,
    actor_indices: np.ndarray,
    critic_indices: np.ndarray,
    inference_backend: _RemoteNNMCTSInferenceBackend,
) -> int:
    full_state = full_native_features(game, map_type, game.current_player)
    evaluation = inference_backend.evaluate_leaf(
        full_state[actor_indices],
        full_state[critic_indices],
    )
    valid_indices = np.flatnonzero(game.valid_action_mask())
    return int(valid_indices[int(np.argmax(evaluation.policy_logits[valid_indices]))])


def _game_opponent_action(
    game: NativeGame,
    opponent: GameOpponent,
    map_type: MapType,
    actor_indices: np.ndarray,
    critic_indices: np.ndarray,
    inference_backend: _RemoteNNMCTSInferenceBackend,
) -> int:
    if opponent == "value":
        return game.value_action()
    if opponent == "raw":
        return _raw_policy_action(
            game,
            map_type,
            actor_indices,
            critic_indices,
            inference_backend,
        )
    raise ValueError(f"Unsupported native game opponent: {opponent!r}")


def _search(
    *,
    game: NativeGame,
    budget: int,
    decision_index: int,
    episode_seed: int,
    args_dict: dict[str, Any],
    actor_indices: np.ndarray,
    critic_indices: np.ndarray,
    inference_backend: _RemoteNNMCTSInferenceBackend,
    search: NativeMCTSSearch | None = None,
):
    return run_native_search_policy(
        game=game,
        map_type=args_dict["map_type"],
        inference_backend=inference_backend,
        actor_indices=actor_indices,
        critic_indices=critic_indices,
        num_simulations=budget,
        c_puct=float(args_dict["c_puct"]),
        search_seed=derive_seed(episode_seed, "native_budget_search", decision_index),
        add_noise=False,
        dirichlet_alpha=0.3,
        dirichlet_frac=0.0,
        action_temperature=0.0,
        target_temperature=1.0,
        rng=np.random.default_rng(
            derive_seed(episode_seed, "native_budget_action", decision_index)
        ),
        value_scale=float(args_dict.get("value_scale", 1.0)),
        canonical_pruning=bool(args_dict.get("canonical_pruning", False)),
        search=search,
        search_selection=args_dict.get("search_selection", "puct"),
        c_visit=float(args_dict.get("c_visit", 50.0)),
        c_scale=float(args_dict.get("c_scale", 1.0)),
    )


def _play_budget_game(
    *,
    episode_seed: int,
    mcts_seat: int,
    budget: int,
    args_dict: dict[str, Any],
    actor_indices: np.ndarray,
    critic_indices: np.ndarray,
    inference_backend: _RemoteNNMCTSInferenceBackend,
) -> tuple[
    dict[str, Any],
    SearchDiagnosticsAccumulator,
    CriticCalibrationAccumulator,
]:
    map_seed, game_seed = derive_map_and_game_seeds(episode_seed)
    diagnostics = SearchDiagnosticsAccumulator()
    calibration_rows: list[tuple[float, int]] = []
    calibration = CriticCalibrationAccumulator()
    game = NativeGame(
        2,
        args_dict["map_type"],
        seed=game_seed,
        map_seed=map_seed,
        number_placement="random",
        discard_limit=int(args_dict["discard_limit"]),
        vps_to_win=int(args_dict["vps_to_win"]),
    )
    decision_index = 0
    search: NativeMCTSSearch | None = None
    try:
        while game.winner is None and game.num_turns < int(args_dict["turns_limit"]):
            valid_indices = np.flatnonzero(game.valid_action_mask())
            if valid_indices.size == 0:
                raise RuntimeError("Native budget game has no legal action")
            if valid_indices.size == 1:
                action = int(valid_indices[0])
            elif game.current_player == mcts_seat:
                if search is None and bool(args_dict.get("tree_reuse", False)):
                    search = NativeMCTSSearch(
                        game,
                        args_dict["map_type"],
                        c_puct=float(args_dict["c_puct"]),
                        seed=derive_seed(
                            episode_seed,
                            "native_budget_search",
                            decision_index,
                        ),
                        canonical_pruning=bool(args_dict.get("canonical_pruning", False)),
                        search_selection=args_dict.get("search_selection", "puct"),
                        c_visit=float(args_dict.get("c_visit", 50.0)),
                        c_scale=float(args_dict.get("c_scale", 1.0)),
                    )
                search_result = _search(
                    game=game,
                    budget=budget,
                    decision_index=decision_index,
                    episode_seed=episode_seed,
                    args_dict=args_dict,
                    actor_indices=actor_indices,
                    critic_indices=critic_indices,
                    inference_backend=inference_backend,
                    search=search,
                )
                diagnostics.add(search_result.diagnostics)
                calibration_rows.append(
                    (search_result.diagnostics.network_value, game.current_player)
                )
                action = search_result.action
            else:
                action = _game_opponent_action(
                    game,
                    args_dict["game_opponent"],
                    args_dict["map_type"],
                    actor_indices,
                    critic_indices,
                    inference_backend,
                )
            search = step_game_and_reconcile_search(
                game=game,
                map_type=args_dict["map_type"],
                action=action,
                search=search,
            )
            decision_index += 1

        winner = game.winner
        for prediction, player in calibration_rows:
            target = 0.0 if winner is None else (1.0 if winner == player else -1.0)
            calibration.add(prediction, target)
        mcts_vps = game.player_state(mcts_seat).actual_victory_points
        return (
            {
                "budget": budget,
                "seat": "first" if mcts_seat == 0 else "second",
                "episode_seed": episode_seed,
                "win": winner == mcts_seat,
                "draw": winner is None,
                "winner": winner,
                "vps": mcts_vps,
                "total_vps": sum(
                    game.player_state(player).actual_victory_points for player in range(2)
                ),
                "turns": game.num_turns,
                "actions": decision_index,
                "opponent": args_dict["game_opponent"],
            },
            diagnostics,
            calibration,
        )
    finally:
        if search is not None:
            search.close()
        game.close()


def _game_worker_main(
    worker_id: int,
    request_queue: mp.Queue,
    response_queue: mp.Queue,
    result_queue: mp.Queue,
    scenarios: Sequence[tuple[int, int]],
    budget: int,
    args_dict: dict[str, Any],
) -> None:
    inference_backend = _RemoteNNMCTSInferenceBackend(
        worker_id=worker_id,
        request_queue=request_queue,
        response_queue=response_queue,
    )
    try:
        actor_indices, critic_indices = _indices(args_dict)
        for mcts_seat, episode_seed in scenarios:
            result = NativeBudgetGameResult()
            record, diagnostics, calibration = _play_budget_game(
                episode_seed=episode_seed,
                mcts_seat=mcts_seat,
                budget=budget,
                args_dict=args_dict,
                actor_indices=actor_indices,
                critic_indices=critic_indices,
                inference_backend=inference_backend,
            )
            result.game_records.append(record)
            result.diagnostics.merge(diagnostics.payload())
            result.calibration.merge(calibration.payload())
            result_queue.put(
                {
                    "worker_id": worker_id,
                    "done": False,
                    "games": 1,
                    "result": {
                        "game_records": result.game_records,
                        "diagnostics": result.diagnostics.payload(),
                        "calibration": result.calibration.payload(),
                    },
                }
            )
        result_queue.put(
            {
                "worker_id": worker_id,
                "done": True,
                "games": 0,
                "result": {
                    "game_records": [],
                    "diagnostics": SearchDiagnosticsAccumulator().payload(),
                    "calibration": CriticCalibrationAccumulator().payload(),
                },
            }
        )
    except BaseException:  # noqa: BLE001 - propagate worker failures through the queue
        result_queue.put({"worker_id": worker_id, "error": traceback.format_exc()})
    finally:
        inference_backend.close()


def _probe_worker_main(
    worker_id: int,
    request_queue: mp.Queue,
    response_queue: mp.Queue,
    result_queue: mp.Queue,
    indexed_episode_seeds: Sequence[tuple[int, int]],
    budgets: Sequence[int],
    args_dict: dict[str, Any],
) -> None:
    inference_backend = _RemoteNNMCTSInferenceBackend(
        worker_id=worker_id,
        request_queue=request_queue,
        response_queue=response_queue,
    )
    position_records: list[dict[str, Any]] = []
    try:
        actor_indices, critic_indices = _indices(args_dict)
        for game_index, episode_seed in indexed_episode_seeds:
            map_seed, game_seed = derive_map_and_game_seeds(episode_seed)
            game = NativeGame(
                2,
                args_dict["map_type"],
                seed=game_seed,
                map_seed=map_seed,
                number_placement="random",
                discard_limit=int(args_dict["discard_limit"]),
                vps_to_win=int(args_dict["vps_to_win"]),
            )
            decision_index = 0
            probe_index = 0
            try:
                while game.winner is None and game.num_turns < int(args_dict["turns_limit"]):
                    valid_indices = np.flatnonzero(game.valid_action_mask())
                    if valid_indices.size == 0:
                        raise RuntimeError("Native probe trajectory has no legal action")
                    if valid_indices.size == 1:
                        action = int(valid_indices[0])
                    else:
                        if probe_index % int(args_dict["probe_stride"]) == 0:
                            for budget in budgets:
                                search_result = _search(
                                    game=game,
                                    budget=int(budget),
                                    decision_index=decision_index,
                                    episode_seed=episode_seed,
                                    args_dict=args_dict,
                                    actor_indices=actor_indices,
                                    critic_indices=critic_indices,
                                    inference_backend=inference_backend,
                                )
                                position_records.append(
                                    {
                                        "game_index": game_index,
                                        "episode_seed": episode_seed,
                                        "decision_index": decision_index,
                                        "turn": game.num_turns,
                                        "player": game.current_player,
                                        "budget": int(budget),
                                        **asdict(search_result.diagnostics),
                                    }
                                )
                        action = _raw_policy_action(
                            game,
                            args_dict["map_type"],
                            actor_indices,
                            critic_indices,
                            inference_backend,
                        )
                        probe_index += 1
                    game.step(action)
                    decision_index += 1
            finally:
                game.close()
        result_queue.put(
            {
                "worker_id": worker_id,
                "games": len(indexed_episode_seeds),
                "result": {"position_records": position_records},
            }
        )
    except BaseException:  # noqa: BLE001 - propagate worker failures through the queue
        result_queue.put({"worker_id": worker_id, "error": traceback.format_exc()})
    finally:
        inference_backend.close()


def _distribute(items: Sequence[Any], num_workers: int) -> list[list[Any]]:
    assignments: list[list[Any]] = [[] for _ in range(max(1, num_workers))]
    for index, item in enumerate(items):
        assignments[index % len(assignments)].append(item)
    return [assignment for assignment in assignments if assignment]


def _common_args(
    *,
    map_type: MapType,
    actor_observation_level: ActorObservationLevel,
    critic_observation_level: CriticObservationLevel,
    c_puct: float,
    vps_to_win: int,
    discard_limit: int,
    turns_limit: int,
    probe_stride: int = 1,
    value_scale: float = 1.0,
    tree_reuse: bool = False,
    canonical_pruning: bool = False,
    game_opponent: GameOpponent = "raw",
    search_selection: str = "puct",
    c_visit: float = 50.0,
    c_scale: float = 1.0,
) -> dict[str, Any]:
    return {
        "map_type": map_type,
        "actor_observation_level": actor_observation_level,
        "critic_observation_level": critic_observation_level,
        "c_puct": c_puct,
        "vps_to_win": vps_to_win,
        "discard_limit": discard_limit,
        "turns_limit": turns_limit,
        "probe_stride": probe_stride,
        "value_scale": value_scale,
        "tree_reuse": tree_reuse,
        "canonical_pruning": canonical_pruning,
        "game_opponent": game_opponent,
        "search_selection": search_selection,
        "c_visit": c_visit,
        "c_scale": c_scale,
    }


def run_native_budget_games(
    *,
    policy_model,
    critic_model,
    model_type: str,
    map_type: MapType,
    actor_observation_level: ActorObservationLevel,
    critic_observation_level: CriticObservationLevel,
    budget: int,
    games_per_seat: int,
    num_workers: int,
    inference_batch_size: int,
    inference_wait_ms: float,
    c_puct: float,
    seed: int,
    vps_to_win: int,
    discard_limit: int,
    device: str | torch.device,
    turns_limit: int = TURNS_LIMIT,
    show_tqdm: bool = True,
    value_scale: float = 1.0,
    tree_reuse: bool = False,
    canonical_pruning: bool = False,
    game_opponent: GameOpponent = "raw",
    search_selection: str = "puct",
    c_visit: float = 50.0,
    c_scale: float = 1.0,
) -> NativeBudgetGameResult:
    if budget < 1:
        raise ValueError("budget must be at least 1")
    if games_per_seat < 1:
        raise ValueError("games_per_seat must be at least 1")
    if game_opponent not in ("raw", "value"):
        raise ValueError("game_opponent must be 'raw' or 'value'")
    if search_selection not in ("puct", "completed-q"):
        raise ValueError("search_selection must be 'puct' or 'completed-q'")
    scenarios = [
        (seat, derive_seed(seed, "native_budget_episode", game_index))
        for seat in range(2)
        for game_index in range(games_per_seat)
    ]
    assignments = _distribute(scenarios, num_workers)
    args_dict = _common_args(
        map_type=map_type,
        actor_observation_level=actor_observation_level,
        critic_observation_level=critic_observation_level,
        c_puct=c_puct,
        vps_to_win=vps_to_win,
        discard_limit=discard_limit,
        turns_limit=turns_limit,
        value_scale=value_scale,
        tree_reuse=tree_reuse,
        canonical_pruning=canonical_pruning,
        game_opponent=game_opponent,
        search_selection=search_selection,
        c_visit=c_visit,
        c_scale=c_scale,
    )
    aggregate = NativeBudgetGameResult()

    def handle_result(message: dict[str, Any]) -> None:
        aggregate.merge(message["result"])

    error = run_inference_server_workers(
        policy_model=policy_model,
        critic_model=critic_model,
        model_type=model_type,
        device=device,
        num_workers=len(assignments),
        inference_batch_size=inference_batch_size,
        inference_wait_ms=inference_wait_ms,
        worker_target=_game_worker_main,
        worker_args=[(assignment, budget, args_dict) for assignment in assignments],
        handle_result=handle_result,
        total=len(scenarios),
        show_tqdm=show_tqdm,
    )
    if error is not None:
        raise RuntimeError(f"Native MCTS budget game worker failed:\n{error}")
    if len(aggregate.game_records) != len(scenarios):
        raise RuntimeError(
            f"Native MCTS budget evaluation returned {len(aggregate.game_records)} games, "
            f"expected {len(scenarios)}"
        )
    return aggregate


def run_native_budget_position_probes(
    *,
    policy_model,
    critic_model,
    model_type: str,
    map_type: MapType,
    actor_observation_level: ActorObservationLevel,
    critic_observation_level: CriticObservationLevel,
    budgets: Sequence[int],
    num_games: int,
    probe_stride: int,
    num_workers: int,
    inference_batch_size: int,
    inference_wait_ms: float,
    c_puct: float,
    seed: int,
    vps_to_win: int,
    discard_limit: int,
    device: str | torch.device,
    turns_limit: int = TURNS_LIMIT,
    show_tqdm: bool = True,
    value_scale: float = 1.0,
    canonical_pruning: bool = False,
    search_selection: str = "puct",
    c_visit: float = 50.0,
    c_scale: float = 1.0,
) -> NativeBudgetProbeResult:
    if not budgets or any(int(budget) < 1 for budget in budgets):
        raise ValueError("budgets must contain positive simulation counts")
    if num_games < 1:
        raise ValueError("num_games must be at least 1")
    if probe_stride < 1:
        raise ValueError("probe_stride must be at least 1")
    if search_selection not in ("puct", "completed-q"):
        raise ValueError("search_selection must be 'puct' or 'completed-q'")
    indexed_seeds = [
        (game_index, derive_seed(seed, "native_probe_episode", game_index))
        for game_index in range(num_games)
    ]
    assignments = _distribute(indexed_seeds, num_workers)
    args_dict = _common_args(
        map_type=map_type,
        actor_observation_level=actor_observation_level,
        critic_observation_level=critic_observation_level,
        c_puct=c_puct,
        vps_to_win=vps_to_win,
        discard_limit=discard_limit,
        turns_limit=turns_limit,
        probe_stride=probe_stride,
        value_scale=value_scale,
        canonical_pruning=canonical_pruning,
        search_selection=search_selection,
        c_visit=c_visit,
        c_scale=c_scale,
    )
    aggregate = NativeBudgetProbeResult()

    def handle_result(message: dict[str, Any]) -> None:
        aggregate.merge(message["result"])

    error = run_inference_server_workers(
        policy_model=policy_model,
        critic_model=critic_model,
        model_type=model_type,
        device=device,
        num_workers=len(assignments),
        inference_batch_size=inference_batch_size,
        inference_wait_ms=inference_wait_ms,
        worker_target=_probe_worker_main,
        worker_args=[(assignment, tuple(budgets), args_dict) for assignment in assignments],
        handle_result=handle_result,
        total=num_games,
        show_tqdm=show_tqdm,
    )
    if error is not None:
        raise RuntimeError(f"Native MCTS budget probe worker failed:\n{error}")
    return aggregate


__all__ = [
    "CriticCalibrationAccumulator",
    "NativeBudgetGameResult",
    "NativeBudgetProbeResult",
    "SearchDiagnosticsAccumulator",
    "run_native_budget_games",
    "run_native_budget_position_probes",
]
