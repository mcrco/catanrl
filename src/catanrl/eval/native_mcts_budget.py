"""Native MCTS budget diagnostics and frozen-policy strength evaluation."""

from __future__ import annotations

import multiprocessing as mp
import traceback
from collections.abc import Iterator, Sequence
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import numpy as np
import torch
from catanatron.game import Game
from catanatron.models.enums import DEVELOPMENT_CARDS, RESOURCES, ActionType
from catanatron.models.player import Color, SimplePlayer
from catanatron.players.value import ValueFunctionPlayer
from catanatron.state_functions import get_actual_victory_points

from catanrl.algorithms.alphazero.native_search import (
    NativeSearchDiagnostics,
    reconcile_search_after_observed_step,
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
    full_game_to_features,
    get_observation_indices_from_full,
)
from catanrl.players.nn_mcts_player import _RemoteNNMCTSInferenceBackend
from catanrl.utils.catanatron_action_space import (
    canopy_action_count_increment,
    from_action_space,
    to_action_space,
)
from catanrl.utils.catanatron_game import force_player_order
from catanrl.utils.catanatron_map import build_catan_map_from_native_game
from catanrl.utils.seeding import derive_map_and_game_seeds, derive_seed

MapType = Literal["BASE", "MINI", "TOURNAMENT"]
GameOpponent = Literal["random", "raw", "value"]
AuthoritativeEngine = Literal["native", "catanatron"]

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
    rng: np.random.Generator | None = None,
) -> int:
    if opponent == "random":
        if rng is None:
            raise ValueError("A random generator is required for the random opponent")
        valid_indices = np.flatnonzero(game.valid_action_mask())
        if valid_indices.size == 0:
            raise RuntimeError("Random native opponent has no legal action")
        return int(rng.choice(valid_indices))
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
        add_noise=float(args_dict.get("root_dirichlet_fraction", 0.0)) > 0.0,
        dirichlet_alpha=float(args_dict.get("root_dirichlet_alpha", 0.3)),
        dirichlet_frac=float(args_dict.get("root_dirichlet_fraction", 0.0)),
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
    action_count = 0
    max_actions = int(args_dict.get("max_actions", 0))
    opponent_rng = np.random.default_rng(derive_seed(episode_seed, "native_budget_random_opponent"))
    search: NativeMCTSSearch | None = None
    try:
        while (
            game.winner is None
            and game.num_turns < int(args_dict["turns_limit"])
            and (max_actions <= 0 or action_count < max_actions)
        ):
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
                    opponent_rng,
                )
            search = step_game_and_reconcile_search(
                game=game,
                map_type=args_dict["map_type"],
                action=action,
                search=search,
            )
            decision_index += 1
            action_count += canopy_action_count_increment(action, 2, args_dict["map_type"])

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
                "actions": action_count,
                "native_decisions": decision_index,
                "opponent": args_dict["game_opponent"],
            },
            diagnostics,
            calibration,
        )
    finally:
        if search is not None:
            search.close()
        game.close()


def _catanatron_native_mask(game: Game, map_type: MapType, action_space_size: int) -> np.ndarray:
    mask = np.zeros(action_space_size, dtype=np.bool_)
    colors = tuple(game.state.colors)
    for action in game.playable_actions:
        mask[to_action_space(action, 2, map_type, colors)] = True
    return mask


def _replay_native_action(
    native: NativeGame, action: int, action_type: ActionType, result: Any
) -> None:
    kwargs: dict[str, Any] = {}
    if action_type == ActionType.ROLL:
        kwargs["dice"] = tuple(int(value) for value in result)
    elif action_type == ActionType.BUY_DEVELOPMENT_CARD:
        kwargs["development_card"] = DEVELOPMENT_CARDS.index(result)
    elif action_type == ActionType.MOVE_ROBBER and result is not None:
        kwargs["stolen_resource"] = RESOURCES.index(result)
    native.step(action, **kwargs)


def _play_catanatron_shadow_budget_game(
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
    """Run native search while Catanatron owns every actual state transition."""

    map_seed, game_seed = derive_map_and_game_seeds(episode_seed)
    diagnostics = SearchDiagnosticsAccumulator()
    calibration_rows: list[tuple[float, int]] = []
    calibration = CriticCalibrationAccumulator()
    native = NativeGame(
        2,
        args_dict["map_type"],
        seed=game_seed,
        map_seed=map_seed,
        number_placement="random",
        discard_limit=int(args_dict["discard_limit"]),
        vps_to_win=int(args_dict["vps_to_win"]),
    )
    candidate = SimplePlayer(Color.RED)
    opponent = ValueFunctionPlayer(Color.BLUE)
    players = [candidate, opponent] if mcts_seat == 0 else [opponent, candidate]
    game = Game(
        players=players,
        catan_map=build_catan_map_from_native_game(native, args_dict["map_type"]),
        seed=game_seed,
        discard_limit=int(args_dict["discard_limit"]),
        vps_to_win=int(args_dict["vps_to_win"]),
    )
    force_player_order(game, players)
    decision_index = 0
    action_count = 0
    max_actions = int(args_dict.get("max_actions", 0))
    opponent_rng = np.random.default_rng(derive_seed(episode_seed, "native_budget_random_opponent"))
    search: NativeMCTSSearch | None = None
    try:
        while (
            game.winning_color() is None
            and game.state.num_turns < int(args_dict["turns_limit"])
            and (max_actions <= 0 or action_count < max_actions)
        ):
            python_mask = _catanatron_native_mask(
                game, args_dict["map_type"], native.action_space_size
            )
            native_mask = native.valid_action_mask()
            if not np.array_equal(native_mask, python_mask):
                differing = np.flatnonzero(native_mask != python_mask).tolist()
                raise RuntimeError(
                    f"Catanatron/native legal actions diverged at decision {decision_index}: "
                    f"{differing}"
                )
            python_features = full_game_to_features(
                game,
                2,
                args_dict["map_type"],
                base_color=game.state.current_color(),
            )
            native_features = full_native_features(
                native,
                args_dict["map_type"],
                native.current_player,
            )
            if not np.allclose(native_features, python_features, rtol=0.0, atol=1e-7):
                differing = np.flatnonzero(
                    ~np.isclose(native_features, python_features, rtol=0.0, atol=1e-7)
                )
                first = differing[:16].tolist()
                raise RuntimeError(
                    "Catanatron/native feature representations diverged at decision "
                    f"{decision_index}; first differing indices: {first}"
                )
            valid_indices = np.flatnonzero(native_mask)
            if valid_indices.size == 0:
                raise RuntimeError("Catanatron parity game has no legal action")

            if valid_indices.size == 1:
                action_index = int(valid_indices[0])
            elif native.current_player == mcts_seat:
                if search is None and bool(args_dict.get("tree_reuse", False)):
                    search = NativeMCTSSearch(
                        native,
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
                    game=native,
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
                calibration_rows.append((search_result.diagnostics.network_value, mcts_seat))
                action_index = search_result.action
            elif args_dict["game_opponent"] == "value":
                opponent_action = game.state.current_player().decide(game, game.playable_actions)
                action_index = to_action_space(
                    opponent_action,
                    2,
                    args_dict["map_type"],
                    tuple(game.state.colors),
                )
            elif args_dict["game_opponent"] == "random":
                action_index = int(opponent_rng.choice(valid_indices))
            else:
                action_index = _raw_policy_action(
                    native,
                    args_dict["map_type"],
                    actor_indices,
                    critic_indices,
                    inference_backend,
                )

            catanatron_action = from_action_space(
                action_index,
                game.state.current_color(),
                2,
                args_dict["map_type"],
                tuple(game.state.colors),
                game.playable_actions,
            )
            if catanatron_action not in game.playable_actions:
                raise RuntimeError(
                    f"Native action {action_index} is not an authoritative Catanatron action: "
                    f"{catanatron_action}"
                )
            record = game.execute(catanatron_action)
            _replay_native_action(
                native,
                action_index,
                catanatron_action.action_type,
                record.result,
            )
            search = reconcile_search_after_observed_step(
                game=native,
                map_type=args_dict["map_type"],
                action=action_index,
                search=search,
            )
            decision_index += 1
            action_count += canopy_action_count_increment(action_index, 2, args_dict["map_type"])

        winner_color = game.winning_color()
        winner = None if winner_color is None else game.state.color_to_index[winner_color]
        if native.winner != winner:
            raise RuntimeError(f"Catanatron/native winners diverged: {winner} != {native.winner}")
        for prediction, player in calibration_rows:
            target = 0.0 if winner is None else (1.0 if winner == player else -1.0)
            calibration.add(prediction, target)
        candidate_vps = get_actual_victory_points(game.state, candidate.color)
        return (
            {
                "budget": budget,
                "seat": "first" if mcts_seat == 0 else "second",
                "episode_seed": episode_seed,
                "win": winner == mcts_seat,
                "draw": winner is None,
                "winner": winner,
                "vps": int(candidate_vps),
                "total_vps": int(
                    sum(get_actual_victory_points(game.state, color) for color in game.state.colors)
                ),
                "turns": int(game.state.num_turns),
                "actions": action_count,
                "native_decisions": decision_index,
                "opponent": args_dict["game_opponent"],
                "authoritative_engine": "Catanatron",
            },
            diagnostics,
            calibration,
        )
    finally:
        if search is not None:
            search.close()
        native.close()


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
        for record, diagnostics, calibration in _iter_budget_game_results(
            scenarios=scenarios,
            budget=budget,
            args_dict=args_dict,
            actor_indices=actor_indices,
            critic_indices=critic_indices,
            inference_backend=inference_backend,
            game_concurrency=int(args_dict.get("games_per_worker", 1)),
        ):
            result = NativeBudgetGameResult()
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


def _iter_budget_game_results(
    *,
    scenarios: Sequence[tuple[int, int]],
    budget: int,
    args_dict: dict[str, Any],
    actor_indices: np.ndarray,
    critic_indices: np.ndarray,
    inference_backend: _RemoteNNMCTSInferenceBackend,
    game_concurrency: int,
) -> Iterator[
    tuple[
        dict[str, Any],
        SearchDiagnosticsAccumulator,
        CriticCalibrationAccumulator,
    ]
]:
    """Yield games while multiplexing independent searches in one process."""
    if game_concurrency < 1:
        raise ValueError("game_concurrency must be at least 1")

    def play(
        scenario: tuple[int, int],
    ) -> tuple[
        dict[str, Any],
        SearchDiagnosticsAccumulator,
        CriticCalibrationAccumulator,
    ]:
        mcts_seat, episode_seed = scenario
        play_game = (
            _play_catanatron_shadow_budget_game
            if args_dict.get("authoritative_engine") == "catanatron"
            else _play_budget_game
        )
        return play_game(
            episode_seed=episode_seed,
            mcts_seat=mcts_seat,
            budget=budget,
            args_dict=args_dict,
            actor_indices=actor_indices,
            critic_indices=critic_indices,
            inference_backend=inference_backend,
        )

    if game_concurrency == 1 or len(scenarios) <= 1:
        for scenario in scenarios:
            yield play(scenario)
        return

    executor = ThreadPoolExecutor(
        max_workers=min(game_concurrency, len(scenarios)),
        thread_name_prefix="native-budget-game",
    )
    futures: list[
        Future[
            tuple[
                dict[str, Any],
                SearchDiagnosticsAccumulator,
                CriticCalibrationAccumulator,
            ]
        ]
    ] = [executor.submit(play, scenario) for scenario in scenarios]
    try:
        for future in as_completed(futures):
            yield future.result()
    finally:
        # The coordinator terminates a failed worker; report errors immediately
        # instead of waiting for another long search game to finish.
        executor.shutdown(wait=False, cancel_futures=True)


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
            action_count = 0
            max_actions = int(args_dict.get("max_actions", 0))
            probe_index = 0
            try:
                while (
                    game.winner is None
                    and game.num_turns < int(args_dict["turns_limit"])
                    and (max_actions <= 0 or action_count < max_actions)
                ):
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
                    action_count += canopy_action_count_increment(
                        action,
                        2,
                        args_dict["map_type"],
                    )
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
    max_actions: int = 0,
    probe_stride: int = 1,
    value_scale: float = 1.0,
    tree_reuse: bool = False,
    canonical_pruning: bool = False,
    game_opponent: GameOpponent = "raw",
    search_selection: str = "puct",
    c_visit: float = 50.0,
    c_scale: float = 1.0,
    root_dirichlet_alpha: float = 0.3,
    root_dirichlet_fraction: float = 0.0,
    authoritative_engine: AuthoritativeEngine = "native",
) -> dict[str, Any]:
    return {
        "map_type": map_type,
        "actor_observation_level": actor_observation_level,
        "critic_observation_level": critic_observation_level,
        "c_puct": c_puct,
        "vps_to_win": vps_to_win,
        "discard_limit": discard_limit,
        "turns_limit": turns_limit,
        "max_actions": max_actions,
        "probe_stride": probe_stride,
        "value_scale": value_scale,
        "tree_reuse": tree_reuse,
        "canonical_pruning": canonical_pruning,
        "game_opponent": game_opponent,
        "search_selection": search_selection,
        "c_visit": c_visit,
        "c_scale": c_scale,
        "root_dirichlet_alpha": root_dirichlet_alpha,
        "root_dirichlet_fraction": root_dirichlet_fraction,
        "authoritative_engine": authoritative_engine,
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
    games_per_worker: int = 1,
    inference_batch_size: int,
    inference_wait_ms: float,
    c_puct: float,
    seed: int,
    vps_to_win: int,
    discard_limit: int,
    device: str | torch.device,
    turns_limit: int = TURNS_LIMIT,
    max_actions: int = 0,
    show_tqdm: bool = True,
    value_scale: float = 1.0,
    tree_reuse: bool = False,
    canonical_pruning: bool = False,
    game_opponent: GameOpponent = "raw",
    search_selection: str = "puct",
    c_visit: float = 50.0,
    c_scale: float = 1.0,
    root_dirichlet_alpha: float = 0.3,
    root_dirichlet_fraction: float = 0.0,
    authoritative_engine: AuthoritativeEngine = "native",
) -> NativeBudgetGameResult:
    if budget < 1:
        raise ValueError("budget must be at least 1")
    if games_per_seat < 1:
        raise ValueError("games_per_seat must be at least 1")
    if games_per_worker < 1:
        raise ValueError("games_per_worker must be at least 1")
    if authoritative_engine not in ("native", "catanatron"):
        raise ValueError("authoritative_engine must be 'native' or 'catanatron'")
    if authoritative_engine == "catanatron" and games_per_worker != 1:
        raise ValueError(
            "Catanatron-authoritative games require games_per_worker=1 because "
            "upstream Catanatron uses process-global randomness"
        )
    if max_actions < 0:
        raise ValueError("max_actions cannot be negative")
    if game_opponent not in ("random", "raw", "value"):
        raise ValueError("game_opponent must be 'random', 'raw', or 'value'")
    if search_selection not in ("puct", "completed-q"):
        raise ValueError("search_selection must be 'puct' or 'completed-q'")
    if not np.isfinite(root_dirichlet_alpha) or root_dirichlet_alpha <= 0.0:
        raise ValueError("root_dirichlet_alpha must be finite and positive")
    if (
        not np.isfinite(root_dirichlet_fraction)
        or root_dirichlet_fraction < 0.0
        or root_dirichlet_fraction > 1.0
    ):
        raise ValueError("root_dirichlet_fraction must be finite and in [0, 1]")
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
        max_actions=max_actions,
        value_scale=value_scale,
        tree_reuse=tree_reuse,
        canonical_pruning=canonical_pruning,
        game_opponent=game_opponent,
        search_selection=search_selection,
        c_visit=c_visit,
        c_scale=c_scale,
        root_dirichlet_alpha=root_dirichlet_alpha,
        root_dirichlet_fraction=root_dirichlet_fraction,
        authoritative_engine=authoritative_engine,
    )
    args_dict["games_per_worker"] = games_per_worker
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
    max_actions: int = 0,
    show_tqdm: bool = True,
    value_scale: float = 1.0,
    canonical_pruning: bool = False,
    search_selection: str = "puct",
    c_visit: float = 50.0,
    c_scale: float = 1.0,
    root_dirichlet_alpha: float = 0.3,
    root_dirichlet_fraction: float = 0.0,
) -> NativeBudgetProbeResult:
    if not budgets or any(int(budget) < 1 for budget in budgets):
        raise ValueError("budgets must contain positive simulation counts")
    if num_games < 1:
        raise ValueError("num_games must be at least 1")
    if probe_stride < 1:
        raise ValueError("probe_stride must be at least 1")
    if max_actions < 0:
        raise ValueError("max_actions cannot be negative")
    if search_selection not in ("puct", "completed-q"):
        raise ValueError("search_selection must be 'puct' or 'completed-q'")
    if not np.isfinite(root_dirichlet_alpha) or root_dirichlet_alpha <= 0.0:
        raise ValueError("root_dirichlet_alpha must be finite and positive")
    if (
        not np.isfinite(root_dirichlet_fraction)
        or root_dirichlet_fraction < 0.0
        or root_dirichlet_fraction > 1.0
    ):
        raise ValueError("root_dirichlet_fraction must be finite and in [0, 1]")
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
        max_actions=max_actions,
        probe_stride=probe_stride,
        value_scale=value_scale,
        canonical_pruning=canonical_pruning,
        search_selection=search_selection,
        c_visit=c_visit,
        c_scale=c_scale,
        root_dirichlet_alpha=root_dirichlet_alpha,
        root_dirichlet_fraction=root_dirichlet_fraction,
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
