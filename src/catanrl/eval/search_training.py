"""Deployable policy evaluation and promotion gates for search-guided training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from catanatron.models.player import Player
from catanatron.players.value import ValueFunctionPlayer

from catanrl.features.catanatron_utils import ActorObservationLevel, COLOR_ORDER
from catanrl.models.wrappers import PolicyNetworkWrapper, PolicyValueNetworkWrapper
from catanrl.models.wrappers import policy_value_to_policy_only
from catanrl.players import BeliefAveragedPolicyPlayer, NNPolicyPlayer
from catanrl.utils.seeding import derive_seed

from .eval_nn_vs_catanatron import eval

PolicyModel = PolicyNetworkWrapper | PolicyValueNetworkWrapper


@dataclass(frozen=True)
class PolicyEvalResult:
    wins: int
    games: int
    first_wins: int
    first_games: int
    second_wins: int
    second_games: int
    avg_vps: float
    avg_turns: float
    first_avg_turns: float
    second_avg_turns: float

    @property
    def win_rate(self) -> float:
        return self.wins / self.games if self.games else 0.0

    @property
    def first_win_rate(self) -> float:
        return self.first_wins / self.first_games if self.first_games else 0.0

    @property
    def second_win_rate(self) -> float:
        return self.second_wins / self.second_games if self.second_games else 0.0

    def metrics(self, prefix: str) -> dict[str, float]:
        return {
            f"{prefix}/wins": float(self.wins),
            f"{prefix}/games": float(self.games),
            f"{prefix}/win_rate": self.win_rate,
            f"{prefix}/first_win_rate": self.first_win_rate,
            f"{prefix}/second_win_rate": self.second_win_rate,
            f"{prefix}/avg_vps": self.avg_vps,
            f"{prefix}/avg_turns": self.avg_turns,
        }


@dataclass(frozen=True)
class PromotionDecision:
    promote: bool
    h2h_win_rate: float
    candidate_baseline_win_rate: float
    champion_baseline_win_rate: float
    reason: str


def decide_promotion(
    *,
    h2h_win_rate: float,
    candidate_baseline_win_rate: float,
    champion_baseline_win_rate: float,
    h2h_threshold: float,
    max_baseline_regression: float,
) -> PromotionDecision:
    """Apply both a direct H2H gate and a fixed-opponent regression guard."""
    h2h_pass = h2h_win_rate >= h2h_threshold
    baseline_floor = champion_baseline_win_rate - max_baseline_regression
    baseline_pass = candidate_baseline_win_rate >= baseline_floor
    if h2h_pass and baseline_pass:
        reason = "candidate passed H2H and fixed-opponent gates"
    elif not h2h_pass and not baseline_pass:
        reason = "candidate failed H2H and fixed-opponent gates"
    elif not h2h_pass:
        reason = "candidate failed H2H gate"
    else:
        reason = "candidate failed fixed-opponent regression guard"
    return PromotionDecision(
        promote=h2h_pass and baseline_pass,
        h2h_win_rate=h2h_win_rate,
        candidate_baseline_win_rate=candidate_baseline_win_rate,
        champion_baseline_win_rate=champion_baseline_win_rate,
        reason=reason,
    )


def _policy_only(model: PolicyModel) -> PolicyNetworkWrapper:
    if isinstance(model, PolicyValueNetworkWrapper):
        return policy_value_to_policy_only(model)
    return model


def _build_deployable_player(
    *,
    color,
    model: PolicyModel,
    model_type: str,
    map_type: Literal["BASE", "MINI", "TOURNAMENT"],
    actor_observation_level: ActorObservationLevel,
    seed: int,
) -> Player:
    policy = _policy_only(model)
    policy.eval()
    device = next(policy.parameters()).device
    if actor_observation_level == "full":
        return BeliefAveragedPolicyPlayer(
            color=color,
            model_type=model_type,
            model=policy,
            map_type=map_type,
            sample=False,
            seed=seed,
            device=device,
        )
    return NNPolicyPlayer(
        color=color,
        model_type=model_type,
        model=policy,
        map_type=map_type,
        actor_observation_level=actor_observation_level,
        device=device,
    )


def _require_balanced_games(num_games: int) -> int:
    if num_games < 2 or num_games % 2:
        raise ValueError("Balanced evaluation requires a positive even number of games.")
    return num_games // 2


def _combine_seats(
    first: tuple[int, list[int], list[int], list[int]],
    second: tuple[int, list[int], list[int], list[int]],
) -> PolicyEvalResult:
    first_wins, first_vps, _, first_turns = first
    second_wins, second_vps, _, second_turns = second
    all_vps = first_vps + second_vps
    all_turns = first_turns + second_turns
    return PolicyEvalResult(
        wins=first_wins + second_wins,
        games=len(all_turns),
        first_wins=first_wins,
        first_games=len(first_turns),
        second_wins=second_wins,
        second_games=len(second_turns),
        avg_vps=sum(all_vps) / len(all_vps) if all_vps else 0.0,
        avg_turns=sum(all_turns) / len(all_turns) if all_turns else 0.0,
        first_avg_turns=sum(first_turns) / len(first_turns) if first_turns else 0.0,
        second_avg_turns=sum(second_turns) / len(second_turns) if second_turns else 0.0,
    )


def evaluate_policy_vs_value(
    *,
    policy_model: PolicyModel,
    model_type: str,
    map_type: Literal["BASE", "MINI", "TOURNAMENT"],
    actor_observation_level: ActorObservationLevel,
    num_players: int,
    num_games: int,
    seed: int,
    vps_to_win: int,
    discard_limit: int,
    show_tqdm: bool = False,
) -> PolicyEvalResult:
    """Evaluate the legally deployable actor against fixed value bots, by seat."""
    games_per_seat = _require_balanced_games(num_games)
    if num_players < 2:
        raise ValueError("Baseline evaluation requires at least two players.")

    def opponents() -> list[Player]:
        return [ValueFunctionPlayer(COLOR_ORDER[index]) for index in range(1, num_players)]

    player = _build_deployable_player(
        color=COLOR_ORDER[0],
        model=policy_model,
        model_type=model_type,
        map_type=map_type,
        actor_observation_level=actor_observation_level,
        seed=seed,
    )
    first = eval(
        player,
        opponents(),
        map_type=map_type,
        num_games=games_per_seat,
        seed=derive_seed(seed, "seat", "first"),
        vps_to_win=vps_to_win,
        discard_limit=discard_limit,
        show_tqdm=show_tqdm,
        nn_seat="first",
    )
    second = eval(
        player,
        opponents(),
        map_type=map_type,
        num_games=games_per_seat,
        seed=derive_seed(seed, "seat", "second"),
        vps_to_win=vps_to_win,
        discard_limit=discard_limit,
        show_tqdm=show_tqdm,
        nn_seat="second",
    )
    return _combine_seats(first, second)


def evaluate_candidate_vs_champion(
    *,
    candidate_model: PolicyModel,
    champion_model: PolicyModel,
    model_type: str,
    map_type: Literal["BASE", "MINI", "TOURNAMENT"],
    actor_observation_level: ActorObservationLevel,
    num_games: int,
    seed: int,
    vps_to_win: int,
    discard_limit: int,
    show_tqdm: bool = False,
) -> PolicyEvalResult:
    """Run balanced, deployable 1v1 games between candidate and champion."""
    games_per_seat = _require_balanced_games(num_games)
    candidate = _build_deployable_player(
        color=COLOR_ORDER[0],
        model=candidate_model,
        model_type=model_type,
        map_type=map_type,
        actor_observation_level=actor_observation_level,
        seed=derive_seed(seed, "candidate"),
    )
    champion = _build_deployable_player(
        color=COLOR_ORDER[1],
        model=champion_model,
        model_type=model_type,
        map_type=map_type,
        actor_observation_level=actor_observation_level,
        seed=derive_seed(seed, "champion"),
    )
    first = eval(
        candidate,
        [champion],
        map_type=map_type,
        num_games=games_per_seat,
        seed=derive_seed(seed, "seat", "first"),
        vps_to_win=vps_to_win,
        discard_limit=discard_limit,
        show_tqdm=show_tqdm,
        nn_seat="first",
    )
    second = eval(
        candidate,
        [champion],
        map_type=map_type,
        num_games=games_per_seat,
        seed=derive_seed(seed, "seat", "second"),
        vps_to_win=vps_to_win,
        discard_limit=discard_limit,
        show_tqdm=show_tqdm,
        nn_seat="second",
    )
    return _combine_seats(first, second)


__all__ = [
    "PolicyEvalResult",
    "PromotionDecision",
    "decide_promotion",
    "evaluate_candidate_vs_champion",
    "evaluate_policy_vs_value",
]
