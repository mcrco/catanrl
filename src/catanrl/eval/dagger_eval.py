"""
Frozen-set imitation eval for DAgger against ``ValueFunctionPlayer`` (``F``).

Builds a fixed pool of non-forced decision points from F-vs-F self-play, scores
each legal action with F's heuristic value function once, then evaluates any
policy checkpoint with a batched forward pass. Metrics are low-variance and
comparable across model sizes on the same state set:

- ``imitation/value_regret_norm``: normalized value gap for the model's action
- ``imitation/expert_ce``: NLL of F's action under the masked policy
- ``imitation/agreement`` / ``imitation/top3_agreement``: top-1 / top-3 match
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Sequence

import numpy as np
import torch

from catanatron.game import Game
from catanatron.players.value import ValueFunctionPlayer, get_value_fn

from ..features.catanatron_utils import (
    COLOR_ORDER,
    full_game_to_features,
    game_to_features,
    get_observation_indices_from_full,
)
from ..models.wrappers import PolicyNetworkWrapper, PolicyValueNetworkWrapper
from ..utils.catanatron_action_space import to_action_space
from ..utils.catanatron_map import build_catan_map
from ..utils.seeding import derive_map_and_game_seeds, derive_seed
from .vectorized_rollout import _get_policy_logits

_EPS = 1e-8


@dataclass(frozen=True)
class DecisionPoint:
    """A single non-forced decision encountered during F-vs-F self-play."""

    private_features: np.ndarray
    full_features: np.ndarray
    legal_indices: np.ndarray
    expert_values: np.ndarray
    expert_offset: int


@dataclass
class FrozenImitationEvalSet:
    """Fixed decision points plus metadata for repeatable imitation scoring."""

    decision_points: List[DecisionPoint]
    map_type: str
    num_players: int
    vps_to_win: int
    discard_limit: int
    num_games: int
    seed: int

    @classmethod
    def generate(
        cls,
        map_type: Literal["BASE", "MINI", "TOURNAMENT"],
        num_players: int,
        vps_to_win: int,
        discard_limit: int,
        num_games: int,
        max_decision_points: int,
        seed: int,
        show_progress: bool = False,
    ) -> "FrozenImitationEvalSet":
        decision_points = _generate_decision_points(
            map_type=map_type,
            num_players=num_players,
            vps_to_win=vps_to_win,
            discard_limit=discard_limit,
            num_games=num_games,
            max_decision_points=max_decision_points,
            seed=seed,
            show_progress=show_progress,
        )
        return cls(
            decision_points=decision_points,
            map_type=map_type,
            num_players=num_players,
            vps_to_win=vps_to_win,
            discard_limit=discard_limit,
            num_games=num_games,
            seed=seed,
        )

    def observations_for(self, observation_level: str) -> np.ndarray:
        return build_observation_matrix(
            self.decision_points,
            self.num_players,
            self.map_type,
            observation_level,
        )

    def eval_policy(
        self,
        policy_model: PolicyNetworkWrapper | PolicyValueNetworkWrapper,
        model_type: str,
        observation_level: str,
        device: torch.device | str,
        batch_size: int = 4096,
    ) -> Dict[str, float]:
        torch_device = torch.device(device)
        observations = self.observations_for(observation_level)
        return eval_policy_imitation(
            policy_model=policy_model,
            model_type=model_type,
            decision_points=self.decision_points,
            observations=observations,
            device=torch_device,
            batch_size=batch_size,
        )


class _RecordingValuePlayer(ValueFunctionPlayer):
    def __init__(self, color, sink: List[DecisionPoint], map_type: str, max_points: int):
        super().__init__(color)
        self._sink = sink
        self._map_type = map_type
        self._max_points = max_points

    def decide(self, game, playable_actions):
        if len(playable_actions) == 1:
            return playable_actions[0]

        value_fn = get_value_fn(self.value_fn_builder_name, self.params)
        values = np.empty(len(playable_actions), dtype=np.float64)
        for i, action in enumerate(playable_actions):
            game_copy = game.copy()
            game_copy.execute(action)
            values[i] = value_fn(game_copy, self.color)
        best_offset = int(np.argmax(values))

        if len(self._sink) < self._max_points:
            num_players = len(game.state.colors)
            game_colors = tuple(game.state.colors)
            legal_indices = np.array(
                [
                    to_action_space(a, num_players, self._map_type, game_colors)
                    for a in playable_actions
                ],
                dtype=np.int64,
            )
            self._sink.append(
                DecisionPoint(
                    private_features=np.asarray(
                        game_to_features(game, self.color, num_players, self._map_type),
                        dtype=np.float32,
                    ),
                    full_features=np.asarray(
                        full_game_to_features(
                            game, num_players, self._map_type, base_color=self.color
                        ),
                        dtype=np.float32,
                    ),
                    legal_indices=legal_indices,
                    expert_values=values.copy(),
                    expert_offset=best_offset,
                )
            )

        return playable_actions[best_offset]


def _generate_decision_points(
    map_type: str,
    num_players: int,
    vps_to_win: int,
    discard_limit: int,
    num_games: int,
    max_decision_points: int,
    seed: int,
    show_progress: bool,
) -> List[DecisionPoint]:
    if num_players != 2:
        raise ValueError("Frozen imitation eval assumes 2-player F-vs-F games.")

    sink: List[DecisionPoint] = []
    episode_seeds = [derive_seed(seed, "imitation_episode", g) for g in range(num_games)]

    iterator = episode_seeds
    if show_progress:
        from tqdm import tqdm

        iterator = tqdm(episode_seeds, desc="imitation eval (F vs F)", unit="game")

    for episode_seed in iterator:
        if len(sink) >= max_decision_points:
            break
        players = [
            _RecordingValuePlayer(COLOR_ORDER[i], sink, map_type, max_decision_points)
            for i in range(num_players)
        ]
        for player in players:
            player.reset_state()
        map_seed, game_seed = derive_map_and_game_seeds(episode_seed)
        game = Game(
            players=players,
            catan_map=build_catan_map(map_type, seed=map_seed, number_placement="random"),
            seed=game_seed,
            discard_limit=discard_limit,
            vps_to_win=vps_to_win,
        )
        game.play()

    return sink[:max_decision_points]


def build_observation_matrix(
    decision_points: Sequence[DecisionPoint],
    num_players: int,
    map_type: str,
    observation_level: str,
) -> np.ndarray:
    if observation_level == "private":
        return np.stack([dp.private_features for dp in decision_points], axis=0)
    indices = get_observation_indices_from_full(num_players, map_type, observation_level)
    full = np.stack([dp.full_features for dp in decision_points], axis=0)
    return full[:, indices]


def eval_policy_imitation(
    policy_model: PolicyNetworkWrapper | PolicyValueNetworkWrapper,
    model_type: str,
    decision_points: Sequence[DecisionPoint],
    observations: np.ndarray,
    device: torch.device,
    batch_size: int = 4096,
) -> Dict[str, float]:
    """Score a policy on a frozen imitation set."""
    n = len(decision_points)
    if n == 0:
        raise ValueError("No decision points to score.")

    policy_model.eval()
    logits_chunks: List[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, n, batch_size):
            chunk = torch.from_numpy(observations[start : start + batch_size]).float().to(device)
            logits = _get_policy_logits(policy_model, chunk, model_type)
            logits_chunks.append(logits.cpu().numpy())
    all_logits = np.concatenate(logits_chunks, axis=0)

    agree = np.zeros(n, dtype=np.float64)
    top3 = np.zeros(n, dtype=np.float64)
    regret = np.zeros(n, dtype=np.float64)
    ce = np.zeros(n, dtype=np.float64)
    p_expert = np.zeros(n, dtype=np.float64)

    for i, dp in enumerate(decision_points):
        legal_logits = all_logits[i, dp.legal_indices]
        shifted = legal_logits - legal_logits.max()
        exp = np.exp(shifted)
        probs = exp / (exp.sum() + _EPS)

        model_offset = int(np.argmax(legal_logits))
        expert_offset = dp.expert_offset

        agree[i] = float(model_offset == expert_offset)
        k = min(3, legal_logits.shape[0])
        topk = np.argpartition(legal_logits, -k)[-k:]
        top3[i] = float(expert_offset in topk)

        values = dp.expert_values
        best_value = values[expert_offset]
        denom = best_value - values.min()
        regret[i] = (
            0.0 if denom <= _EPS else float((best_value - values[model_offset]) / denom)
        )

        prob_expert = float(probs[expert_offset])
        p_expert[i] = prob_expert
        ce[i] = -float(np.log(prob_expert + _EPS))

    return {
        "imitation/value_regret_norm": float(regret.mean()),
        "imitation/expert_ce": float(ce.mean()),
        "imitation/expert_action_prob": float(p_expert.mean()),
        "imitation/agreement": float(agree.mean()),
        "imitation/top3_agreement": float(top3.mean()),
        "imitation/num_decision_points": float(n),
    }


def expert_supports_frozen_imitation_eval(expert_config: str) -> bool:
    """True when the training expert is ``ValueFunctionPlayer`` (``F``)."""
    return expert_config.strip().split(":")[0] == "F"


def is_better_policy_checkpoint(
    value_regret: float,
    best_value_regret: float,
    *,
    tol: float = 1e-6,
) -> bool:
    """Return True when ``value_regret`` improves the saved policy checkpoint."""
    return value_regret < best_value_regret - tol


def is_better_critic_checkpoint(
    explained_variance: float,
    best_explained_variance: float,
    *,
    tol: float = 1e-6,
) -> bool:
    """Return True when ``explained_variance`` improves the saved critic checkpoint."""
    return explained_variance > best_explained_variance + tol


def is_better_imitation_checkpoint(
    value_regret: float,
    expert_ce: float,
    best_value_regret: float,
    best_expert_ce: float,
    *,
    regret_tol: float = 1e-6,
    ce_tol: float = 1e-6,
) -> bool:
    """Return True when ``(value_regret, expert_ce)`` lexicographically improves."""
    if value_regret < best_value_regret - regret_tol:
        return True
    if value_regret > best_value_regret + regret_tol:
        return False
    return expert_ce < best_expert_ce - ce_tol
