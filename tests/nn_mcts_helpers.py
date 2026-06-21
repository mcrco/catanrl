"""Shared mocks for NN MCTS tests (no real policy/critic networks)."""

from __future__ import annotations

from typing import Literal

import numpy as np
from catanatron.models.player import Color

from catanrl.features.catanatron_utils import COLOR_ORDER
from catanrl.players import NNMCTSPlayer
from catanrl.players.nn_mcts_player import (
    _LeafEvaluation,
    _NNMCTSInferenceBackend,
)
from catanrl.utils.catanatron_action_space import get_action_space_size


class MockInferenceBackend(_NNMCTSInferenceBackend):
    """Returns fixed logits + value for any leaf; ignores feature vectors."""

    def __init__(
        self,
        num_actions: int,
        *,
        value: float = 0.0,
        policy_logits: np.ndarray | None = None,
    ) -> None:
        self.num_actions = num_actions
        self.value = float(value)
        if policy_logits is None:
            self.policy_logits = np.zeros(num_actions, dtype=np.float32)
        else:
            logits = np.asarray(policy_logits, dtype=np.float32).reshape(-1)
            if logits.shape[0] != num_actions:
                raise ValueError(
                    f"policy_logits length {logits.shape[0]} != num_actions {num_actions}"
                )
            self.policy_logits = logits

    def evaluate_leaf(
        self,
        actor_features: np.ndarray,
        critic_features: np.ndarray,
    ) -> _LeafEvaluation:
        del actor_features, critic_features
        return _LeafEvaluation(
            policy_logits=self.policy_logits.copy(),
            value=self.value,
        )


def action_space_size(
    num_players: int,
    map_type: Literal["BASE", "MINI", "TOURNAMENT"],
) -> int:
    return get_action_space_size(num_players, map_type)


def build_mock_player(
    *,
    map_type: Literal["BASE", "MINI", "TOURNAMENT"] = "MINI",
    num_players: int = 2,
    color: Color = COLOR_ORDER[0],
    num_simulations: int = 16,
    inference_backend: _NNMCTSInferenceBackend | None = None,
) -> NNMCTSPlayer:
    num_actions = action_space_size(num_players, map_type)
    backend = inference_backend or MockInferenceBackend(num_actions)
    return NNMCTSPlayer(
        color=color,
        model_type="flat",
        policy_model=None,
        critic_model=None,
        map_type=map_type,
        num_simulations=num_simulations,
        actor_observation_level="private",
        critic_observation_level="full",
        inference_backend=backend,
    )
