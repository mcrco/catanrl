"""Shared mocks for NN MCTS tests (no real policy/critic networks)."""

from __future__ import annotations

from typing import Literal

import numpy as np
import torch
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


class FakePolicyModel(torch.nn.Module):
    """Flat policy head mock: uniform logits for any actor feature batch."""

    def __init__(self, num_actions: int) -> None:
        super().__init__()
        self.num_actions = num_actions

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros(x.size(0), self.num_actions, device=x.device, dtype=x.dtype)


class FakeCriticModel(torch.nn.Module):
    """Value head mock: constant zero value for any critic feature batch."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros(x.size(0), 1, device=x.device, dtype=x.dtype)


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


def build_fake_torch_models(
    num_players: int,
    map_type: Literal["BASE", "MINI", "TOURNAMENT"],
) -> tuple[FakePolicyModel, FakeCriticModel]:
    num_actions = action_space_size(num_players, map_type)
    return FakePolicyModel(num_actions), FakeCriticModel()
