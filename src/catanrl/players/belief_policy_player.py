"""A player that marginalizes a full-information policy over a belief.

Instead of feeding partial information to the policy network, this player wraps a
policy network trained on *full* (perfect) information and, at decision time,
averages the policy over the belief about the opponents' hidden dev cards:

    pi_bar(a) = sum_h  P(h) * pi(a | determinize(state, h))

For 1v1 the belief is enumerated exactly (multivariate hypergeometric over the
unseen dev pool, pruned by the "opponent hasn't won" constraint), so this is the
exact belief-weighted marginal, not an estimate. The action is then chosen by
argmax over the marginal by default (set ``sample=True`` to sample instead).

This realizes the "train on full info, marginalize at test time" idea (Perfect
Information Monte Carlo / averaging).
"""

from __future__ import annotations

import random
from typing import Dict, Literal, Optional

import numpy as np
import torch
from catanatron.models.player import Color, Player

from catanrl.beliefs.determinize import patch_full_features
from catanrl.beliefs.dev_card_belief import DevCardBelief
from catanrl.features.catanatron_utils import (
    full_game_to_features,
    get_full_numeric_feature_names,
)
from catanrl.models import PolicyNetworkWrapper
from catanrl.utils.catanatron_action_space import to_action_space


class BeliefAveragedPolicyPlayer(Player):
    """Marginalize a full-info policy network over the dev-card belief.

    Args:
        color: This player's color.
        model_type: "flat" or "hierarchical".
        model: A policy network trained on the ``full`` observation level.
        map_type: Catan map variant.
        num_samples: Number of belief samples to use for games with >2 players
            (ignored in 1v1, where the belief is enumerated exactly).
        sample: If True, sample the action from the marginal; otherwise argmax.
        seed: Optional RNG seed for reproducibility.
    """

    def __init__(
        self,
        color: Color,
        model_type: str,
        model: PolicyNetworkWrapper,
        map_type: Literal["BASE", "MINI", "TOURNAMENT"] = "BASE",
        num_samples: int = 100,
        sample: bool = False,
        seed: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(color, is_bot=True, **kwargs)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_type = model_type
        self.map_type = map_type
        self.num_samples = num_samples
        self.sample = sample
        self.model = model.to(self.device)
        self.model.eval()
        self._rng = random.Random(seed)
        self._numeric_index_cache: Dict[int, Dict[str, int]] = {}

    def _numeric_name_to_index(self, num_players: int) -> Dict[str, int]:
        cached = self._numeric_index_cache.get(num_players)
        if cached is None:
            names = get_full_numeric_feature_names(num_players, self.map_type)
            cached = {name: idx for idx, name in enumerate(names)}
            self._numeric_index_cache[num_players] = cached
        return cached

    def decide(self, game, playable_actions):
        if len(playable_actions) == 1:
            return playable_actions[0]

        num_players = len(game.state.colors)
        colors = tuple(game.state.colors)
        belief = DevCardBelief(game, self.color)
        if num_players == 2:
            hypotheses = belief.enumerate_hypotheses()
        else:
            hypotheses = belief.sample_hypotheses(self.num_samples, self._rng)

        # Build full-info features once, then patch opponent dev slots per
        # hypothesis. The board tensor and legal-action mask are identical across
        # hypotheses, so this is a single batched forward pass.
        base_full = full_game_to_features(
            game, num_players, self.map_type, base_color=self.color
        )
        numeric_index = self._numeric_name_to_index(num_players)
        batch = np.stack(
            [
                patch_full_features(
                    base_full,
                    hyp,
                    base_color=self.color,
                    colors=colors,
                    numeric_name_to_index=numeric_index,
                )
                for hyp in hypotheses
            ]
        )

        states = torch.from_numpy(batch).to(self.device)
        with torch.no_grad():
            if self.model_type == "flat":
                logits = self.model(states)
            elif self.model_type == "hierarchical":
                action_type_logits, param_logits = self.model(states)
                logits = self.model.get_flat_action_logits(
                    action_type_logits, param_logits
                )
            else:
                raise ValueError(f"Unknown model_type '{self.model_type}'")
        logits = logits.float().cpu().numpy()  # [H, A]

        playable_indices = [
            to_action_space(action, num_players, self.map_type, colors)
            for action in playable_actions
        ]
        # Masked softmax over playable actions per hypothesis, then belief-weighted
        # average to get the marginal over legal actions.
        masked_logits = logits[:, playable_indices]  # [H, L]
        masked_logits = masked_logits - masked_logits.max(axis=1, keepdims=True)
        probs = np.exp(masked_logits)
        probs /= probs.sum(axis=1, keepdims=True)
        weights = np.array([h.weight for h in hypotheses], dtype=np.float64)
        weights /= weights.sum()
        marginal = (weights[:, None] * probs).sum(axis=0)  # [L]

        if self.sample:
            offset = int(self._rng.choices(range(len(playable_actions)), weights=marginal)[0])
        else:
            offset = int(np.argmax(marginal))
        return playable_actions[offset]

    def __repr__(self) -> str:
        return super().__repr__().replace("Player", "BeliefAveragedPolicyPlayer")
