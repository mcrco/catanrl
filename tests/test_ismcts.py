from __future__ import annotations

import numpy as np
import pytest

from catanatron.models.player import Color, RandomPlayer
from catanatron.state_functions import player_key

from catanrl.utils.catanatron_action_space import get_action_space_size
from catanrl.utils.catanatron_game import force_player_order
from catanrl.utils.catanatron_map import build_catan_map

# The MCTS player pulls in the full model/env stack (torch, pufferlib). Skip the
# whole module if those aren't importable in this environment.
nn_mcts_player = pytest.importorskip("catanrl.players.nn_mcts_player")

from catanatron.game import Game  # noqa: E402
from catanatron.models.enums import KNIGHT  # noqa: E402


class _UniformBackend(nn_mcts_player._NNMCTSInferenceBackend):
    """Fake inference backend: uniform policy logits, neutral value."""

    def __init__(self, action_space_size: int):
        self.action_space_size = action_space_size

    def evaluate_leaf(self, actor_features, critic_features):
        return nn_mcts_player._LeafEvaluation(
            policy_logits=np.zeros(self.action_space_size, dtype=np.float32),
            value=0.0,
        )


def _make_game() -> Game:
    players = [RandomPlayer(Color.BLUE), RandomPlayer(Color.RED)]
    game = Game(players, catan_map=build_catan_map("BASE"))
    force_player_order(game, players)
    return game


def _make_player(num_determinizations: int = 4):
    action_space_size = get_action_space_size(2, "BASE")
    return nn_mcts_player.NNMCTSInformationSetPlayer(
        color=Color.BLUE,
        model_type="flat",
        policy_model=None,
        critic_model=None,
        map_type="BASE",
        num_simulations=4,
        ismcts_determinizations=num_determinizations,
        actor_observation_level="full",
        inference_backend=_UniformBackend(action_space_size),
        device="cpu",
    )


def test_ismcts_decide_returns_legal_action():
    game = _make_game()
    player = _make_player()
    action = player.decide(game, game.playable_actions)
    assert action in game.playable_actions


def test_ismcts_runs_a_search_per_determinization():
    game = _make_game()
    # Give the opponent 2 hidden knights so the belief has multiple hypotheses.
    key = player_key(game.state, Color.RED)
    game.state.player_state[f"{key}_{KNIGHT}_IN_HAND"] = 2
    player = _make_player(num_determinizations=8)

    dets = player._determinizations(game)
    assert len(dets) >= 1
    weights = sum(w for _, w in dets)
    assert weights == pytest.approx(1.0, abs=1e-6)
    # Aggregated visit counts should put mass on at least one legal action.
    agg = player._ismcts_aggregate_visits(game, add_noise=False)
    assert agg.sum() > 0


def test_ismcts_requires_full_information_policy():
    action_space_size = get_action_space_size(2, "BASE")
    with pytest.raises(ValueError, match="full"):
        nn_mcts_player.NNMCTSPlayer(
            color=Color.BLUE,
            model_type="flat",
            policy_model=None,
            critic_model=None,
            map_type="BASE",
            num_simulations=4,
            ismcts_determinizations=4,
            actor_observation_level="private",
            inference_backend=_UniformBackend(action_space_size),
            device="cpu",
        )


def test_ismcts_search_policy_is_normalized_over_legal_actions():
    game = _make_game()
    player = _make_player()
    policy, action = player.run_search_policy(game, temperature=1.0, add_noise=False)
    assert action in game.playable_actions
    assert policy.sum() == pytest.approx(1.0, abs=1e-5)
