"""Tests for the unified NN MCTS engine: determinism, to_play perspective,
training hooks (policy targets + Dirichlet noise), and the shared across-games
self-play runner."""

import random

import numpy as np
import torch
from catanatron.game import Game
from catanatron.models.player import RandomPlayer

from catanrl.algorithms.alphazero.parallel_self_play import generate_self_play_data
from catanrl.features.catanatron_utils import COLOR_ORDER
from catanrl.players import NNMCTSPlayer
from catanrl.players.nn_mcts_player import _Node
from catanrl.utils.catanatron_action_space import to_action_space
from catanrl.utils.catanatron_game import force_player_order
from catanrl.utils.catanatron_map import build_catan_map
from nn_mcts_helpers import build_fake_torch_models, build_mock_player

MAP_TYPE = "MINI"
NUM_PLAYERS = 2


def _build_game(player: NNMCTSPlayer, seed: int = 0) -> Game:
    players = [player] + [
        RandomPlayer(color) for color in COLOR_ORDER[1:NUM_PLAYERS]
    ]
    game = Game(
        players=players,
        catan_map=build_catan_map(MAP_TYPE, seed=seed, number_placement="random"),
        seed=seed,
    )
    force_player_order(game, players)
    return game


def test_node_records_to_play_perspective():
    player = build_mock_player()
    game = _build_game(player)
    root = _Node(game=game.copy(), parent=None)
    assert root.to_play == game.state.current_color()
    assert root.to_play == player.color


def test_run_search_policy_returns_valid_distribution():
    player = build_mock_player(num_simulations=16)
    game = _build_game(player)
    num_players = len(game.state.colors)
    game_colors = tuple(game.state.colors)

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    policy, action = player.run_search_policy(game, temperature=1.0, add_noise=False)

    assert abs(float(policy.sum()) - 1.0) < 1e-5
    assert action in game.playable_actions

    playable_indices = {
        to_action_space(a, num_players, MAP_TYPE, game_colors) for a in game.playable_actions
    }
    nonzero = {idx for idx in range(len(policy)) if policy[idx] > 0}
    assert nonzero <= playable_indices


def test_run_search_policy_is_deterministic_under_seed():
    player = build_mock_player(num_simulations=16)
    game = _build_game(player)

    def run_once() -> np.ndarray:
        random.seed(123)
        np.random.seed(123)
        torch.manual_seed(123)
        policy, _ = player.run_search_policy(game, temperature=1.0, add_noise=True)
        return policy

    np.testing.assert_array_equal(run_once(), run_once())


def test_zero_temperature_policy_is_argmax_onehot():
    player = build_mock_player(num_simulations=16)
    game = _build_game(player)

    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)
    policy, action = player.run_search_policy(game, temperature=0.0, add_noise=False)

    assert np.count_nonzero(policy) == 1
    assert float(policy.max()) == 1.0
    assert action in game.playable_actions


def test_dirichlet_noise_perturbs_root_priors_only():
    player = build_mock_player(num_simulations=4)
    game = _build_game(player)

    root = _Node(game=game.copy(), parent=None)
    player._expand(root, player._inference_backend)
    assert root.action_priors

    before = dict(root.action_priors)
    random.seed(7)
    np.random.seed(7)
    player._add_root_dirichlet_noise(root)
    after = root.action_priors

    assert set(before) == set(after)
    assert abs(sum(after.values()) - 1.0) < 1e-5
    assert any(abs(after[a] - before[a]) > 1e-9 for a in before)


def test_generate_self_play_data_smoke():
    policy_model, critic_model = build_fake_torch_models(NUM_PLAYERS, MAP_TYPE)

    experiences, stats = generate_self_play_data(
        policy_model=policy_model,
        critic_model=critic_model,
        model_type="flat",
        map_type=MAP_TYPE,
        num_players=NUM_PLAYERS,
        num_games=1,
        num_game_workers=1,
        num_simulations=2,
        c_puct=1.5,
        prunning=False,
        actor_observation_level="private",
        critic_observation_level="full",
        ismcts_determinizations=1,
        inference_batch_size=8,
        inference_wait_ms=2.0,
        temperature=1.0,
        final_temperature=0.1,
        temperature_drop_move=30,
        noise_turns=20,
        dirichlet_alpha=0.3,
        dirichlet_frac=0.25,
        vps_to_win=3,
        discard_limit=7,
        seed=0,
        device="cpu",
        show_tqdm=False,
    )

    assert int(stats.get("games", 0)) == 1
    assert len(experiences) >= 1
    sample = experiences[0]
    assert abs(float(sample.policy.sum()) - 1.0) < 1e-4
    assert sample.value in (-1.0, 0.0, 1.0)
    assert sample.actor_state.ndim == 1
    assert sample.critic_state.ndim == 1
