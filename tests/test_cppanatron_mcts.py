from __future__ import annotations

import numpy as np
import pytest

from catanrl.envs.cppanatron import (
    NativeGame,
    NativeMCTSSearch,
    find_cppanatron_library,
)


@pytest.fixture(scope="module", autouse=True)
def _require_native_library():
    try:
        find_cppanatron_library()
    except FileNotFoundError as exc:
        pytest.skip(str(exc))


def _run_search(seed: int) -> tuple[np.ndarray, np.ndarray]:
    game = NativeGame(
        2,
        "MINI",
        seed=91,
        map_seed=73,
        number_placement="random",
        vps_to_win=6,
    )
    logits = np.linspace(-1.0, 1.0, game.action_space_size, dtype=np.float32)
    leaf_players: list[int] = []
    try:
        with NativeMCTSSearch(game, "MINI", c_puct=1.5, seed=seed) as search:
            search.initialize_root(logits)
            search.add_root_dirichlet_noise(0.3, 0.25)
            for _ in range(24):
                leaf = search.select_leaf()
                if leaf is None:
                    continue
                observation, player = leaf
                assert observation.dtype == np.float32
                assert observation.shape == (search.observation_size,)
                assert np.isfinite(observation).all()
                assert 0 <= player < game.num_players
                leaf_players.append(player)
                search.evaluate_leaf(logits, value=0.0)
            return search.root_visits(), np.asarray(leaf_players)
    finally:
        game.close()


def test_native_mcts_search_is_deterministic_and_visits_legal_actions():
    first_visits, first_players = _run_search(1234)
    second_visits, second_players = _run_search(1234)

    np.testing.assert_array_equal(first_visits, second_visits)
    np.testing.assert_array_equal(first_players, second_players)
    assert int(first_visits.sum()) == 24

    game = NativeGame(2, "MINI", seed=91, map_seed=73, number_placement="random")
    try:
        legal = game.valid_action_mask()
        assert np.all(first_visits[~legal] == 0)
    finally:
        game.close()


def test_native_mcts_rejects_wrong_policy_shape():
    game = NativeGame(2, "MINI", seed=5, map_seed=7)
    try:
        with NativeMCTSSearch(game, "MINI") as search:
            with pytest.raises(ValueError, match="Expected policy logits"):
                search.initialize_root(np.zeros(game.action_space_size - 1))
    finally:
        game.close()
