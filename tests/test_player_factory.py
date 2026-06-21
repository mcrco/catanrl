from __future__ import annotations

import pickle

import pytest
from catanatron.game import Game
from catanatron.models.player import Color, Player, RandomPlayer

from catanrl.experiment_store import load_experiment
from catanrl.players.factory import validate_game_context, validate_player_spec
from catanrl.players.lazy_player import LazyConfiguredPlayer
from catanrl.players.player_config import PlayerSpec, load_all_player_specs
from catanrl.utils.catanatron_map import build_catan_map
from tests.player_factory_helpers import write_experiment_stub, write_player_spec_yaml


class _StubInnerPlayer(Player):
    """Stand-in for a cached factory player (no torch)."""

    def decide(self, game, playable_actions):
        return playable_actions[0]


@pytest.fixture
def experiment_root(tmp_path, monkeypatch):
    monkeypatch.setenv("CATANRL_EXPERIMENTS_DIR", str(tmp_path))
    return tmp_path


@pytest.fixture
def policy_experiment(experiment_root):
    return write_experiment_stub(
        experiment_root,
        "foo",
        observation_level="private",
        include_critic=False,
    )


@pytest.fixture
def mcts_experiment(experiment_root):
    return write_experiment_stub(
        experiment_root,
        "mcts-exp",
        observation_level="private",
        include_critic=True,
    )


def _mini_game() -> Game:
    players = [RandomPlayer(Color.BLUE), RandomPlayer(Color.RED)]
    return Game(players, catan_map=build_catan_map("MINI"))


def test_belief_requires_full_observation(policy_experiment):
    spec = PlayerSpec(id="bad-belief", type="belief", experiment="foo")
    with pytest.raises(ValueError, match="full-info"):
        validate_player_spec(spec, load_experiment("foo"))


def test_mcts_requires_critic_checkpoint(policy_experiment):
    spec = PlayerSpec(id="bad-mcts", type="mcts", experiment="foo")
    with pytest.raises(ValueError, match="critic"):
        validate_player_spec(spec, load_experiment("foo"))


def test_validate_game_context_rejects_wrong_player_count(policy_experiment):
    spec = PlayerSpec(id="foo-policy", type="policy", experiment="foo")
    with pytest.raises(ValueError, match="expects 2 players"):
        validate_game_context(spec, num_players=3, map_type="MINI")


def test_lazy_player_pickle_roundtrip(monkeypatch):
    monkeypatch.setattr(
        "catanrl.players.lazy_player.get_cached_player",
        lambda spec_id, color_value: _StubInnerPlayer(Color[color_value], is_bot=True),
    )

    game = _mini_game()
    player = LazyConfiguredPlayer("any-spec", Color.BLUE)
    restored = pickle.loads(pickle.dumps(player))
    action = restored.decide(game, game.playable_actions)
    assert action in game.playable_actions


def test_auto_discover_dedup(experiment_root, monkeypatch):
    write_experiment_stub(experiment_root, "foo", observation_level="private")
    config_dir = experiment_root / "players"
    write_player_spec_yaml(
        config_dir / "foo.yaml",
        {"id": "foo", "type": "policy", "experiment": "foo"},
    )
    monkeypatch.setenv("CATANRL_PLAYER_CONFIG_DIR", str(config_dir))
    monkeypatch.setenv("CATANRL_WEB_AUTO_DISCOVER", "true")

    specs = load_all_player_specs(experiments_root=str(experiment_root))
    foo_specs = [spec for spec in specs if spec.experiment == "foo"]
    assert len(foo_specs) == 1
