from argparse import Namespace

import pytest

from catanrl.eval.dagger_eval import FrozenImitationEvalSet
from catanrl.experiment_store import ExperimentMetadata, GameConfig
from catanrl.experiments.common_args import (
    apply_number_placement,
    resolve_number_placement,
)


def test_new_game_config_defaults_to_official_spiral():
    assert GameConfig(num_players=2, map_type="BASE").number_placement == "official_spiral"


def test_missing_number_placement_loads_as_random():
    meta = ExperimentMetadata.from_dict(
        {
            "name": "legacy-run",
            "algorithm": "sarl_ppo",
            "game": {"num_players": 2, "map_type": "BASE"},
            "networks": {},
        }
    )
    assert meta.game.number_placement == "random"


def test_saved_number_placement_is_preserved_on_load():
    meta = ExperimentMetadata.from_dict(
        {
            "name": "spiral-run",
            "algorithm": "marl_cc",
            "game": {
                "num_players": 2,
                "map_type": "BASE",
                "number_placement": "official_spiral",
            },
            "networks": {},
        }
    )
    assert meta.game.number_placement == "official_spiral"


def test_new_run_defaults_to_official_spiral():
    assert resolve_number_placement(Namespace()) == "official_spiral"


def test_explicit_cli_wins_for_new_run():
    args = Namespace(number_placement="random")
    assert resolve_number_placement(args) == "random"


def test_resume_inherits_saved_placement():
    args = Namespace()
    assert (
        resolve_number_placement(args, resume_active=True, saved_number_placement="random")
        == "random"
    )


def test_resume_rejects_conflicting_cli():
    args = Namespace(number_placement="official_spiral")
    with pytest.raises(ValueError, match="cannot change number placement"):
        resolve_number_placement(
            args,
            resume_active=True,
            saved_number_placement="random",
        )


def test_eval_inherits_experiment_unless_overridden():
    args = Namespace()
    assert resolve_number_placement(args, experiment_number_placement="random") == "random"
    args = Namespace(number_placement="official_spiral")
    assert (
        resolve_number_placement(args, experiment_number_placement="random") == "official_spiral"
    )


def test_apply_number_placement_stores_resolved_value():
    args = Namespace()
    assert apply_number_placement(args) == "official_spiral"
    assert args.number_placement == "official_spiral"


def test_frozen_imitation_eval_forwards_number_placement(monkeypatch):
    captured: dict[str, str] = {}

    class DummyGame:
        def __init__(self, **kwargs):
            captured["catan_map"] = kwargs.get("catan_map")

        def play(self):
            return None

    def fake_build_catan_map(map_type, *, seed=None, number_placement="official_spiral"):
        captured["number_placement"] = number_placement
        return object()

    monkeypatch.setattr("catanrl.eval.dagger_eval.Game", DummyGame)
    monkeypatch.setattr("catanrl.eval.dagger_eval.build_catan_map", fake_build_catan_map)

    FrozenImitationEvalSet.generate(
        map_type="BASE",
        num_players=2,
        vps_to_win=15,
        discard_limit=9,
        num_games=1,
        max_decision_points=1,
        seed=1,
        number_placement="random",
    )
    assert captured["number_placement"] == "random"
