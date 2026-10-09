import json

import pytest

pytest.importorskip("flask", reason="needs the web extra (uv sync --extra web)")

import torch  # noqa: E402
from catanatron.models.player import Color  # noqa: E402
from catanatron.registry import REGISTRY  # noqa: E402
from catanatron.web.models import StoredGame, db, get_game_state  # noqa: E402

from catanrl import web  # noqa: E402
from catanrl.experiment_store import (  # noqa: E402
    KIND_POLICY,
    GameConfig,
    network_spec_from_model,
    save_experiment,
)
from catanrl.experiments.architecture_config import load_architecture_preset  # noqa: E402
from catanrl.models import build_policy_model  # noqa: E402

GRAPH_S_PRESET = "configs/models/catan-graph-s-flat-2p-full-sep.yaml"


def _save_graph_experiment(root, name):
    arch = load_architecture_preset(GRAPH_S_PRESET)
    policy = build_policy_model(
        model_type="flat",
        hidden_dims=arch.policy_hidden_dims,
        actor_observation_level="full",
        backbone_type=arch.backbone_type,
        num_players=arch.num_players,
        map_type=arch.map_type,
        device="cpu",
        graph_hidden_dim=arch.graph_hidden_dim,
        graph_global_hidden_dim=arch.graph_global_hidden_dim,
        graph_num_layers=arch.graph_num_layers,
        graph_head_hidden_dim=arch.graph_head_hidden_dim,
        graph_normalize_inputs=arch.graph_normalize_inputs,
        graph_semantic_inputs=arch.graph_semantic_inputs,
    )
    ckpt_dir = root / name / "checkpoints"
    ckpt_dir.mkdir(parents=True)
    torch.save(policy.state_dict(), ckpt_dir / "policy_best.pt")
    save_experiment(
        name,
        str(ckpt_dir),
        algorithm="marl_cc",
        game=GameConfig(num_players=2, map_type="BASE", vps_to_win=15, discard_limit=9),
        networks={
            "policy": network_spec_from_model(
                policy, kind=KIND_POLICY, model_type="flat", observation_level="full"
            )
        },
    )


@pytest.fixture
def experiments_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("CATANRL_EXPERIMENTS_DIR", str(tmp_path))
    _save_graph_experiment(tmp_path, "graph-exp")
    (tmp_path / "unfinished" / "checkpoints").mkdir(parents=True)  # no metadata.json
    web.clear_caches()
    yield tmp_path
    for key in [k for k in REGISTRY if k.startswith(web.KEY_PREFIX)]:
        REGISTRY.pop(key)
    web.clear_caches()


@pytest.fixture
def client(experiments_dir):
    app = web.create_app({"TESTING": True, "SQLALCHEMY_DATABASE_URI": "sqlite:///:memory:"})
    return app.test_client()


def test_full_info_experiment_registers_belief_and_oracle_players(client):
    players = {p["key"]: p for p in json.loads(client.get("/api/players").data)}
    catanrl_keys = sorted(k for k in players if k.startswith(web.KEY_PREFIX))
    assert catanrl_keys == ["CATANRL_GRAPH-EXP", "CATANRL_ORACLE_GRAPH-EXP"]
    assert players["CATANRL_GRAPH-EXP"]["name"] == "graph-exp (belief-averaged)"
    assert players["CATANRL_GRAPH-EXP"]["is_bot"] is True
    assert players["CATANRL_GRAPH-EXP"]["params"] == []  # nothing a browser can re-point
    assert {"HUMAN", "CATANATRON"} <= set(players)

    seat = {"key": "CATANRL_GRAPH-EXP", "params": {"which": "latest"}}
    assert client.post("/api/games", json={"players": [seat, "HUMAN"]}).status_code == 400


def test_players_are_stored_by_key_and_share_the_cached_model(client):
    response = client.post("/api/games", json={"players": ["CATANRL_GRAPH-EXP", "HUMAN"]})
    assert response.status_code == 200
    game_id = json.loads(response.data)["game_id"]

    with client.application.app_context():
        row = db.session.get(StoredGame, game_id)
        assert {"key": "CATANRL_GRAPH-EXP", "params": {}} in json.loads(row.player_specs)
        game = get_game_state(game_id)
    # Seating order is shuffled, so find the bot by type; its color is fixed.
    bot = next(p for p in game.state.players if isinstance(p, web.ExperimentBeliefPlayer))
    assert bot.color == Color.RED
    assert bot.EXPERIMENT == "graph-exp"
    assert bot.model is web.cached_policy("graph-exp", "best", "cpu")


def test_bot_plays_its_turns_through_the_api(client):
    response = client.post(
        "/api/games", json={"players": ["CATANRL_ORACLE_GRAPH-EXP", "CATANRL_GRAPH-EXP"]}
    )
    game_id = json.loads(response.data)["game_id"]
    before = json.loads(client.get(f"/api/games/{game_id}/states/latest").data)
    for _ in range(6):
        after = json.loads(client.post(f"/api/games/{game_id}/actions", json={}).data)
    assert len(after["action_records"]) == len(before["action_records"]) + 6


def test_bot_refuses_games_it_was_not_trained_for(client):
    response = client.post("/api/games", json={"players": ["CATANRL_GRAPH-EXP", "R", "R"]})
    game_id = json.loads(response.data)["game_id"]
    with pytest.raises(ValueError, match="trained for 2 players"):
        for _ in range(6):  # initial placement: every seat acts within 6 ticks
            client.post(f"/api/games/{game_id}/actions", json={})

    response = client.post(
        "/api/games", json={"players": ["CATANRL_GRAPH-EXP", "R"], "map_template": "MINI"}
    )
    game_id = json.loads(response.data)["game_id"]
    with pytest.raises(ValueError, match="BASE map"):
        for _ in range(4):
            client.post(f"/api/games/{game_id}/actions", json={})


def test_explicit_experiment_without_checkpoint_fails_loudly(experiments_dir):
    (experiments_dir / "graph-exp" / "checkpoints" / "policy_best.pt").unlink()
    web.clear_caches()
    with pytest.raises(FileNotFoundError):
        web.register_experiment_players(["graph-exp"])
    assert web.register_experiment_players() == []  # discovery just skips it
