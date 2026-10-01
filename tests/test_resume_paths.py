import argparse
import os
import shutil

import torch

from catanrl.experiment_store import (
    KIND_POLICY,
    GameConfig,
    ResumeContext,
    TrainingWarmStart,
    load_experiment,
    network_spec_from_model,
    prepare_resume,
    run_paths,
    save_experiment,
)
from catanrl.models.backbone_builder import build_backbone_config
from catanrl.models.models import build_flat_policy_network


def _save_tiny_experiment(name: str) -> None:
    policy = build_flat_policy_network(
        build_backbone_config(backbone_type="mlp", hidden_dims=(8,), input_dim=4), num_actions=3
    )
    ckpt_dir = os.path.join(os.environ["CATANRL_EXPERIMENTS_DIR"], name, "checkpoints")
    os.makedirs(ckpt_dir)
    torch.save(policy.state_dict(), os.path.join(ckpt_dir, "policy_best.pt"))
    save_experiment(
        name,
        ckpt_dir,
        algorithm="marl_cc",
        game=GameConfig(num_players=2, map_type="BASE"),
        networks={"policy": network_spec_from_model(policy, kind=KIND_POLICY, model_type="flat")},
    )


def test_resume_writes_back_to_the_loaded_directory(tmp_path, monkeypatch):
    monkeypatch.setenv("CATANRL_EXPERIMENTS_DIR", str(tmp_path))
    _save_tiny_experiment("original")
    shutil.copytree(tmp_path / "original", tmp_path / "copy")

    exp = load_experiment("copy")
    assert exp.metadata.name == "original"
    resume = prepare_resume(
        argparse.Namespace(resume=True),
        TrainingWarmStart(experiment=exp, checkpoints=None),
    )
    exp_path, ckpt_dir, state_path = run_paths(resume.experiment_name, resume)

    copy_dir = str(tmp_path / "copy")
    assert exp_path == copy_dir
    assert ckpt_dir == os.path.join(copy_dir, "checkpoints")
    assert state_path == os.path.join(copy_dir, "training_state.pt")


def test_new_runs_use_the_experiments_root(tmp_path, monkeypatch):
    monkeypatch.setenv("CATANRL_EXPERIMENTS_DIR", str(tmp_path))
    exp_path, ckpt_dir, state_path = run_paths("fresh-run", ResumeContext(active=False))
    assert exp_path == str(tmp_path / "fresh-run")
    assert ckpt_dir == str(tmp_path / "fresh-run" / "checkpoints")
    assert state_path == str(tmp_path / "fresh-run" / "training_state.pt")
