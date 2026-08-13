from __future__ import annotations

from dataclasses import replace

import pytest

from catanrl.experiment_store import (
    KIND_POLICY_VALUE,
    CheckpointRegistry,
    Experiment,
    ExperimentMetadata,
    GameConfig,
    NetworkSpec,
)
from catanrl.experiments.architecture_config import load_architecture_preset
from catanrl.experiments.canopy_contract import (
    CanopyContractError,
    canopy_checkpoint_global_iteration,
    validate_canopy_architecture,
    validate_canopy_experiment,
)
from catanrl.models.backbone_builder import build_backbone_config


def _experiment(tmp_path, *, board_layout: str = "width_height", reward: str = "win"):
    backbone = build_backbone_config(
        backbone_type="xdim_compact",
        input_dim=16,
        hidden_dims=(32,),
        board_height=11,
        board_width=21,
        board_channels=1,
        numeric_dim=0,
        xdim_cnn_channels=(8,),
        xdim_cnn_kernel_size=(3, 5),
        xdim_fusion_hidden_dim=16,
        xdim_board_layout=board_layout,
    )
    metadata = ExperimentMetadata(
        name="contract-test",
        algorithm="dagger",
        game=GameConfig(num_players=2, map_type="BASE", vps_to_win=15, discard_limit=9),
        networks={
            "policy": NetworkSpec(
                kind=KIND_POLICY_VALUE,
                backbone=backbone,
                model_type="flat",
                observation_level="full",
                num_actions=4,
                value_head_type="wdl",
            )
        },
        train_config={"env_backend": "cppanatron", "reward_function": reward, "gamma": 1.0},
    )
    return Experiment(str(tmp_path), metadata, CheckpointRegistry())


def test_parity_model_config_satisfies_contract():
    arch = load_architecture_preset("configs/models/xdim-compact-medium-flat-2p-full-shared.yaml")

    validate_canopy_architecture(arch)


def test_architecture_contract_rejects_changed_game_rules():
    arch = load_architecture_preset("configs/models/xdim-compact-medium-flat-2p-full-shared.yaml")

    with pytest.raises(CanopyContractError, match="game.vps_to_win"):
        validate_canopy_architecture(replace(arch, vps_to_win=10))


def test_experiment_contract_accepts_terminal_win_dagger(tmp_path):
    validate_canopy_experiment(
        _experiment(tmp_path),
        require_terminal_dagger=True,
    )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"board_layout": "legacy_height_width"}, "board_layout"),
        ({"reward": "shaped"}, "reward_function"),
    ],
)
def test_experiment_contract_rejects_legacy_layout_or_shaped_reward(
    tmp_path,
    kwargs,
    message,
):
    with pytest.raises(CanopyContractError, match=message):
        validate_canopy_experiment(
            _experiment(tmp_path, **kwargs),
            require_terminal_dagger=True,
        )


def test_checkpoint_global_iteration_composes_branch_offset(tmp_path):
    experiment = _experiment(tmp_path)
    experiment.metadata.train_config["self_play_iteration_offset"] = 9

    assert canopy_checkpoint_global_iteration(experiment, 10) == 19


@pytest.mark.parametrize("offset", [-1, True, 1.5])
def test_checkpoint_global_iteration_rejects_invalid_branch_offset(tmp_path, offset):
    experiment = _experiment(tmp_path)
    experiment.metadata.train_config["self_play_iteration_offset"] = offset

    with pytest.raises(CanopyContractError, match="self_play_iteration_offset"):
        canopy_checkpoint_global_iteration(experiment, 2)
