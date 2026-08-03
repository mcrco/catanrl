from __future__ import annotations

import torch

from catanrl.experiment_store import backbone_config_from_dict, backbone_config_to_dict
from catanrl.experiments.architecture_config import load_architecture_preset
from catanrl.models.backbone_builder import build_backbone_config
from catanrl.models.backbones import CompactCrossDimensionalBackbone, create_backbone
from catanrl.models.model_builders import build_policy_value_model
from catanrl.utils.catanatron_action_space import get_action_space_size


def test_compact_xdim_preserves_policy_value_shapes_with_small_parameter_count() -> None:
    model = build_policy_value_model(
        backbone_type="xdim_compact",
        model_type="flat",
        hidden_dims=(256, 256),
        num_players=2,
        map_type="BASE",
        actor_observation_level="full",
        critic_observation_level="full",
        device="cpu",
        xdim_cnn_channels=(64, 128, 128),
        xdim_cnn_kernel_size=(3, 5),
        xdim_fusion_hidden_dim=256,
    )
    assert isinstance(model.backbone, CompactCrossDimensionalBackbone)
    input_dim = model.backbone.numeric_dim + (
        model.backbone.board_height * model.backbone.board_width * model.backbone.board_channels
    )

    policy, value = model(torch.zeros(3, input_dim))

    assert policy.shape == (3, get_action_space_size(2, "BASE"))
    assert value.shape == (3,)
    assert sum(parameter.numel() for parameter in model.parameters()) < 2_000_000


def test_compact_xdim_config_round_trips_through_experiment_schema() -> None:
    config = build_backbone_config(
        backbone_type="xdim_compact",
        hidden_dims=(256, 256),
        board_height=11,
        board_width=21,
        board_channels=16,
        numeric_dim=74,
        xdim_cnn_channels=(64, 128, 128),
        xdim_cnn_kernel_size=(3, 5),
        xdim_fusion_hidden_dim=256,
    )

    restored = backbone_config_from_dict(backbone_config_to_dict(config))
    backbone, output_dim = create_backbone(restored)

    assert restored == config
    assert output_dim == 256
    assert backbone(torch.zeros(2, 3770)).shape == (2, 256)


def test_compact_xdim_training_preset_loads() -> None:
    preset = load_architecture_preset("configs/models/xdim-compact-flat-2p-full-shared.yaml")

    assert preset.backbone_type == "xdim_compact"
    assert preset.network_mode == "shared"
    assert preset.actor_observation_level == preset.critic_observation_level == "full"
