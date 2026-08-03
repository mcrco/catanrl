from __future__ import annotations

import torch

from catanrl.algorithms.common import PolicyAgent
from catanrl.algorithms.imitation_learning.dagger import _train_on_dataset
from catanrl.algorithms.imitation_learning.dataset import AggregatedDataset
from catanrl.envs.puffer.common import compute_single_agent_dims
from catanrl.experiment_store import (
    KIND_POLICY_VALUE,
    GameConfig,
    NetworkSpec,
    backbone_config_from_dict,
    backbone_config_to_dict,
    build_network,
    network_spec_from_model,
)
from catanrl.experiments.architecture_config import load_architecture_preset
from catanrl.models.backbone_builder import build_backbone_config
from catanrl.models.backbones import (
    CompactCrossDimensionalBackbone,
    CrossDimensionalBackboneConfig,
    _reshape_board_tensor,
    create_backbone,
)
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
        value_head_type="wdl",
    )
    assert isinstance(model.backbone, CompactCrossDimensionalBackbone)
    input_dim = model.backbone.numeric_dim + (
        model.backbone.board_height * model.backbone.board_width * model.backbone.board_channels
    )

    policy, value = model(torch.zeros(3, input_dim))

    assert policy.shape == (3, get_action_space_size(2, "BASE"))
    assert value.shape == (3,)
    assert torch.all(value >= -1.0)
    assert torch.all(value <= 1.0)
    assert sum(parameter.numel() for parameter in model.parameters()) < 2_000_000

    spec = network_spec_from_model(
        model,
        kind=KIND_POLICY_VALUE,
        model_type="flat",
        observation_level="full",
    )
    restored_spec = NetworkSpec.from_dict(spec.to_dict())
    restored_model = build_network(
        restored_spec,
        GameConfig(num_players=2, map_type="BASE", vps_to_win=15, discard_limit=9),
    )
    assert spec.value_head_type == restored_spec.value_head_type == "wdl"
    restored_model.load_state_dict(model.state_dict())


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


def test_fresh_xdim_config_preserves_catanatron_board_layout() -> None:
    config = build_backbone_config(
        backbone_type="xdim_compact",
        hidden_dims=(32,),
        board_height=11,
        board_width=21,
        board_channels=2,
        numeric_dim=4,
        xdim_cnn_channels=(8,),
    )
    assert isinstance(config.args, CrossDimensionalBackboneConfig)
    assert config.args.board_layout == "width_height"

    board = torch.arange(21 * 11 * 2).reshape(21, 11, 2)
    restored = _reshape_board_tensor(board.flatten().unsqueeze(0), config.args)

    assert restored.shape == (1, 2, 21, 11)
    assert torch.equal(restored[0], board.permute(2, 0, 1))


def test_old_xdim_metadata_keeps_legacy_board_layout() -> None:
    old_metadata = {
        "architecture": "compact_cross_dimensional",
        "args": {
            "board_height": 11,
            "board_width": 21,
            "board_channels": 2,
            "numeric_dim": 4,
            "cnn_channels": [8],
            "cnn_kernel_size": [3, 5],
            "numeric_hidden_dims": [32],
            "fusion_hidden_dim": 32,
            "output_dim": 32,
        },
    }

    config = backbone_config_from_dict(old_metadata)
    assert isinstance(config.args, CrossDimensionalBackboneConfig)
    assert config.args.board_layout == "legacy_height_width"

    board_flat = torch.arange(21 * 11 * 2).unsqueeze(0)
    restored = _reshape_board_tensor(board_flat, config.args)

    assert restored.shape == (1, 2, 11, 21)


def test_compact_xdim_training_preset_loads() -> None:
    preset = load_architecture_preset("configs/models/xdim-compact-flat-2p-full-shared.yaml")

    assert preset.backbone_type == "xdim_compact"
    assert preset.network_mode == "shared"
    assert preset.value_head_type == "wdl"
    assert preset.actor_observation_level == preset.critic_observation_level == "full"


def test_compact_wdl_head_runs_through_shared_dagger_update() -> None:
    num_players = 2
    map_type = "MINI"
    dims = compute_single_agent_dims(
        num_players,
        map_type,
        actor_observation_level="full",
        critic_observation_level="full",
    )
    model = build_policy_value_model(
        backbone_type="xdim_compact",
        model_type="flat",
        hidden_dims=(32,),
        num_players=num_players,
        map_type=map_type,
        actor_observation_level="full",
        critic_observation_level="full",
        device="cpu",
        xdim_cnn_channels=(8, 8),
        xdim_cnn_kernel_size=(3, 3),
        xdim_fusion_hidden_dim=32,
        value_head_type="wdl",
    )
    dataset = AggregatedDataset(
        full_state_dim=dims["critic_dim"],
        num_players=num_players,
        map_type=map_type,
        actor_observation_level="full",
        critic_observation_level="full",
        max_size=4,
    )
    num_actions = get_action_space_size(num_players, map_type)
    dataset.add_samples(
        full_states=torch.randn(4, dims["critic_dim"]).numpy(),
        expert_actions=[0, 1, 2, 3],
        returns=[1.0, -1.0, 0.5, -0.5],
        is_single_action=[False] * 4,
        action_masks=torch.ones(4, num_actions, dtype=torch.bool).numpy(),
    )

    metrics = _train_on_dataset(
        dataset=dataset,
        policy_agent=PolicyAgent(model, "flat", torch.device("cpu")),
        critic_model=model,
        policy_optimizer=torch.optim.Adam(model.parameters(), lr=1e-3),
        critic_optimizer=None,
        epochs=1,
        batch_size=4,
        device=torch.device("cpu"),
        num_players=num_players,
        map_type=map_type,
        model_type="flat",
        uses_shared_network=True,
    )

    assert metrics["value_loss"] > 0.0
