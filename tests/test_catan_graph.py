from __future__ import annotations

import copy

import numpy as np
import pytest
import torch
from catanatron.game import Game
from catanatron.gym.board_tensor_features import create_board_tensor
from catanatron.models.enums import RESOURCES
from catanatron.models.map import number_probability
from catanatron.models.player import Color, RandomPlayer

from catanrl.algorithms.alphazero.parallel_self_play import SelfPlayExperience
from catanrl.algorithms.alphazero.trainer import AlphaZeroConfig, AlphaZeroTrainer
from catanrl.experiment_store import backbone_config_from_dict, backbone_config_to_dict
from catanrl.experiments.architecture_config import load_architecture_preset
from catanrl.features.catanatron_utils import (
    compute_observation_feature_vector_dim,
    get_full_numeric_feature_names,
)
from catanrl.models.backbone_builder import build_backbone_config
from catanrl.models.backbones import (
    BackboneConfig,
    CatanGraphBackbone,
    CatanGraphBackboneConfig,
)
from catanrl.models.heads import (
    CatanGraphAuxiliaryValueHead,
    CatanGraphPolicyHead,
    CatanGraphSoftPolicyHead,
    CatanGraphWDLValueHead,
)
from catanrl.models.model_builders import build_policy_value_model
from catanrl.models.models import (
    build_flat_policy_value_network,
    build_hierarchical_policy_value_network,
)
from catanrl.models.wrappers import PolicyValueNetworkWrapper
from catanrl.utils.catanatron_action_space import get_action_array, get_action_space_size
from catanrl.utils.catanatron_map import build_catan_map


def _graph_config(*, hidden_dim: int = 16, normalize_inputs: bool = False) -> BackboneConfig:
    numeric_dim = 74
    return build_backbone_config(
        backbone_type="catan_graph",
        hidden_dims=(hidden_dim,),
        input_dim=compute_observation_feature_vector_dim(2, "BASE", "full"),
        board_height=11,
        board_width=21,
        board_channels=16,
        numeric_dim=numeric_dim,
        num_players=2,
        map_type="BASE",
        graph_hidden_dim=hidden_dim,
        graph_global_hidden_dim=8,
        graph_num_layers=2,
        graph_head_hidden_dim=16,
        graph_normalize_inputs=normalize_inputs,
    )


@pytest.mark.parametrize("normalize_inputs", [False, True])
def test_catan_graph_config_round_trips_through_experiment_metadata(
    normalize_inputs: bool,
) -> None:
    config = _graph_config(normalize_inputs=normalize_inputs)
    restored = backbone_config_from_dict(backbone_config_to_dict(config))

    assert restored == config
    assert isinstance(restored.args, CatanGraphBackboneConfig)
    assert restored.args.board_layout == "width_height"
    assert restored.args.normalize_inputs is normalize_inputs


def test_catan_graph_preserves_existing_observation_and_action_contracts() -> None:
    config = _graph_config()
    model = build_flat_policy_value_network(
        config,
        get_action_space_size(2, "BASE"),
        value_head_type="wdl",
    )
    assert isinstance(model, PolicyValueNetworkWrapper)
    assert isinstance(model.backbone, CatanGraphBackbone)
    assert isinstance(model.policy_head, CatanGraphPolicyHead)
    assert isinstance(model.value_head, CatanGraphWDLValueHead)

    observation_dim = compute_observation_feature_vector_dim(2, "BASE", "full")
    states = torch.randn(3, observation_dim)
    policy, value = model(states)

    assert policy.shape == (3, get_action_space_size(2, "BASE"))
    assert value.shape == (3,)
    assert torch.isfinite(policy).all()
    assert torch.isfinite(value).all()
    assert torch.all(value >= -1.0)
    assert torch.all(value <= 1.0)


def test_catan_graph_policy_mapping_covers_every_flat_action_once() -> None:
    model = build_flat_policy_value_network(
        _graph_config(),
        get_action_space_size(2, "BASE"),
        value_head_type="wdl",
    )
    assert isinstance(model.policy_head, CatanGraphPolicyHead)
    head = model.policy_head
    mapped = torch.cat(
        (
            head.settlement_action_indices,
            head.road_action_indices,
            head.city_action_indices,
            head.robber_action_indices,
            head.other_action_indices,
        )
    )

    assert mapped.numel() == len(get_action_array(2, "BASE"))
    assert torch.equal(mapped.sort().values, torch.arange(mapped.numel()))
    assert head.num_nodes == 54
    assert head.num_edges == 72
    assert head.num_tiles == 19


def test_catan_graph_injects_edge_roads_into_endpoint_node_features() -> None:
    model = build_flat_policy_value_network(
        _graph_config(),
        get_action_space_size(2, "BASE"),
        value_head_type="wdl",
    )
    assert isinstance(model.backbone, CatanGraphBackbone)
    backbone = model.backbone
    state = torch.zeros(1, backbone.config.input_dim)
    board = state[:, backbone.numeric_dim :].reshape(
        1,
        backbone.board_width,
        backbone.board_height,
        backbone.board_channels,
    )
    edge_x, edge_y = backbone.edge_positions[0].tolist()
    current_player_road_channel = int(backbone.road_channel_indices[0])
    board[0, edge_x, edge_y, current_player_road_channel] = 1.0
    captured_node_inputs: list[torch.Tensor] = []

    def capture_node_inputs(_module, inputs) -> None:
        captured_node_inputs.append(inputs[0].detach().clone())

    handle = backbone.node_projection.register_forward_pre_hook(capture_node_inputs)
    try:
        backbone(state)
    finally:
        handle.remove()

    assert len(captured_node_inputs) == 1
    first, second = backbone.edge_pairs[0]
    road_features = captured_node_inputs[0][0, :, current_player_road_channel]
    assert road_features[first] == pytest.approx(1.0 / 3.0)
    assert road_features[second] == pytest.approx(1.0 / 3.0)
    assert torch.count_nonzero(road_features) == 2


def test_catan_graph_normalization_bounds_numeric_and_spatial_inputs() -> None:
    model = build_flat_policy_value_network(
        _graph_config(normalize_inputs=True),
        get_action_space_size(2, "BASE"),
        value_head_type="wdl",
    )
    assert isinstance(model.backbone, CatanGraphBackbone)
    backbone = model.backbone
    state = torch.zeros(1, backbone.config.input_dim)
    turn_index = get_full_numeric_feature_names(2, "BASE").index("TURN_NUMBER")
    state[0, turn_index] = 10_000.0
    board = state[:, backbone.numeric_dim :].reshape(
        1,
        backbone.board_width,
        backbone.board_height,
        backbone.board_channels,
    )
    node_x, node_y = backbone.node_positions[0].tolist()
    board[0, node_x, node_y, 0] = 2.0
    board[0, node_x, node_y, 4] = 5.0 / 36.0
    captured_global: list[torch.Tensor] = []
    captured_nodes: list[torch.Tensor] = []

    global_handle = backbone.global_projection.register_forward_pre_hook(
        lambda _module, inputs: captured_global.append(inputs[0].detach().clone())
    )
    node_handle = backbone.node_projection.register_forward_pre_hook(
        lambda _module, inputs: captured_nodes.append(inputs[0].detach().clone())
    )
    try:
        output = backbone(state)
    finally:
        global_handle.remove()
        node_handle.remove()

    assert output.shape == (1, backbone.output_dim)
    assert captured_global[0][0, turn_index] == 1.0
    assert captured_nodes[0][0, 0, 0] == 1.0
    assert captured_nodes[0][0, 0, 4] == pytest.approx(5.0 / 13.0)


def test_catan_graph_normalization_preserves_checkpoint_tensor_contract() -> None:
    raw = build_flat_policy_value_network(
        _graph_config(normalize_inputs=False),
        get_action_space_size(2, "BASE"),
        value_head_type="wdl",
    )
    normalized = build_flat_policy_value_network(
        _graph_config(normalize_inputs=True),
        get_action_space_size(2, "BASE"),
        value_head_type="wdl",
    )

    assert raw.state_dict().keys() == normalized.state_dict().keys()
    normalized.load_state_dict(raw.state_dict())


def test_catan_graph_rejects_generic_hierarchical_policy_head() -> None:
    with pytest.raises(ValueError, match="require the flat action model"):
        build_hierarchical_policy_value_network(
            _graph_config(),
            num_players=2,
            map_type="BASE",
            value_head_type="wdl",
        )


def test_catan_graph_tile_decoder_is_exact_on_catanatron_board_subspace() -> None:
    model = build_flat_policy_value_network(
        _graph_config(),
        get_action_space_size(2, "BASE"),
        value_head_type="wdl",
    )
    assert isinstance(model.backbone, CatanGraphBackbone)
    backbone = model.backbone
    incidence = backbone.node_to_tile.transpose(0, 1) * 6.0
    tile_values = torch.randn(backbone.num_tiles, 6)
    encoded_nodes = incidence @ tile_values
    recovered_tiles = backbone.tile_decoder @ encoded_nodes

    torch.testing.assert_close(recovered_tiles, tile_values, atol=2e-5, rtol=2e-5)


def test_catan_graph_recovers_real_catanatron_tile_features() -> None:
    model = build_flat_policy_value_network(
        _graph_config(),
        get_action_space_size(2, "BASE"),
        value_head_type="wdl",
    )
    assert isinstance(model.backbone, CatanGraphBackbone)
    backbone = model.backbone
    players = [RandomPlayer(Color.BLUE), RandomPlayer(Color.RED)]
    game = Game(players, catan_map=build_catan_map("BASE"))
    board = torch.from_numpy(create_board_tensor(game, Color.BLUE)).float()
    node_features = board[
        backbone.node_positions[:, 0],
        backbone.node_positions[:, 1],
        4:10,
    ]
    recovered = backbone.tile_decoder @ node_features

    expected = torch.zeros(backbone.num_tiles, 6)
    for tile_index, coordinate in enumerate(backbone.tile_coordinates):
        tile = game.state.board.map.land_tiles[coordinate]
        if tile.resource is not None:
            expected[tile_index, RESOURCES.index(tile.resource)] = number_probability(tile.number)
        if coordinate == game.state.board.robber_coordinate:
            expected[tile_index, 5] = 1.0

    torch.testing.assert_close(recovered, expected, atol=2e-5, rtol=2e-5)


def test_catan_graph_soft_policy_shares_hard_intermediate_features() -> None:
    model = build_flat_policy_value_network(
        _graph_config(),
        get_action_space_size(2, "BASE"),
        value_head_type="wdl",
    )
    assert isinstance(model.policy_head, CatanGraphPolicyHead)
    features = model.backbone(
        torch.randn(2, compute_observation_feature_vector_dim(2, "BASE", "full"))
    )
    _, hidden = model.policy_head.forward_with_hidden(features)
    soft_head = CatanGraphSoftPolicyHead(model.policy_head)
    soft_logits = soft_head(hidden, model.policy_head)

    assert soft_logits.shape == (2, get_action_space_size(2, "BASE"))
    soft_logits.square().mean().backward()
    first_hidden_layer = model.policy_head.settlement_hidden[0]
    assert isinstance(first_hidden_layer, torch.nn.Linear)
    assert first_hidden_layer.weight.grad is not None
    assert model.policy_head.settlement_output.weight.grad is None
    assert sum(p.numel() for p in soft_head.parameters()) < sum(
        p.numel() for p in model.policy_head.parameters()
    )


def test_alphazero_updates_graph_shared_soft_policy() -> None:
    student = build_flat_policy_value_network(
        _graph_config(),
        get_action_space_size(2, "BASE"),
        value_head_type="wdl",
    )
    teacher = copy.deepcopy(student)
    trainer = AlphaZeroTrainer(
        AlphaZeroConfig(
            mode="iterate",
            model_type="flat",
            network_mode="shared",
            num_players=2,
            actor_observation_level="full",
            critic_observation_level="full",
            buffer_size=2,
            batch_size=2,
            policy_lr=1e-3,
            value_loss_weight=1.0,
            soft_policy_temperature=4.0,
            soft_policy_weight=8.0,
            aux_value_horizons=(10, 50, 150),
            aux_value_weight=0.5,
            self_play_backend="cppanatron",
            offload_inactive_models=False,
            device="cpu",
        ),
        student,
        None,
        teacher,
        None,
    )
    assert isinstance(trainer.student_soft_policy_head, CatanGraphSoftPolicyHead)
    assert isinstance(trainer.student_aux_value_head, CatanGraphAuxiliaryValueHead)

    observation_dim = compute_observation_feature_vector_dim(2, "BASE", "full")
    action_dim = get_action_space_size(2, "BASE")
    experiences = []
    for seed, action, value in ((1, 0, 1.0), (2, 1, -1.0)):
        policy = np.zeros(action_dim, dtype=np.float32)
        policy[action] = 1.0
        state = np.random.default_rng(seed).normal(size=observation_dim).astype(np.float32)
        experiences.append(
            SelfPlayExperience(
                actor_state=state,
                critic_state=state.copy(),
                policy=policy,
                action_mask=np.ones(action_dim, dtype=np.bool_),
                value=value,
                full_search=True,
                aux_value_targets=np.asarray([value, value * 0.5, 0.0], dtype=np.float32),
            )
        )

    metrics = trainer.update_weights(experiences)
    assert metrics is not None
    assert metrics["soft_policy_loss"] > 0.0
    assert metrics["aux_value_loss"] > 0.0


def test_nexus_v3_preset_builds_and_reloads_identically() -> None:
    preset = load_architecture_preset(
        "configs/models/catan-graph-nexus-v3-flat-2p-full-shared.yaml"
    )
    model = build_policy_value_model(
        backbone_type=preset.backbone_type,
        model_type=preset.model_type,
        hidden_dims=preset.policy_hidden_dims,
        num_players=2,
        map_type="BASE",
        actor_observation_level="full",
        critic_observation_level="full",
        device="cpu",
        value_head_type=preset.value_head_type,
        graph_hidden_dim=preset.graph_hidden_dim,
        graph_global_hidden_dim=preset.graph_global_hidden_dim,
        graph_num_layers=preset.graph_num_layers,
        graph_head_hidden_dim=preset.graph_head_hidden_dim,
        graph_normalize_inputs=preset.graph_normalize_inputs,
    )
    rebuilt = build_flat_policy_value_network(
        backbone_config_from_dict(backbone_config_to_dict(model.backbone_config)),
        get_action_space_size(2, "BASE"),
        value_head_type="wdl",
    )
    rebuilt.load_state_dict(model.state_dict())
    state = torch.from_numpy(
        np.random.default_rng(3)
        .normal(size=(1, compute_observation_feature_vector_dim(2, "BASE", "full")))
        .astype(np.float32)
    )

    model.eval()
    rebuilt.eval()
    with torch.inference_mode():
        expected = model(state)
        actual = rebuilt(state)
    torch.testing.assert_close(actual[0], expected[0])
    torch.testing.assert_close(actual[1], expected[1])


def test_normalized_nexus_v3_preset_is_opt_in_and_preserves_game_contract() -> None:
    baseline = load_architecture_preset(
        "configs/models/catan-graph-nexus-v3-flat-2p-full-shared.yaml"
    )
    normalized = load_architecture_preset(
        "configs/models/catan-graph-nexus-v3-normalized-flat-2p-full-shared.yaml"
    )

    assert baseline.graph_normalize_inputs is False
    assert normalized.graph_normalize_inputs is True
    assert normalized.policy_mode == baseline.policy_mode == "full"
    assert normalized.critic_mode == baseline.critic_mode == "full"
    assert normalized.map_type == baseline.map_type == "BASE"
    assert normalized.vps_to_win == baseline.vps_to_win == 15
    assert normalized.discard_limit == baseline.discard_limit == 9
