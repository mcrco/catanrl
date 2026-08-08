from dataclasses import dataclass, field
from typing import List, Literal, Tuple, Union

import torch
import torch.nn as nn

from .utils import orthogonal_init


@dataclass
class MLPBackboneConfig:
    input_dim: int
    hidden_dims: List[int]


@dataclass
class CrossDimensionalBackboneConfig:
    """Based on this paper: https://arxiv.org/pdf/2008.07079."""

    board_height: int = 11
    board_width: int = 21
    board_channels: int = 20
    numeric_dim: int = 56
    cnn_channels: List[int] = field(default_factory=lambda: [32, 64, 64])
    cnn_kernel_size: Tuple[int, int] = (3, 5)  # height x width (paper uses 5x3)
    numeric_hidden_dims: List[int] = field(default_factory=lambda: [64, 64])
    fusion_hidden_dim: int = 256
    output_dim: int = 256
    # Catanatron flattens board tensors as (WIDTH, HEIGHT, CHANNELS).
    # Legacy checkpoints reconstructed them as (HEIGHT, WIDTH, CHANNELS), so
    # the layout is explicit to preserve those checkpoints while fixing new
    # training runs.
    board_layout: str = "legacy_height_width"


@dataclass
class CatanGraphBackboneConfig:
    """Canopy-style heterogeneous graph encoder over a Catanatron vector.

    The input contract remains ``[numeric, flattened board tensor]``.  Static
    indices gather the active Catan nodes and tiles from that tensor; no
    alternate feature encoder or game representation is introduced.
    """

    input_dim: int
    numeric_dim: int
    board_height: int
    board_width: int
    board_channels: int
    num_players: int
    map_type: Literal["BASE", "MINI", "TOURNAMENT"]
    hidden_dim: int = 256
    global_hidden_dim: int = 96
    num_layers: int = 4
    head_hidden_dim: int = 256
    board_layout: str = "width_height"


@dataclass
class BackboneConfig:
    architecture: str
    args: Union[MLPBackboneConfig, CrossDimensionalBackboneConfig, CatanGraphBackboneConfig]


def _board_spatial_shape(config: CrossDimensionalBackboneConfig) -> tuple[int, int]:
    if config.board_layout == "legacy_height_width":
        return config.board_height, config.board_width
    if config.board_layout == "width_height":
        return config.board_width, config.board_height
    raise ValueError(
        f"board_layout must be 'legacy_height_width' or 'width_height', got {config.board_layout!r}"
    )


def _board_kernel_size(config: CrossDimensionalBackboneConfig) -> tuple[int, int]:
    """Translate semantic (height, width) kernels to the stored tensor axes."""
    kernel_height, kernel_width = config.cnn_kernel_size
    if config.board_layout == "legacy_height_width":
        return kernel_height, kernel_width
    if config.board_layout == "width_height":
        return kernel_width, kernel_height
    # Keep layout validation and its error message in one place.
    _board_spatial_shape(config)
    raise AssertionError("unreachable")


def _reshape_board_tensor(
    board_flat: torch.Tensor,
    config: CrossDimensionalBackboneConfig,
) -> torch.Tensor:
    first_spatial, second_spatial = _board_spatial_shape(config)
    board_tensor = board_flat.reshape(
        -1,
        first_spatial,
        second_spatial,
        config.board_channels,
    )
    return board_tensor.permute(0, 3, 1, 2)


class MLPBackbone(nn.Module):
    def __init__(self, config: MLPBackboneConfig):
        super().__init__()
        layers = []
        in_dim = config.input_dim
        for hidden_dim in config.hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        self.layers = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self) -> None:
        # For PPO
        for module in self.modules():
            orthogonal_init(module)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class CrossDimensionalBackbone(nn.Module):
    """
    Cross-dimensional backbone that processes both spatial (2D board) and
    non-spatial (1D numeric) features, then fuses them into a unified representation.

    Follows this paper: https://arxiv.org/pdf/2008.07079.
    """

    def __init__(self, config: CrossDimensionalBackboneConfig):
        super().__init__()
        self.config = config

        # Store dimensions for input splitting
        self.numeric_dim = config.numeric_dim
        self.board_height = config.board_height
        self.board_width = config.board_width
        self.board_channels = config.board_channels
        self.board_layout = config.board_layout
        self.board_flat_dim = config.board_height * config.board_width * config.board_channels

        # CNN branch for board tensor
        cnn_layers = []
        in_channels = config.board_channels
        current_h, current_w = _board_spatial_shape(config)
        kernel_size = _board_kernel_size(config)

        for out_channels in config.cnn_channels:
            # Use padding to maintain spatial dimensions initially
            pad_h = (kernel_size[0] - 1) // 2
            pad_w = (kernel_size[1] - 1) // 2

            cnn_layers.append(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    padding=(pad_h, pad_w),
                )
            )
            cnn_layers.append(nn.BatchNorm2d(out_channels))
            cnn_layers.append(nn.ReLU())

            in_channels = out_channels

        self.cnn = nn.Sequential(*cnn_layers)

        cnn_output_size = config.cnn_channels[-1] * current_h * current_w
        self.cnn_projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(cnn_output_size, config.fusion_hidden_dim),
            nn.LayerNorm(config.fusion_hidden_dim),
            nn.ReLU(),
        )

        # MLP branch for numeric features
        numeric_layers = []
        in_dim = config.numeric_dim
        for hidden_dim in config.numeric_hidden_dims:
            numeric_layers.append(nn.Linear(in_dim, hidden_dim))
            numeric_layers.append(nn.LayerNorm(hidden_dim))
            numeric_layers.append(nn.ReLU())
            in_dim = hidden_dim

        self.numeric_mlp = nn.Sequential(*numeric_layers)
        numeric_output_dim = config.numeric_hidden_dims[-1]

        # Fusion
        fusion_input_dim = config.fusion_hidden_dim + numeric_output_dim
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, config.fusion_hidden_dim),
            nn.LayerNorm(config.fusion_hidden_dim),
            nn.ReLU(),
            nn.Linear(config.fusion_hidden_dim, config.output_dim),
            nn.LayerNorm(config.output_dim),
            nn.ReLU(),
        )

        self.output_dim = config.output_dim
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights using orthogonal initialization (PPO style)."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                orthogonal_init(module)
            elif isinstance(module, nn.Conv2d):
                # Use orthogonal init for conv layers too
                nn.init.orthogonal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Split flattened input into numeric and board components
        numeric_features = x[:, : self.numeric_dim]
        board_flat = x[:, self.numeric_dim :]

        # Reshape board to (batch, H, W, C) then permute to (batch, C, H, W) for Conv2d
        board_tensor = _reshape_board_tensor(board_flat, self.config)

        # CNN branch
        cnn_out = self.cnn(board_tensor)
        cnn_features = self.cnn_projection(cnn_out)

        # Numeric branch
        numeric_out = self.numeric_mlp(numeric_features)

        # Fusion
        combined = torch.cat([cnn_features, numeric_out], dim=-1)
        output = self.fusion(combined)

        return output


class CompactCrossDimensionalBackbone(nn.Module):
    """Strided xdim encoder sized for high-throughput search self-play.

    The observation and action contracts are identical to the established xdim
    models. Unlike :class:`CrossDimensionalBackbone`, the spatial branch is
    downsampled before projection, avoiding a tens-of-millions-parameter dense
    layer over the full 11x21 grid.
    """

    def __init__(self, config: CrossDimensionalBackboneConfig):
        super().__init__()
        if not config.cnn_channels:
            raise ValueError("cnn_channels cannot be empty")
        if not config.numeric_hidden_dims:
            raise ValueError("numeric_hidden_dims cannot be empty")
        self.config = config
        self.numeric_dim = config.numeric_dim
        self.board_height = config.board_height
        self.board_width = config.board_width
        self.board_channels = config.board_channels
        self.board_layout = config.board_layout

        cnn_layers: list[nn.Module] = []
        in_channels = config.board_channels
        current_h, current_w = _board_spatial_shape(config)
        kernel_size = _board_kernel_size(config)
        pad_h = (kernel_size[0] - 1) // 2
        pad_w = (kernel_size[1] - 1) // 2
        for out_channels in config.cnn_channels:
            cnn_layers.append(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=2,
                    padding=(pad_h, pad_w),
                )
            )
            groups = next(
                candidate
                for candidate in range(min(8, out_channels), 0, -1)
                if out_channels % candidate == 0
            )
            cnn_layers.append(nn.GroupNorm(groups, out_channels))
            cnn_layers.append(nn.SiLU())
            in_channels = out_channels
            current_h = (current_h + 1) // 2
            current_w = (current_w + 1) // 2
        self.cnn = nn.Sequential(*cnn_layers)
        spatial_dim = config.cnn_channels[-1] * current_h * current_w
        self.spatial_projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(spatial_dim, config.fusion_hidden_dim),
            nn.LayerNorm(config.fusion_hidden_dim),
            nn.SiLU(),
        )

        numeric_layers: list[nn.Module] = []
        in_dim = config.numeric_dim
        for hidden_dim in config.numeric_hidden_dims:
            numeric_layers.extend(
                (nn.Linear(in_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU())
            )
            in_dim = hidden_dim
        self.numeric_mlp = nn.Sequential(*numeric_layers)

        self.fusion = nn.Sequential(
            nn.Linear(config.fusion_hidden_dim + in_dim, config.fusion_hidden_dim),
            nn.LayerNorm(config.fusion_hidden_dim),
            nn.SiLU(),
            nn.Linear(config.fusion_hidden_dim, config.output_dim),
            nn.LayerNorm(config.output_dim),
            nn.SiLU(),
        )
        self.output_dim = config.output_dim
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                orthogonal_init(module)
            elif isinstance(module, nn.Conv2d):
                nn.init.orthogonal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        numeric_features = x[:, : self.numeric_dim]
        board_tensor = _reshape_board_tensor(x[:, self.numeric_dim :], self.config)
        spatial_features = self.spatial_projection(self.cnn(board_tensor))
        numeric_features = self.numeric_mlp(numeric_features)
        return self.fusion(torch.cat((spatial_features, numeric_features), dim=-1))


class _CatanHeteroGraphLayer(nn.Module):
    """Pre-norm residual node/tile message-passing layer used by Nexus v3."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.node_norm = nn.LayerNorm(hidden_dim)
        self.tile_norm = nn.LayerNorm(hidden_dim)
        self.node_self = nn.Linear(hidden_dim, hidden_dim)
        self.node_from_node = nn.Linear(hidden_dim, hidden_dim)
        self.node_from_tile = nn.Linear(hidden_dim, hidden_dim)
        self.tile_self = nn.Linear(hidden_dim, hidden_dim)
        self.tile_from_node = nn.Linear(hidden_dim, hidden_dim)

    def forward(
        self,
        nodes: torch.Tensor,
        tiles: torch.Tensor,
        node_adjacency: torch.Tensor,
        tile_to_node: torch.Tensor,
        node_to_tile: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        normalized_nodes = self.node_norm(nodes)
        normalized_tiles = self.tile_norm(tiles)
        node_update = (
            self.node_self(normalized_nodes)
            + torch.matmul(node_adjacency, self.node_from_node(normalized_nodes))
            + torch.matmul(tile_to_node, self.node_from_tile(normalized_tiles))
        )
        tile_update = self.tile_self(normalized_tiles) + torch.matmul(
            node_to_tile,
            self.tile_from_node(normalized_nodes),
        )
        return nodes + torch.relu(node_update), tiles + torch.relu(tile_update)


class CatanGraphBackbone(nn.Module):
    """Topology-aware shared trunk over the unchanged Catanatron observation.

    Catanatron stores board information on a fixed 21x11 lattice.  This module
    gathers the active map nodes, linearly recovers per-tile resource/robber
    planes from their six-node incidence pattern, and applies heterogeneous
    message passing.  The returned tensor contains pooled, node, and tile
    embeddings so policy heads can retain spatial identity.
    """

    node_positions: torch.Tensor
    edge_positions: torch.Tensor
    node_edge_incidence: torch.Tensor
    road_channel_indices: torch.Tensor
    node_adjacency: torch.Tensor
    tile_to_node: torch.Tensor
    node_to_tile: torch.Tensor
    tile_decoder: torch.Tensor

    def __init__(self, config: CatanGraphBackboneConfig):
        super().__init__()
        if config.board_layout != "width_height":
            raise ValueError("Catan graph backbones require board_layout='width_height'")
        if config.num_layers < 1:
            raise ValueError("Catan graph backbones require at least one graph layer")
        if config.hidden_dim < 1 or config.global_hidden_dim < 1:
            raise ValueError("Catan graph hidden dimensions must be positive")
        expected_input = (
            config.numeric_dim + config.board_width * config.board_height * config.board_channels
        )
        if config.input_dim != expected_input:
            raise ValueError(
                f"Catan graph input_dim={config.input_dim} does not match "
                f"numeric+board dimensions ({expected_input})"
            )

        from catanatron.gym.board_tensor_features import get_node_and_edge_maps
        from catanatron.models.board import get_edges
        from catanatron.models.map import build_map

        catan_map = build_map(config.map_type)
        node_ids = sorted(catan_map.land_nodes)
        node_to_local = {node_id: index for index, node_id in enumerate(node_ids)}
        tile_items = list(catan_map.land_tiles.items())
        num_nodes = len(node_ids)
        num_tiles = len(tile_items)

        node_positions, edge_position_map = get_node_and_edge_maps()
        positions = torch.tensor(
            [node_positions[node_id] for node_id in node_ids],
            dtype=torch.long,
        )

        incidence = torch.zeros(num_nodes, num_tiles, dtype=torch.float32)
        for tile_index, (_, tile) in enumerate(tile_items):
            for node_id in tile.nodes.values():
                incidence[node_to_local[node_id], tile_index] = 1.0

        node_adjacency = torch.zeros(num_nodes, num_nodes, dtype=torch.float32)
        edges = list(get_edges(catan_map.land_nodes))
        node_edge_incidence = torch.zeros(num_nodes, len(edges), dtype=torch.float32)
        edge_pairs: list[tuple[int, int]] = []
        edge_positions: list[tuple[int, int]] = []
        for edge_index, (first, second) in enumerate(edges):
            first_local = node_to_local[first]
            second_local = node_to_local[second]
            node_adjacency[first_local, second_local] = 1.0
            node_adjacency[second_local, first_local] = 1.0
            node_edge_incidence[first_local, edge_index] = 1.0
            node_edge_incidence[second_local, edge_index] = 1.0
            edge_pairs.append((first_local, second_local))
            edge_positions.append(edge_position_map[(first, second)])
        node_adjacency /= node_adjacency.sum(dim=1, keepdim=True).clamp_min(1.0)
        tile_to_node = incidence / incidence.sum(dim=1, keepdim=True).clamp_min(1.0)
        # ``transpose`` returns a view.  Clone before normalizing so this does
        # not scale the raw incidence matrix used by ``tile_decoder`` below.
        node_to_tile = incidence.transpose(0, 1).clone()
        node_to_tile = node_to_tile / node_to_tile.sum(
            dim=1,
            keepdim=True,
        ).clamp_min(1.0)

        # Board resource and robber planes are sums of per-tile contributions
        # at their six corner nodes.  The incidence matrix has full column rank
        # on supported Catan maps, so its pseudoinverse recovers those tile
        # values without introducing a second observation representation.
        tile_decoder = torch.linalg.pinv(incidence)

        self.config = config
        self.numeric_dim = config.numeric_dim
        self.board_width = config.board_width
        self.board_height = config.board_height
        self.board_channels = config.board_channels
        self.hidden_dim = config.hidden_dim
        self.global_hidden_dim = config.global_hidden_dim
        self.head_hidden_dim = config.head_hidden_dim
        self.num_nodes = num_nodes
        self.num_tiles = num_tiles
        self.num_edges = len(edge_pairs)
        self.node_ids = tuple(node_ids)
        self.tile_coordinates = tuple(coordinate for coordinate, _ in tile_items)
        self.edge_pairs = tuple(edge_pairs)
        self.pooled_dim = 2 * config.hidden_dim + config.global_hidden_dim
        self.output_dim = self.pooled_dim + (num_nodes + num_tiles) * config.hidden_dim

        self.register_buffer("node_positions", positions, persistent=False)
        self.register_buffer(
            "edge_positions",
            torch.tensor(edge_positions, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer("node_edge_incidence", node_edge_incidence, persistent=False)
        self.register_buffer(
            "road_channel_indices",
            torch.arange(config.num_players, dtype=torch.long) * 2 + 1,
            persistent=False,
        )
        self.register_buffer("node_adjacency", node_adjacency, persistent=False)
        self.register_buffer("tile_to_node", tile_to_node, persistent=False)
        self.register_buffer("node_to_tile", node_to_tile, persistent=False)
        self.register_buffer("tile_decoder", tile_decoder, persistent=False)

        self.global_projection = nn.Linear(config.numeric_dim, config.global_hidden_dim)
        self.node_projection = nn.Linear(config.board_channels, config.hidden_dim)
        # Five resource-production planes plus the robber plane.
        self.tile_projection = nn.Linear(6, config.hidden_dim)
        self.node_injection = nn.Sequential(
            nn.Linear(config.hidden_dim + config.global_hidden_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.ReLU(),
        )
        self.tile_injection = nn.Sequential(
            nn.Linear(config.hidden_dim + config.global_hidden_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.ReLU(),
        )
        self.graph_layers = nn.ModuleList(
            _CatanHeteroGraphLayer(config.hidden_dim) for _ in range(config.num_layers)
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                orthogonal_init(module)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2 or x.shape[1] != self.config.input_dim:
            raise ValueError(
                f"Expected Catan graph input [batch, {self.config.input_dim}], got {tuple(x.shape)}"
            )
        batch_size = x.shape[0]
        global_features = torch.relu(self.global_projection(x[:, : self.numeric_dim]))
        board = x[:, self.numeric_dim :].reshape(
            batch_size,
            self.board_width,
            self.board_height,
            self.board_channels,
        )
        node_raw = board[
            :,
            self.node_positions[:, 0],
            self.node_positions[:, 1],
            :,
        ]
        # Road ownership lives at Catanatron's edge pixels. Aggregate incident
        # roads into the otherwise-zero road channels at each endpoint so the
        # graph layers retain local network topology without changing the
        # observation or checkpoint tensor shapes.
        edge_raw = board[
            :,
            self.edge_positions[:, 0],
            self.edge_positions[:, 1],
            :,
        ]
        edge_roads = edge_raw.index_select(-1, self.road_channel_indices)
        incident_roads = torch.einsum(
            "ne,bep->bnp",
            self.node_edge_incidence,
            edge_roads,
        ) / 3.0
        node_raw = node_raw.clone()
        node_raw[..., self.road_channel_indices] = incident_roads
        resource_offset = 2 * self.config.num_players
        tile_source = node_raw[:, :, resource_offset : resource_offset + 6]
        tile_raw = torch.einsum("tn,bnc->btc", self.tile_decoder, tile_source)

        nodes = torch.relu(self.node_projection(node_raw))
        tiles = torch.relu(self.tile_projection(tile_raw))
        node_global = global_features.unsqueeze(1).expand(-1, self.num_nodes, -1)
        tile_global = global_features.unsqueeze(1).expand(-1, self.num_tiles, -1)
        nodes = self.node_injection(torch.cat((nodes, node_global), dim=-1))
        tiles = self.tile_injection(torch.cat((tiles, tile_global), dim=-1))
        for layer in self.graph_layers:
            nodes, tiles = layer(
                nodes,
                tiles,
                self.node_adjacency,
                self.tile_to_node,
                self.node_to_tile,
            )

        pooled = torch.cat(
            (nodes.mean(dim=1), tiles.mean(dim=1), global_features),
            dim=-1,
        )
        return torch.cat(
            (pooled, nodes.reshape(batch_size, -1), tiles.reshape(batch_size, -1)),
            dim=-1,
        )


class ResidualCrossDimensionalBackbone(nn.Module):
    """
    Cross-dimensional backbone with residual connections between the CNN
    and MLP branches, following Figure 5 from the paper: https://arxiv.org/pdf/2008.07079.
    """

    def __init__(self, config: CrossDimensionalBackboneConfig):
        super().__init__()
        self.config = config

        # Store dimensions for input splitting
        self.numeric_dim = config.numeric_dim
        self.board_height = config.board_height
        self.board_width = config.board_width
        self.board_channels = config.board_channels
        self.board_layout = config.board_layout
        self.board_flat_dim = config.board_height * config.board_width * config.board_channels

        # CNN branch
        self.cnn_layers = nn.ModuleList()
        self.cnn_norms = nn.ModuleList()
        in_channels = config.board_channels
        kernel_size = _board_kernel_size(config)

        for out_channels in config.cnn_channels:
            pad_h = (kernel_size[0] - 1) // 2
            pad_w = (kernel_size[1] - 1) // 2

            self.cnn_layers.append(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    padding=(pad_h, pad_w),
                )
            )
            self.cnn_norms.append(nn.BatchNorm2d(out_channels))
            in_channels = out_channels

        # CNN output projection
        cnn_spatial_size = config.board_height * config.board_width
        cnn_output_size = config.cnn_channels[-1] * cnn_spatial_size

        self.cnn_projection = nn.Linear(cnn_output_size, config.fusion_hidden_dim)

        # MLP branch
        self.mlp_layers = nn.ModuleList()
        self.mlp_norms = nn.ModuleList()
        in_dim = config.numeric_dim

        for hidden_dim in config.numeric_hidden_dims:
            self.mlp_layers.append(nn.Linear(in_dim, hidden_dim))
            self.mlp_norms.append(nn.LayerNorm(hidden_dim))
            in_dim = hidden_dim

        # Cross-dimensional interaction layers
        # These allow information to flow between branches
        numeric_out_dim = config.numeric_hidden_dims[-1]

        # Inflation: scalar -> spatial (broadcast scalar info to all positions)
        self.inflate = nn.Linear(numeric_out_dim, config.cnn_channels[-1])

        # Deflation: spatial -> scalar (global pooling + projection)
        self.deflate = nn.Linear(config.cnn_channels[-1], numeric_out_dim)

        # Fusion
        fusion_input_dim = config.fusion_hidden_dim + numeric_out_dim
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, config.output_dim),
            nn.LayerNorm(config.output_dim),
            nn.ReLU(),
        )

        self.output_dim = config.output_dim
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                orthogonal_init(module)
            elif isinstance(module, nn.Conv2d):
                nn.init.orthogonal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Split flattened input into numeric and board components
        numeric_features = x[:, : self.numeric_dim]
        board_flat = x[:, self.numeric_dim :]

        # Reshape board to (batch, H, W, C) then permute to (batch, C, H, W) for Conv2d
        board_tensor = _reshape_board_tensor(board_flat, self.config)

        # Process CNN branch
        x_spatial = board_tensor
        for conv, norm in zip(self.cnn_layers, self.cnn_norms):
            x_spatial = torch.relu(norm(conv(x_spatial)))

        # Process MLP branch
        x_scalar = numeric_features
        for linear, norm in zip(self.mlp_layers, self.mlp_norms):
            x_scalar = torch.relu(norm(linear(x_scalar)))

        # Cross-dimensional interaction
        # Inflate: add scalar info to spatial features
        inflated = self.inflate(x_scalar)  # (batch, cnn_channels[-1])
        inflated = inflated.unsqueeze(-1).unsqueeze(-1)  # (batch, C, 1, 1)
        x_spatial = x_spatial + inflated  # broadcast add

        # Deflate: add spatial info to scalar features
        deflated = x_spatial.mean(dim=(-2, -1))  # global avg pool: (batch, C)
        deflated = self.deflate(deflated)  # (batch, numeric_out_dim)
        x_scalar = x_scalar + deflated

        # === Final projection and fusion ===
        x_spatial_flat = x_spatial.flatten(start_dim=1)
        x_spatial_proj = torch.relu(self.cnn_projection(x_spatial_flat))

        combined = torch.cat([x_spatial_proj, x_scalar], dim=-1)
        output = self.fusion(combined)

        return output


def create_backbone(config: BackboneConfig) -> Tuple[nn.Module, int]:
    """
    Factory function to create a backbone from configuration.

    Args:
        config: BackboneConfig specifying architecture and parameters

    Returns:
        Tuple of (backbone_module, output_dimension)
    """
    if config.architecture == "mlp":
        assert isinstance(config.args, MLPBackboneConfig)
        return MLPBackbone(config.args), config.args.hidden_dims[-1]
    elif config.architecture == "cross_dimensional":
        assert isinstance(config.args, CrossDimensionalBackboneConfig)
        backbone = CrossDimensionalBackbone(config.args)
        return backbone, config.args.output_dim
    elif config.architecture == "residual_cross_dimensional":
        assert isinstance(config.args, CrossDimensionalBackboneConfig)
        backbone = ResidualCrossDimensionalBackbone(config.args)
        return backbone, config.args.output_dim
    elif config.architecture == "compact_cross_dimensional":
        assert isinstance(config.args, CrossDimensionalBackboneConfig)
        backbone = CompactCrossDimensionalBackbone(config.args)
        return backbone, config.args.output_dim
    elif config.architecture == "catan_graph":
        assert isinstance(config.args, CatanGraphBackboneConfig)
        backbone = CatanGraphBackbone(config.args)
        return backbone, backbone.output_dim
    else:
        raise ValueError(f"Unsupported architecture: {config.architecture}")
