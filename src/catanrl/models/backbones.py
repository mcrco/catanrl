import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import List, Union, Tuple

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


@dataclass
class BackboneConfig:
    architecture: str
    args: Union[MLPBackboneConfig, CrossDimensionalBackboneConfig]


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
        self.board_flat_dim = config.board_height * config.board_width * config.board_channels

        # CNN branch for board tensor
        cnn_layers = []
        in_channels = config.board_channels
        current_h, current_w = config.board_height, config.board_width

        for out_channels in config.cnn_channels:
            # Use padding to maintain spatial dimensions initially
            pad_h = (config.cnn_kernel_size[0] - 1) // 2
            pad_w = (config.cnn_kernel_size[1] - 1) // 2

            cnn_layers.append(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=config.cnn_kernel_size,
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
        numeric_features = x[:, :self.numeric_dim]
        board_flat = x[:, self.numeric_dim:]

        # Reshape board to (batch, H, W, C) then permute to (batch, C, H, W) for Conv2d
        board_tensor = board_flat.reshape(
            -1, self.board_height, self.board_width, self.board_channels
        )
        board_tensor = board_tensor.permute(0, 3, 1, 2)

        # CNN branch
        cnn_out = self.cnn(board_tensor)
        cnn_features = self.cnn_projection(cnn_out)

        # Numeric branch
        numeric_out = self.numeric_mlp(numeric_features)

        # Fusion
        combined = torch.cat([cnn_features, numeric_out], dim=-1)
        output = self.fusion(combined)

        return output


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
        self.board_flat_dim = config.board_height * config.board_width * config.board_channels

        # CNN branch
        self.cnn_layers = nn.ModuleList()
        self.cnn_norms = nn.ModuleList()
        in_channels = config.board_channels

        for out_channels in config.cnn_channels:
            pad_h = (config.cnn_kernel_size[0] - 1) // 2
            pad_w = (config.cnn_kernel_size[1] - 1) // 2

            self.cnn_layers.append(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=config.cnn_kernel_size,
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
        numeric_features = x[:, :self.numeric_dim]
        board_flat = x[:, self.numeric_dim:]

        # Reshape board to (batch, H, W, C) then permute to (batch, C, H, W) for Conv2d
        board_tensor = board_flat.reshape(
            -1, self.board_height, self.board_width, self.board_channels
        )
        board_tensor = board_tensor.permute(0, 3, 1, 2)

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
    else:
        raise ValueError(f"Unsupported architecture: {config.architecture}")
