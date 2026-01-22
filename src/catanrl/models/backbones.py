import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import List, Union, Tuple

from .utils import orthogonal_init


@dataclass
class MLPBackboneConfig:
    input_dim: int
    hidden_dims: List[int]


@dataclass
class BackboneConfig:
    architecture: str
    args: Union[MLPBackboneConfig]  # only MLP for now, CNN and GNN later


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


def create_backbone(config: BackboneConfig) -> Tuple[nn.Module, int]:
    if config.architecture == "mlp":
        return MLPBackbone(config.args), config.args.hidden_dims[-1]
    else:
        raise ValueError(f"Unsupported architecture: {config.architecture}")
