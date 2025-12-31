import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import List, Union, Tuple


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
        self.layers = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dims[0]),
            nn.ReLU(),
            nn.BatchNorm1d(config.hidden_dims[0]),
            nn.Dropout(0.2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


def create_backbone(config: BackboneConfig) -> Tuple[nn.Module, int]:
    if config.architecture == "mlp":
        return MLPBackbone(config.args), config.args.hidden_dims[-1]
    else:
        raise ValueError(f"Unsupported architecture: {config.architecture}")
