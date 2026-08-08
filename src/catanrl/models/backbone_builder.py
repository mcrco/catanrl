from __future__ import annotations

from typing import Literal, Sequence, Tuple

from .backbones import (
    BackboneConfig,
    CatanGraphBackboneConfig,
    CrossDimensionalBackboneConfig,
    MLPBackboneConfig,
)


def build_backbone_config(
    *,
    backbone_type: str,
    hidden_dims: Sequence[int],
    input_dim: int | None = None,
    board_height: int | None = None,
    board_width: int | None = None,
    board_channels: int | None = None,
    numeric_dim: int | None = None,
    xdim_cnn_channels: Sequence[int] = (),
    xdim_cnn_kernel_size: Tuple[int, int] = (3, 5),
    xdim_fusion_hidden_dim: int | None = None,
    xdim_board_layout: str = "width_height",
    num_players: int | None = None,
    map_type: Literal["BASE", "MINI", "TOURNAMENT"] | None = None,
    graph_hidden_dim: int = 256,
    graph_global_hidden_dim: int = 96,
    graph_num_layers: int = 4,
    graph_head_hidden_dim: int = 256,
    graph_normalize_inputs: bool = False,
    graph_semantic_inputs: bool = False,
) -> BackboneConfig:
    """Build either an MLP or cross-dimensional backbone config."""
    if backbone_type not in ("mlp", "xdim", "xdim_res", "xdim_compact", "catan_graph"):
        raise ValueError(f"Unknown backbone_type '{backbone_type}'")

    if backbone_type == "mlp":
        if input_dim is None:
            raise ValueError("input_dim is required for mlp backbones")
        return BackboneConfig(
            architecture="mlp",
            args=MLPBackboneConfig(input_dim=input_dim, hidden_dims=list(hidden_dims)),
        )

    if backbone_type == "catan_graph":
        required = {
            "input_dim": input_dim,
            "board_height": board_height,
            "board_width": board_width,
            "board_channels": board_channels,
            "numeric_dim": numeric_dim,
            "num_players": num_players,
            "map_type": map_type,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            raise ValueError("Catan graph backbones require " + ", ".join(missing))
        assert input_dim is not None
        assert board_height is not None
        assert board_width is not None
        assert board_channels is not None
        assert numeric_dim is not None
        assert num_players is not None
        assert map_type is not None
        return BackboneConfig(
            architecture="catan_graph",
            args=CatanGraphBackboneConfig(
                input_dim=input_dim,
                numeric_dim=numeric_dim,
                board_height=board_height,
                board_width=board_width,
                board_channels=board_channels,
                num_players=num_players,
                map_type=map_type,
                hidden_dim=graph_hidden_dim,
                global_hidden_dim=graph_global_hidden_dim,
                num_layers=graph_num_layers,
                head_hidden_dim=graph_head_hidden_dim,
                board_layout=xdim_board_layout,
                normalize_inputs=graph_normalize_inputs,
                semantic_inputs=graph_semantic_inputs,
            ),
        )

    if not xdim_cnn_channels:
        raise ValueError("xdim_cnn_channels cannot be empty")
    if None in (board_height, board_width, board_channels, numeric_dim):
        raise ValueError(
            "board_height, board_width, board_channels, and numeric_dim are required "
            "for cross-dimensional backbones"
        )
    assert board_height is not None
    assert board_width is not None
    assert board_channels is not None
    assert numeric_dim is not None

    output_dim = hidden_dims[-1] if hidden_dims else 256
    fusion_hidden_dim = xdim_fusion_hidden_dim if xdim_fusion_hidden_dim is not None else output_dim
    architecture = {
        "xdim": "cross_dimensional",
        "xdim_res": "residual_cross_dimensional",
        "xdim_compact": "compact_cross_dimensional",
    }[backbone_type]
    return BackboneConfig(
        architecture=architecture,
        args=CrossDimensionalBackboneConfig(
            board_height=board_height,
            board_width=board_width,
            board_channels=board_channels,
            numeric_dim=numeric_dim,
            cnn_channels=list(xdim_cnn_channels),
            cnn_kernel_size=xdim_cnn_kernel_size,
            numeric_hidden_dims=list(hidden_dims),
            fusion_hidden_dim=fusion_hidden_dim,
            output_dim=output_dim,
            board_layout=xdim_board_layout,
        ),
    )


__all__ = ["build_backbone_config"]
