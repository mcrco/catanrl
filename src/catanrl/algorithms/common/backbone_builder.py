from __future__ import annotations

from typing import Sequence, Tuple

from ...models.backbones import BackboneConfig, CrossDimensionalBackboneConfig, MLPBackboneConfig


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
) -> BackboneConfig:
    """Build either an MLP or cross-dimensional backbone config."""
    if backbone_type not in ("mlp", "xdim", "xdim_res"):
        raise ValueError(f"Unknown backbone_type '{backbone_type}'")

    if backbone_type == "mlp":
        if input_dim is None:
            raise ValueError("input_dim is required for mlp backbones")
        return BackboneConfig(
            architecture="mlp",
            args=MLPBackboneConfig(input_dim=input_dim, hidden_dims=list(hidden_dims)),
        )

    if not xdim_cnn_channels:
        raise ValueError("xdim_cnn_channels cannot be empty")
    if None in (board_height, board_width, board_channels, numeric_dim):
        raise ValueError(
            "board_height, board_width, board_channels, and numeric_dim are required "
            "for cross-dimensional backbones"
        )

    output_dim = hidden_dims[-1] if hidden_dims else 256
    fusion_hidden_dim = (
        xdim_fusion_hidden_dim if xdim_fusion_hidden_dim is not None else output_dim
    )
    architecture = (
        "residual_cross_dimensional" if backbone_type == "xdim_res" else "cross_dimensional"
    )
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
        ),
    )


__all__ = ["build_backbone_config"]
