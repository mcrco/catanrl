from __future__ import annotations

import numpy as np
import pytest
import torch

from catanrl.features.board_tensor import (
    BOARD_HEIGHT,
    BOARD_WIDTH,
    flatten_board,
    unflatten_board_nchw,
)
from catanrl.models.backbones import (
    CompactCrossDimensionalBackbone,
    CrossDimensionalBackbone,
    CrossDimensionalBackboneConfig,
    ResidualCrossDimensionalBackbone,
)


def _marker_board(channels: int = 4) -> np.ndarray:
    """Unique value per spatial cell: 100 * w + h, copied across channels."""
    board = np.zeros((BOARD_WIDTH, BOARD_HEIGHT, channels), dtype=np.float32)
    for w in range(BOARD_WIDTH):
        for h in range(BOARD_HEIGHT):
            board[w, h, :] = 100 * w + h
    return board


def test_flatten_accepts_channels_first_and_last():
    board_whc = _marker_board()
    board_cwh = np.transpose(board_whc, (2, 0, 1))
    assert np.array_equal(flatten_board(board_whc), flatten_board(board_cwh))


def test_unflatten_places_cells_at_conv2d_h_w():
    channels = 4
    board_whc = _marker_board(channels)
    nchw = unflatten_board_nchw(flatten_board(board_whc), channels=channels)

    assert nchw.shape == (1, channels, BOARD_HEIGHT, BOARD_WIDTH)
    for w in range(BOARD_WIDTH):
        for h in range(BOARD_HEIGHT):
            assert nchw[0, 0, h, w] == 100 * w + h


def test_unflatten_does_not_reinterpret_as_hwc():
    channels = 4
    board_whc = _marker_board(channels)
    nchw = unflatten_board_nchw(flatten_board(board_whc), channels=channels)

    # The old bug reshaped C-order (W, H, C) as (H, W, C), wrapping (w=1, h=0)
    # onto Conv2d (h=0, w=11) instead of (h=0, w=1).
    assert nchw[0, 0, 0, 1] == 100 * 1 + 0
    assert nchw[0, 0, 0, 11] == 100 * 11 + 0


def test_unflatten_torch_matches_numpy():
    channels = 4
    flat = flatten_board(_marker_board(channels))
    numpy_nchw = unflatten_board_nchw(flat, channels=channels)
    torch_nchw = unflatten_board_nchw(torch.from_numpy(flat), channels=channels)
    np.testing.assert_array_equal(torch_nchw.numpy(), numpy_nchw)


def test_flatten_rejects_unknown_layout():
    with pytest.raises(ValueError, match="Unrecognized board tensor layout"):
        flatten_board(np.zeros((BOARD_HEIGHT, BOARD_WIDTH, 4), dtype=np.float32))


@pytest.mark.parametrize(
    "backbone_class",
    [CrossDimensionalBackbone, CompactCrossDimensionalBackbone, ResidualCrossDimensionalBackbone],
)
def test_xdim_backbones_feed_conv2d_canonical_nchw(backbone_class):
    channels, numeric_dim = 4, 3
    config = CrossDimensionalBackboneConfig(
        board_channels=channels,
        numeric_dim=numeric_dim,
        cnn_channels=[8],
        numeric_hidden_dims=[8],
        fusion_hidden_dim=16,
        output_dim=16,
    )
    backbone = backbone_class(config)
    first_conv = next(m for m in backbone.modules() if isinstance(m, torch.nn.Conv2d))
    seen = {}
    first_conv.register_forward_pre_hook(lambda _module, inputs: seen.setdefault("x", inputs[0]))

    board = _marker_board(channels)
    x = torch.from_numpy(np.concatenate([np.zeros(numeric_dim, np.float32), flatten_board(board)]))
    backbone(x.unsqueeze(0))

    expected = unflatten_board_nchw(torch.from_numpy(flatten_board(board)), channels=channels)
    torch.testing.assert_close(seen["x"], expected)
