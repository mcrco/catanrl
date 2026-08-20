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
