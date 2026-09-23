"""Canonical board-tensor layout for flattened observations.

Catanatron's ``create_board_tensor`` (channels last) produces ``(W, H, C)``.
Flattened observations store that array in C-order. CNN backbones consume
``(N, C, H, W)`` for ``Conv2d``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Union, overload

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    import torch

BOARD_WIDTH = 21
BOARD_HEIGHT = 11

# reshape(..., W, H, C) -> (N, C, H, W)
_WHC_TO_NCHW_PERMUTE = (0, 3, 2, 1)


def as_whc(board: np.ndarray) -> np.ndarray:
    """Return a board tensor in canonical ``(W, H, C)`` layout.

    Accepts ``(W, H, C)`` or channels-first ``(C, W, H)``.
    """
    board_arr = np.asarray(board)
    if board_arr.ndim != 3:
        raise ValueError(f"Expected 3D board tensor, got shape {board_arr.shape}")

    if board_arr.shape[0] == BOARD_WIDTH and board_arr.shape[1] == BOARD_HEIGHT:
        return board_arr
    if board_arr.shape[1] == BOARD_WIDTH and board_arr.shape[2] == BOARD_HEIGHT:
        return np.transpose(board_arr, (1, 2, 0))
    raise ValueError(
        "Unrecognized board tensor layout; expected (W, H, C) or (C, W, H), "
        f"got {board_arr.shape}"
    )


def flatten_board(board: np.ndarray) -> np.ndarray:
    """Flatten a board tensor from ``(W, H, C)`` or ``(C, W, H)`` to C-order ``(W, H, C)``."""
    return np.ascontiguousarray(as_whc(board)).reshape(-1)


def _require_batched_flat(board_flat: Any, expected: int) -> Any:
    if getattr(board_flat, "ndim", None) == 1:
        batched = board_flat.reshape(1, -1)
    elif getattr(board_flat, "ndim", None) == 2:
        batched = board_flat
    else:
        raise ValueError(
            f"Expected 1D or 2D board vector, got shape {getattr(board_flat, 'shape', None)}"
        )
    if batched.shape[-1] != expected:
        raise ValueError(
            f"Board vector length {batched.shape[-1]} does not match expected {expected}"
        )
    return batched


@overload
def unflatten_board_nchw(
    board_flat: torch.Tensor,
    *,
    channels: int,
    width: int = BOARD_WIDTH,
    height: int = BOARD_HEIGHT,
) -> torch.Tensor: ...


@overload
def unflatten_board_nchw(
    board_flat: npt.NDArray[np.generic],
    *,
    channels: int,
    width: int = BOARD_WIDTH,
    height: int = BOARD_HEIGHT,
) -> npt.NDArray[np.generic]: ...


def unflatten_board_nchw(
    board_flat: Union[npt.NDArray[np.generic], torch.Tensor],
    *,
    channels: int,
    width: int = BOARD_WIDTH,
    height: int = BOARD_HEIGHT,
) -> Union[npt.NDArray[np.generic], torch.Tensor]:
    """Unflatten a canonical board vector to Conv2d ``(N, C, H, W)``.

    Input is C-order ``(W, H, C)``, with or without a leading batch axis.
    """
    expected = width * height * channels
    batched = _require_batched_flat(board_flat, expected)
    board = batched.reshape(batched.shape[0], width, height, channels)
    if hasattr(board, "permute"):
        return board.permute(*_WHC_TO_NCHW_PERMUTE).contiguous()
    return np.ascontiguousarray(np.transpose(board, _WHC_TO_NCHW_PERMUTE))
