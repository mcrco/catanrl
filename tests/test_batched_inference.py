"""Value transforms of the batched search inference helpers."""

from __future__ import annotations

import numpy as np

from catanrl.eval.batched_inference import to_zero_sum, value_transform_for_reward


def test_shaped_critics_are_mapped_to_zero_sum() -> None:
    assert value_transform_for_reward("shaped") == "win-probability"
    assert value_transform_for_reward("win") == "zero-sum"
    np.testing.assert_allclose(
        to_zero_sum(np.array([0.0, 0.5, 1.0]), "win-probability"), [-1.0, 0.0, 1.0]
    )
