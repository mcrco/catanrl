from __future__ import annotations

import numpy as np
import torch

from catanrl.algorithms.common import mask_action_logits


def test_clamping_does_not_make_masked_actions_sampleable() -> None:
    logits = torch.tensor([[-1_000.0, -2_000.0, 1_000.0]])
    action_mask = np.array([[True, True, False]])

    masked_logits, effective_mask = mask_action_logits(
        logits,
        action_mask,
        clamp_range=(-100.0, 100.0),
    )

    assert effective_mask.tolist() == action_mask.tolist()
    assert masked_logits[0, :2].tolist() == [-100.0, -100.0]
    assert torch.isneginf(masked_logits[0, 2])

    samples = torch.distributions.Categorical(logits=masked_logits).sample((1_000,))
    assert set(samples.flatten().tolist()) <= {0, 1}
