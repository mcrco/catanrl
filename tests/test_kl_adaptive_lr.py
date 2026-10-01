import pytest

from catanrl.algorithms.ppo.marl_ppo_central_critic import kl_adaptive_lr

TARGET, LO, HI = 0.01, 1e-5, 3e-4


@pytest.mark.parametrize(
    ("kl", "expected"),
    [
        (0.025, 1e-4 / 1.5),  # above 2x target: shrink
        (0.004, 1e-4 * 1.5),  # below half the target: grow
        (0.01, 1e-4),  # within the band: unchanged
        (0.02, 1e-4),  # band edges are inclusive
        (0.005, 1e-4),
    ],
)
def test_kl_adaptive_lr_moves_toward_the_target(kl, expected):
    assert kl_adaptive_lr(1e-4, kl, TARGET, LO, HI) == pytest.approx(expected)


def test_kl_adaptive_lr_respects_bounds():
    assert kl_adaptive_lr(2.5e-4, 0.0, TARGET, LO, HI) == HI
    assert kl_adaptive_lr(1.2e-5, 1.0, TARGET, LO, HI) == LO
