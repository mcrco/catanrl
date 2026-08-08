from __future__ import annotations

import pytest

from catanrl.eval.canopy_reference import (
    CANOPY_NEXUS_V3_CHECKPOINT_SHA256,
    CANOPY_NEXUS_V3_RELEASE_COMMIT,
    CANOPY_NEXUS_V3_RELEASE_TAG,
    parse_canopy_tournament_summary,
    validate_official_nexus_v3_reference,
)


def test_parse_canopy_tournament_summary_uses_final_result_and_strips_ansi():
    output = """
W 1-2-0 | 100 evals/s
\x1b[32mINFO\x1b[0m W 18/40 (45.0%) | L 20/40 (50.0%) | D 2/40 (5.0%) | depth 9.7/31 | 4m02s
"""

    summary = parse_canopy_tournament_summary(output)

    assert summary.games == 40
    assert summary.wins == 18
    assert summary.losses == 20
    assert summary.draws == 2
    assert summary.win_rate == pytest.approx(0.45)
    assert summary.score_rate == pytest.approx(0.475)
    assert summary.draw_rate == pytest.approx(0.05)
    assert summary.win_rate_ci95_low < summary.win_rate
    assert summary.win_rate_ci95_high > summary.win_rate
    assert summary.mean_search_depth == pytest.approx(9.7)
    assert summary.maximum_search_depth == 31


def test_official_canopy_reference_requires_release_provenance():
    config = {
        "release_tag": CANOPY_NEXUS_V3_RELEASE_TAG,
        "release_commit": CANOPY_NEXUS_V3_RELEASE_COMMIT,
        "checkpoint_sha256": CANOPY_NEXUS_V3_CHECKPOINT_SHA256,
    }

    validate_official_nexus_v3_reference(config)

    for key in config:
        invalid = dict(config)
        invalid[key] = "wrong"
        with pytest.raises(ValueError, match=key):
            validate_official_nexus_v3_reference(invalid)


@pytest.mark.parametrize(
    "output, message",
    [
        ("W 1-2-0", "no final"),
        ("W 1/4 (25%) | L 2/5 (40%) | D 1/4 (25%)", "inconsistent"),
        ("W 1/4 (25%) | L 1/4 (25%) | D 1/4 (25%)", "do not add up"),
    ],
)
def test_parse_canopy_tournament_summary_rejects_invalid_output(output, message):
    with pytest.raises(ValueError, match=message):
        parse_canopy_tournament_summary(output)
