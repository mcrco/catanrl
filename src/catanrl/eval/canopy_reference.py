"""Parse the released Canopy tournament harness' final result line."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass

from catanrl.eval.reporting import wilson_interval

CANOPY_NEXUS_V3_RELEASE_TAG = "catan-nexus-v3"
CANOPY_NEXUS_V3_RELEASE_COMMIT = "6185983a88ba6802e7fa9893cef5a76a15de2595"
CANOPY_NEXUS_V3_CHECKPOINT_SHA256 = (
    "f8e4e6858930243a30243e38c1b2b96b1a8da23970f5cba69906c65b268c60cc"
)

_ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_RESULT = re.compile(
    r"\bW\s+(?P<wins>\d+)/(?P<w_total>\d+)\s+\([^)]*\)\s*\|\s*"
    r"L\s+(?P<losses>\d+)/(?P<l_total>\d+)\s+\([^)]*\)\s*\|\s*"
    r"D\s+(?P<draws>\d+)/(?P<d_total>\d+)\s+\([^)]*\)"
    r"(?:\s*\|\s*depth\s+(?P<mean_depth>[0-9.eE+-]+)/(?P<max_depth>\d+))?"
)


@dataclass(frozen=True)
class CanopyTournamentSummary:
    games: int
    wins: int
    losses: int
    draws: int
    win_rate: float
    score_rate: float
    draw_rate: float
    win_rate_ci95_low: float
    win_rate_ci95_high: float
    mean_search_depth: float | None
    maximum_search_depth: int | None

    def payload(self) -> dict[str, int | float | None]:
        return asdict(self)


def validate_official_nexus_v3_reference(config: object) -> None:
    """Reject reference artifacts not tied to the official released model."""
    if not isinstance(config, dict):
        raise ValueError("Canopy reference config must be a JSON object")
    expected = {
        "release_tag": CANOPY_NEXUS_V3_RELEASE_TAG,
        "release_commit": CANOPY_NEXUS_V3_RELEASE_COMMIT,
        "checkpoint_sha256": CANOPY_NEXUS_V3_CHECKPOINT_SHA256,
    }
    for key, expected_value in expected.items():
        actual_value = config.get(key)
        if actual_value != expected_value:
            raise ValueError(
                f"Reference is not official Canopy Nexus-v3: "
                f"{key}={actual_value!r}, expected {expected_value!r}"
            )


def parse_canopy_tournament_summary(text: str) -> CanopyTournamentSummary:
    """Return the last complete tournament summary emitted by Canopy."""
    clean_text = _ANSI_ESCAPE.sub("", text).replace("\r", "\n")
    matches = list(_RESULT.finditer(clean_text))
    if not matches:
        raise ValueError("Canopy tournament output contains no final W/L/D summary")

    match = matches[-1]
    wins = int(match.group("wins"))
    losses = int(match.group("losses"))
    draws = int(match.group("draws"))
    totals = {
        int(match.group("w_total")),
        int(match.group("l_total")),
        int(match.group("d_total")),
    }
    if len(totals) != 1:
        raise ValueError(f"Canopy summary reports inconsistent game totals: {sorted(totals)}")
    games = totals.pop()
    if games < 1:
        raise ValueError("Canopy summary must contain at least one game")
    if wins + losses + draws != games:
        raise ValueError(
            f"Canopy summary outcomes do not add up to its total: {wins}+{losses}+{draws}!={games}"
        )

    ci_low, ci_high = wilson_interval(wins, games)
    mean_depth = match.group("mean_depth")
    max_depth = match.group("max_depth")
    return CanopyTournamentSummary(
        games=games,
        wins=wins,
        losses=losses,
        draws=draws,
        win_rate=wins / games,
        score_rate=(wins + 0.5 * draws) / games,
        draw_rate=draws / games,
        win_rate_ci95_low=ci_low,
        win_rate_ci95_high=ci_high,
        mean_search_depth=None if mean_depth is None else float(mean_depth),
        maximum_search_depth=None if max_depth is None else int(max_depth),
    )
