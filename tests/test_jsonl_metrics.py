from __future__ import annotations

import json

import numpy as np
import pytest

from catanrl.utils.jsonl_metrics import append_jsonl_metrics


def test_append_jsonl_metrics_is_incremental_and_json_safe(tmp_path) -> None:
    path = tmp_path / "experiment" / "metrics.jsonl"

    append_jsonl_metrics(
        str(path),
        {
            "iteration": 1,
            "loss": np.float32(0.25),
            "non_finite": float("inf"),
            "nested": {"counts": [np.int64(2), 3]},
        },
    )
    append_jsonl_metrics(str(path), {"iteration": 2, "loss": 0.125})

    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert records == [
        {
            "iteration": 1,
            "loss": 0.25,
            "nested": {"counts": [2, 3]},
            "non_finite": None,
        },
        {"iteration": 2, "loss": 0.125},
    ]


def test_append_jsonl_metrics_rejects_opaque_values(tmp_path) -> None:
    with pytest.raises(TypeError, match="Unsupported local metric value"):
        append_jsonl_metrics(str(tmp_path / "metrics.jsonl"), {"bad": object()})
