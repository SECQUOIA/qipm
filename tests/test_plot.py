"""Tests for plot input filtering and degenerate ranges."""

import json

import numpy as np

from plot import (
    _cycle_counts,
    _difficulty_bins,
    _load_advantage_data,
    _truncate_at_zero,
)


def test_advantage_loading_keeps_zero_runtime_and_filters_bad_counts(tmp_path) -> None:
    instance_dir = tmp_path / "cls" / "item"
    instance_dir.mkdir(parents=True)
    (instance_dir / "item.data").write_text(
        json.dumps(
            {
                "runtime_glpk": 0.0,
                "cycle_count_mnes": 10,
                "cycle_count_oss": float("inf"),
            }
        )
    )
    loaded = _load_advantage_data(["cls"], tmp_path, "runtime_glpk")
    assert len(loaded["cls"]) == 1
    np.testing.assert_array_equal(_cycle_counts(loaded["cls"], "mnes"), [10.0])
    assert _cycle_counts(loaded["cls"], "oss") is None


def test_difficulty_bins_and_zero_truncation_remain_visible() -> None:
    bins = _difficulty_bins(np.array([5.0, 5.0]))
    assert bins[0] < 5.0 < bins[-1]
    times = np.array([1.0, 2.0, 3.0])
    curve = np.zeros(3)
    truncated_times, truncated_curve = _truncate_at_zero(times, curve)
    assert len(truncated_times) == len(truncated_curve) == 2
