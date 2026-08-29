"""Tests for plot input filtering and degenerate ranges."""

import json

import numpy as np
import pytest

import plot as plot_module
from plot import (
    _advantage_pairs,
    _crossover_times,
    _cycle_counts,
    _difficulty_bins,
    _ecdf,
    _load_advantage_data,
    _load_difficulty_data,
    _load_ratio_data,
    _truncate_at_zero,
    _validate_cycle_time,
    plot_ratio,
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


def test_difficulty_loading_excludes_unrepresentable_sparsity(tmp_path) -> None:
    instance_dir = tmp_path / "cls" / "huge"
    instance_dir.mkdir(parents=True)
    (instance_dir / "huge.data").write_text(json.dumps({
        "sparsity_mnes": 10**1000,
        "cond_mnes": 1.0,
    }))

    assert _load_difficulty_data(["cls"], tmp_path, "mnes") == {}


def test_ratio_loading_filters_and_computes_both_ratios(tmp_path) -> None:
    records = {
        "valid": {
            "runtime_highs_std": 2.0,
            "cycle_count_mnes": 100,
            "qlsa_queries_mnes": 10,
            "tomography_reps_mnes": 10,
        },
        "zero": {
            "runtime_highs_std": 0.0,
            "cycle_count_mnes": 100,
            "qlsa_queries_mnes": 10,
            "tomography_reps_mnes": 10,
        },
        "legacy": {"runtime_highs_std": 2.0, "cycle_count_mnes": 100},
        "missing_runtime": {
            "cycle_count_mnes": 100,
            "qlsa_queries_mnes": 10,
            "tomography_reps_mnes": 10,
        },
        "unbenchmarked": {"runtime_highs_std": 2.0},
        "unbenchmarked_missing_runtime": {},
    }
    for name, record in records.items():
        instance_dir = tmp_path / "cls" / name
        instance_dir.mkdir(parents=True)
        (instance_dir / f"{name}.data").write_text(json.dumps(record))

    data, skipped, eligible = _load_ratio_data(
        ["cls"], tmp_path, "runtime_highs_std", 0.5, ["mnes"]
    )
    total, prep = data["cls"]["mnes"]
    np.testing.assert_array_equal(total, [25.0])
    np.testing.assert_array_equal(prep, [2.5])
    assert skipped == {
        "missing_runtime": 1,
        "invalid_runtime": 1,
        "needs_refresh": 1,
        "unplottable_overflow": 0,
    }
    assert eligible == 1


def test_ratio_cycle_time_must_be_positive_and_finite() -> None:
    for value in (0.0, -1.0, float("inf"), float("nan")):
        with pytest.raises(ValueError):
            _validate_cycle_time(value)


def test_ratio_loading_checks_large_integer_invariant_exactly(tmp_path) -> None:
    queries = 38_185_778_926_698_981
    repetitions = 5_088_745
    records = {
        "valid": {
            "runtime_highs_std": 1.0,
            "cycle_count_mnes": queries * repetitions,
            "qlsa_queries_mnes": queries,
            "tomography_reps_mnes": repetitions,
        },
        "corrupt": {
            "runtime_highs_std": 1.0,
            "cycle_count_mnes": 2**53 + 1,
            "qlsa_queries_mnes": 2**53,
            "tomography_reps_mnes": 1,
        },
    }
    for name, record in records.items():
        instance_dir = tmp_path / "cls" / name
        instance_dir.mkdir(parents=True)
        (instance_dir / f"{name}.data").write_text(json.dumps(record))

    data, skipped, eligible = _load_ratio_data(
        ["cls"], tmp_path, "runtime_highs_std", 1e-20, ["mnes"]
    )

    total, prep = data["cls"]["mnes"]
    assert len(total) == len(prep) == 1
    assert skipped["needs_refresh"] == 1
    assert eligible == 1


def test_huge_count_flows_through_advantage_and_ratio_loaders(tmp_path) -> None:
    huge_count = 10**1000
    instance_dir = tmp_path / "cls" / "huge"
    instance_dir.mkdir(parents=True)
    (instance_dir / "huge.data").write_text(json.dumps({
        "runtime_highs_std": 2.0,
        "cycle_count_mnes": huge_count,
        "qlsa_queries_mnes": huge_count,
        "tomography_reps_mnes": 1,
    }))

    advantage = _load_advantage_data(["cls"], tmp_path, "runtime_highs_std")
    counts, runtimes = _advantage_pairs(
        advantage, ["mnes"], "runtime_highs_std"
    )["cls"]["mnes"]
    np.testing.assert_array_equal(_crossover_times(counts, runtimes), [0.0])

    ratio, skipped, eligible = _load_ratio_data(
        ["cls"], tmp_path, "runtime_highs_std", 0.5, ["mnes"]
    )
    assert ratio == {}
    assert skipped["unplottable_overflow"] == 1
    assert eligible == 0


def test_ratio_loader_recovers_finite_ratio_after_float_overflow(tmp_path) -> None:
    count = 10**309
    instance_dir = tmp_path / "cls" / "recoverable"
    instance_dir.mkdir(parents=True)
    (instance_dir / "recoverable.data").write_text(json.dumps({
        "runtime_highs_std": 1.0,
        "cycle_count_mnes": count,
        "qlsa_queries_mnes": count,
        "tomography_reps_mnes": 1,
    }))

    data, skipped, eligible = _load_ratio_data(
        ["cls"], tmp_path, "runtime_highs_std", 8e-10, ["mnes"]
    )

    total, prep = data["cls"]["mnes"]
    assert total[0] == pytest.approx(8e299)
    assert prep[0] == pytest.approx(8e299)
    assert skipped["unplottable_overflow"] == 0
    assert eligible == 1


def test_ratio_loader_counts_exact_retry_overflow(tmp_path) -> None:
    count = 10**400
    instance_dir = tmp_path / "cls" / "unplottable"
    instance_dir.mkdir(parents=True)
    (instance_dir / "unplottable.data").write_text(json.dumps({
        "runtime_highs_std": 1e-8,
        "cycle_count_mnes": count,
        "qlsa_queries_mnes": count,
        "tomography_reps_mnes": 1,
    }))

    data, skipped, eligible = _load_ratio_data(
        ["cls"], tmp_path, "runtime_highs_std", 1.0, ["mnes"]
    )

    assert data == {}
    assert skipped["unplottable_overflow"] == 1
    assert eligible == 0


def test_ratio_loader_counts_nonfinite_float_result(tmp_path) -> None:
    instance_dir = tmp_path / "cls" / "nonfinite"
    instance_dir.mkdir(parents=True)
    (instance_dir / "nonfinite.data").write_text(json.dumps({
        "runtime_highs_std": 1e-8,
        "cycle_count_mnes": 10,
        "qlsa_queries_mnes": 10,
        "tomography_reps_mnes": 1,
    }))

    data, skipped, eligible = _load_ratio_data(
        ["cls"], tmp_path, "runtime_highs_std", 1e308, ["mnes"]
    )

    assert data == {}
    assert skipped["unplottable_overflow"] == 1
    assert eligible == 0


def test_ecdf_starts_at_zero_and_reaches_one_hundred() -> None:
    x_values, percentages = _ecdf(np.array([2.0, 1.0]))
    assert x_values[0] < x_values[1]
    np.testing.assert_array_equal(x_values[1:], [1.0, 2.0])
    np.testing.assert_array_equal(percentages, [0.0, 50.0, 100.0])


@pytest.mark.parametrize("style", ["box", "ecdf"])
def test_ratio_plot_styles_smoke(style, tmp_path, monkeypatch, capsys) -> None:
    instance_dir = tmp_path / "cls" / "item"
    instance_dir.mkdir(parents=True)
    (instance_dir / "item.data").write_text(json.dumps({
        "runtime_highs_std": 2.0,
        "cycle_count_mnes": 100,
        "qlsa_queries_mnes": 10,
        "tomography_reps_mnes": 10,
        "cycle_count_oss": 200,
        "qlsa_queries_oss": 20,
        "tomography_reps_oss": 10,
    }))
    monkeypatch.setitem(plot_module._RCPARAMS, "text.usetex", False)
    output = tmp_path / f"ratio-{style}.pdf"

    plot_ratio(
        ["cls"], tmp_path, output,
        runtime_key="runtime_highs_std",
        baseline_label="highs-std",
        cycle_time=0.5,
        style=style,
    )

    assert output.is_file()
    output_text = capsys.readouterr().out
    assert "benchmark.py --refresh-counts" in output_text
    assert "overflow plotting" in output_text
    assert f"Saved to {output}" in output_text
