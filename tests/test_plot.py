"""Tests for plot input filtering and degenerate ranges."""

import json

import numpy as np
import pytest

import plot as plot_module
from plot import (
    _advantage_pairs,
    _crossover_times,
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
                "benchmark_model": 2,
                "runtime_glpk": 0.0,
                "cycle_count_mnes": 10,
                "cycle_count_oss": float("inf"),
            }
        )
    )
    loaded = _load_advantage_data(["cls"], tmp_path, "runtime_glpk")
    assert len(loaded["cls"]) == 1
    pairs = _advantage_pairs(loaded, ["mnes", "oss"], "runtime_glpk")
    np.testing.assert_array_equal(pairs["cls"]["mnes"][0], [10.0])
    assert "oss" not in pairs["cls"]


def test_plot_loader_skips_and_reports_outdated_model(tmp_path, capsys) -> None:
    for name, model in (("current", 2), ("old", 1)):
        instance_dir = tmp_path / "cls" / name
        instance_dir.mkdir(parents=True)
        (instance_dir / f"{name}.data").write_text(json.dumps({
            "benchmark_model": model,
            "runtime_glpk": 1.0,
            "cycle_count_mnes": 10,
        }))
    never_benchmarked = tmp_path / "cls" / "solve_only"
    never_benchmarked.mkdir(parents=True)
    (never_benchmarked / "solve_only.data").write_text(json.dumps({
        "solve_status_std": "ok",
        "runtime_highs_std": 1.0,
    }))
    loaded = _load_advantage_data(["cls"], tmp_path, "runtime_glpk")
    assert len(loaded["cls"]) == 1
    assert "Skipped 1 outdated benchmark-model records." in capsys.readouterr().out


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
        "benchmark_model": 2,
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
        record["benchmark_model"] = 2
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
        "invalid_breakdown": 1,
        "unplottable_overflow": 0,
        "condition_number": 0,
        "min_runtime": 0,
    }
    assert eligible == 1


def test_ratio_cycle_time_must_be_positive_and_finite() -> None:
    for value in (0.0, -1.0, float("inf"), float("nan")):
        with pytest.raises(ValueError):
            _validate_cycle_time(value)


@pytest.mark.parametrize(
    ("estimate", "expected"),
    [("best-known", 20.0), ("floor", 5.0)],
)
def test_estimate_selects_matching_count_and_query_keys(
    estimate: str, expected: float, tmp_path
) -> None:
    instance_dir = tmp_path / "cls" / "item"
    instance_dir.mkdir(parents=True)
    (instance_dir / "item.data").write_text(json.dumps({
        "benchmark_model": 2,
        "runtime_highs_std": 2.0,
        "cycle_count_mnes": 100,
        "qlsa_queries_mnes": 10,
        "cycle_count_best_known_mnes": 40,
        "qlsa_queries_best_known_mnes": 4,
        "cycle_count_floor_mnes": 10,
        "qlsa_queries_floor_mnes": 1,
        "tomography_reps_mnes": 10,
    }))

    loaded = _load_advantage_data(
        ["cls"], tmp_path, "runtime_highs_std", estimate
    )
    pairs = _advantage_pairs(
        loaded, ["mnes"], "runtime_highs_std", estimate
    )
    np.testing.assert_array_equal(
        pairs["cls"]["mnes"][0],
        [40.0 if estimate == "best-known" else 10.0],
    )
    ratio, skipped, eligible = _load_ratio_data(
        ["cls"], tmp_path, "runtime_highs_std", 1.0, ["mnes"], estimate
    )
    np.testing.assert_array_equal(ratio["cls"]["mnes"][0], [expected])
    assert skipped["invalid_breakdown"] == 0
    assert eligible == 1


def test_plot_loaders_report_condition_and_runtime_exclusions(
    tmp_path, capsys
) -> None:
    records = {
        "ill_conditioned": {
            "runtime_highs_std": 1.0,
            "cycle_count_mnes": 100,
            "qlsa_queries_mnes": 10,
            "tomography_reps_mnes": 10,
            "sparsity_mnes": 2,
            "cond_mnes": 1e17,
            "cycle_count_oss": 200,
            "qlsa_queries_oss": 20,
            "tomography_reps_oss": 10,
            "cond_oss": 2.0,
        },
        "too_fast": {
            "runtime_highs_std": 0.005,
            "cycle_count_mnes": 100,
            "qlsa_queries_mnes": 10,
            "tomography_reps_mnes": 10,
            "sparsity_mnes": 2,
            "cond_mnes": 2.0,
        },
    }
    for name, record in records.items():
        record["benchmark_model"] = 2
        instance_dir = tmp_path / "cls" / name
        instance_dir.mkdir(parents=True)
        (instance_dir / f"{name}.data").write_text(json.dumps(record))

    advantage = _load_advantage_data(
        ["cls"], tmp_path, "runtime_highs_std", min_runtime=0.01
    )
    assert list(advantage["cls"]) == [records["ill_conditioned"]]
    assert set(
        _advantage_pairs(
            advantage, ["mnes", "oss"], "runtime_highs_std"
        )["cls"]
    ) == {"oss"}
    ratio, skipped, _ = _load_ratio_data(
        ["cls"], tmp_path, "runtime_highs_std", 1.0,
        ["mnes", "oss"], min_runtime=0.01,
    )
    assert set(ratio["cls"]) == {"oss"}
    assert skipped["condition_number"] == 1
    assert skipped["min_runtime"] == 1
    assert _load_difficulty_data(["cls"], tmp_path, "mnes") == {
        "cls": np.array([4.0])
    }
    output = capsys.readouterr().out
    assert "condition number > 1e+16" in output
    assert "records below the 0.01 s runtime floor" in output


@pytest.mark.parametrize("excluded_variant", ["mnes", "oss"])
def test_both_advantage_uses_common_condition_eligible_population(
    excluded_variant, tmp_path, monkeypatch
) -> None:
    records = {
        "eligible": {
            "record_id": "eligible",
            "runtime_highs_std": 1.0,
            "cycle_count_mnes": 100,
            "cond_mnes": 2.0,
            "cycle_count_oss": 200,
            "cond_oss": 3.0,
        },
        "condition_excluded": {
            "record_id": "condition_excluded",
            "runtime_highs_std": 1.0,
            "cycle_count_mnes": 100,
            "cond_mnes": 1e17 if excluded_variant == "mnes" else 2.0,
            "cycle_count_oss": 200,
            "cond_oss": 1e17 if excluded_variant == "oss" else 3.0,
        },
    }
    for name, record in records.items():
        record["benchmark_model"] = 2
        instance_dir = tmp_path / "cls" / name
        instance_dir.mkdir(parents=True)
        (instance_dir / f"{name}.data").write_text(json.dumps(record))

    seen_records = []
    real_advantage_pairs = plot_module._advantage_pairs

    def capture_pairs(data, variants, runtime_key, estimate="modeled"):
        seen_records.extend(data["cls"])
        return real_advantage_pairs(data, variants, runtime_key, estimate)

    monkeypatch.setattr(plot_module, "_advantage_pairs", capture_pairs)
    monkeypatch.setitem(plot_module._RCPARAMS, "text.usetex", False)
    output = tmp_path / "advantage.pdf"
    plot_module.plot_advantage(
        ["cls"], "both", tmp_path, output, runtime_key="runtime_highs_std"
    )

    assert output.is_file()
    assert [record["record_id"] for record in seen_records] == ["eligible"]


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
        record["benchmark_model"] = 2
        instance_dir = tmp_path / "cls" / name
        instance_dir.mkdir(parents=True)
        (instance_dir / f"{name}.data").write_text(json.dumps(record))

    data, skipped, eligible = _load_ratio_data(
        ["cls"], tmp_path, "runtime_highs_std", 1e-20, ["mnes"]
    )

    total, prep = data["cls"]["mnes"]
    assert len(total) == len(prep) == 1
    assert skipped["invalid_breakdown"] == 1
    assert eligible == 1


def test_huge_count_flows_through_advantage_and_ratio_loaders(tmp_path) -> None:
    huge_count = 10**1000
    instance_dir = tmp_path / "cls" / "huge"
    instance_dir.mkdir(parents=True)
    (instance_dir / "huge.data").write_text(json.dumps({
        "benchmark_model": 2,
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
        "benchmark_model": 2,
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
        "benchmark_model": 2,
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
        "benchmark_model": 2,
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
        "benchmark_model": 2,
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
    assert "invalid query/repetition breakdown" in output_text
    assert "overflow plotting" in output_text
    assert f"Saved to {output}" in output_text
