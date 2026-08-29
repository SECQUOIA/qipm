#!/usr/bin/env python3
"""Plot quantum advantage, fixed-cycle ratios, and difficulty from benchmark data."""

from __future__ import annotations

import math
from fractions import Fraction
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches

from store import (
    BENCHMARK_VALUE_KEYS,
    SOLVE_RESULT_KEYS,
    VARIANTS,
    list_class_names,
    list_instance_dirs,
    read_ledger,
    resolve_cache_root,
)

# 800 ps √SWAP two-qubit gate, He et al., Nature 571, 371 (2019); Sec. V
# of the paper uses it as an optimistic physical proxy for a logical cycle.
DEFAULT_CYCLE_DURATION = 8e-10
N_POINTS = 500
N_BINS = 30

CLASS_LABELS = {
    "independent_set": "Independent Set",
    "clique":          "Clique",
    "vertex_cover":    "Vertex Cover",
    "max_flow":        "Max Flow",
    "netlib":          "Netlib",
    "miplib":          "MIPlib",
    "stochlp":         "StochLP",
    "misc":            "Misc",
}
CLASS_COLORS = {
    "independent_set": "#E8A87C",
    "clique":          "#6B8FA8",
    "vertex_cover":    "#7AAA7A",
    "max_flow":        "#C97B7B",
    "netlib":          "#9080B8",
    "miplib":          "#A0A0A0",
    "stochlp":         "#E8A8C8",
    "misc":            "#B5A882",
}

RUNTIME_KEYS = {
    "glpk":      "runtime_glpk",
    "highs-std": SOLVE_RESULT_KEYS["std"][0],
    "highs-mps": SOLVE_RESULT_KEYS["mps"][0],
}

_RCPARAMS = {
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "text.usetex": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
}


def _finite_number(value: object, *, positive: bool = False) -> bool:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    if isinstance(value, int):
        return value > 0 if positive else value >= 0
    numeric = float(value)
    return math.isfinite(numeric) and (numeric > 0 if positive else numeric >= 0)


def _as_float(value: object) -> float:
    try:
        return float(value)
    except OverflowError:
        return math.inf


def _difficulty_bins(values: np.ndarray) -> np.ndarray:
    positive = values[np.isfinite(values) & (values > 0)]
    value_min = float(positive.min())
    value_max = float(positive.max())
    if value_min == value_max:
        value_min /= 1.1
        value_max *= 1.1
    return np.logspace(np.log10(value_min), np.log10(value_max), N_BINS + 1)


# ---------------------------------------------------------------------------
# Shared data loading
# ---------------------------------------------------------------------------

def _iter_records(instance_classes: list[str], cache_dir: Path) -> dict[str, list[dict]]:
    """Load all .data JSON files, grouped by class."""
    result: dict[str, list[dict]] = {}
    for cls in instance_classes:
        cls_dir = cache_dir / cls
        if not cls_dir.is_dir():
            continue
        records = []
        for instance_dir in list_instance_dirs(cls_dir):
            data_path = instance_dir / (instance_dir.name + ".data")
            if not data_path.exists():
                continue
            try:
                record, state = read_ledger(data_path)
            except OSError:
                continue
            if state == "valid":
                records.append(record)
        if records:
            result[cls] = records
    return result


# ---------------------------------------------------------------------------
# Advantage plot
# ---------------------------------------------------------------------------

def _load_advantage_data(
    instance_classes: list[str],
    cache_dir: Path,
    runtime_key: str,
) -> dict[str, list[dict]]:
    """Filter records to those with a valid runtime and at least one cycle count."""
    all_records = _iter_records(instance_classes, cache_dir)
    result = {}
    for cls, records in all_records.items():
        filtered = [
            r for r in records
            if _finite_number(r.get(runtime_key))
            and (
                _finite_number(r.get(BENCHMARK_VALUE_KEYS["mnes"].count), positive=True)
                or _finite_number(r.get(BENCHMARK_VALUE_KEYS["oss"].count), positive=True)
            )
        ]
        if filtered:
            result[cls] = filtered
    return result


def _cycle_counts(records: list[dict], variant: str) -> np.ndarray | None:
    """Extract cycle counts for a single variant ('mnes' or 'oss')."""
    key = BENCHMARK_VALUE_KEYS[variant].count
    vals = [r[key] for r in records if _finite_number(r.get(key), positive=True)]
    if not vals:
        return None
    return np.array([_as_float(value) for value in vals], dtype=np.float64)


def _crossover_times(cycle_counts: np.ndarray, runtimes: np.ndarray) -> np.ndarray:
    return runtimes / cycle_counts


def _advantage_pairs(
    data: dict[str, list[dict]], variants: list[str], runtime_key: str
) -> dict[str, dict[str, tuple[np.ndarray, np.ndarray]]]:
    """Build aligned cycle-count/runtime arrays once for every class and variant."""
    result: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
    for cls, records in data.items():
        class_pairs = {}
        for variant in variants:
            key = BENCHMARK_VALUE_KEYS[variant].count
            pairs = [
                (record[key], record[runtime_key])
                for record in records
                if _finite_number(record.get(key), positive=True)
            ]
            if pairs:
                class_pairs[variant] = (
                    np.array(
                        [_as_float(count) for count, _ in pairs], dtype=np.float64
                    ),
                    np.array(
                        [_as_float(runtime) for _, runtime in pairs], dtype=np.float64
                    ),
                )
        if class_pairs:
            result[cls] = class_pairs
    return result


def _advantage_curve(ct: np.ndarray, t_values: np.ndarray) -> np.ndarray:
    return np.array([100.0 * np.mean(t < ct) for t in t_values])


def _truncate_at_zero(
    t_values: np.ndarray, curve: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    zero_idx = np.argmax(curve == 0.0)
    if curve[zero_idx] == 0.0:
        end = max(2, zero_idx + 1)
        return t_values[:end], curve[:end]
    return t_values, curve


def plot_advantage(
    instance_classes: list[str],
    variant: str,
    cache_dir: Path,
    output: Path,
    runtime_key: str = SOLVE_RESULT_KEYS["std"][0],
) -> None:
    """Plot advantage curves. variant: 'mnes', 'oss', or 'both'."""
    data = _load_advantage_data(instance_classes, cache_dir, runtime_key)
    if not data:
        print("No data found.")
        return

    variants = list(VARIANTS) if variant == "both" else [variant]
    if variant == "both":
        complete = {}
        for cls, records in data.items():
            selected = [
                record
                for record in records
                if _finite_number(record.get(BENCHMARK_VALUE_KEYS["mnes"].count), positive=True)
                and _finite_number(record.get(BENCHMARK_VALUE_KEYS["oss"].count), positive=True)
            ]
            if selected:
                complete[cls] = selected
        data = complete

    pair_data = _advantage_pairs(data, variants, runtime_key)
    all_cts = [
        _crossover_times(counts, runtimes)
        for class_pairs in pair_data.values()
        for counts, runtimes in class_pairs.values()
    ]

    if not all_cts:
        print("No valid data for the requested variant.")
        return

    combined = np.concatenate(all_cts)
    combined = combined[np.isfinite(combined) & (combined >= 0)]
    if combined.size == 0:
        print("No finite crossover times for the requested variant.")
        return
    positive = combined[combined > 0]
    x_min = max(1e-28, float(positive.min())) if positive.size else 1e-28
    x_max = max(float(combined.max()), DEFAULT_CYCLE_DURATION) * 10
    t_values = np.geomspace(x_min, x_max, N_POINTS)

    plt.rcParams.update({
        **_RCPARAMS,
        "axes.grid": True,
        "grid.color": "#E0E0E0",
        "grid.linewidth": 0.8,
        "axes.facecolor": "#FAFAFA",
        "figure.facecolor": "white",
    })

    fig, ax = plt.subplots(figsize=(10, 5))
    fallback_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    sorted_classes = sorted(data.keys(), key=str.lower)
    class_colors = {
        cls: CLASS_COLORS.get(cls, fallback_colors[i % len(fallback_colors)])
        for i, cls in enumerate(sorted_classes)
    }

    linestyles = {"mnes": "-", "oss": "--"}

    for cls, class_pairs in pair_data.items():
        color = class_colors[cls]
        for v in variants:
            if v not in class_pairs:
                continue
            gc, hr = class_pairs[v]
            ct = _crossover_times(gc, hr)
            curve = _advantage_curve(ct, t_values)
            tv, cv = _truncate_at_zero(t_values, curve)
            ax.plot(tv, cv, color=color, linestyle=linestyles[v], linewidth=1.8)

    ax.axvline(DEFAULT_CYCLE_DURATION, color="#444444", linestyle=":", linewidth=1.2)
    ax.text(
        DEFAULT_CYCLE_DURATION * 0.82, 50,
        "current speed record for\n an entangling gate operation",
        ha="right", va="center", fontsize=8.5, rotation=90, color="#444444",
    )

    ax.set_xscale("log")
    ax.set_xlim(x_min, x_max)
    ax.set_xlabel("quantum cycle duration ($s$)", fontsize=11, labelpad=8)
    ax.set_ylabel(r"instances with quantum advantage (\%)", fontsize=11, labelpad=8)
    ax.set_ylim(0, 102)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.tick_params(axis="both", labelsize=9.5)
    ax.spines["left"].set_color("#CCCCCC")
    ax.spines["bottom"].set_color("#CCCCCC")

    class_handles = [
        mpatches.Patch(
            facecolor=class_colors[cls],
            edgecolor="none",
            label=CLASS_LABELS.get(cls, cls),
        )
        for cls in sorted_classes
    ]
    fig.legend(
        handles=class_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=min(len(class_handles), 4),
        frameon=True,
        framealpha=0.95,
        edgecolor="#CCCCCC",
        fontsize=9,
    )

    if variant == "both":
        h1 = mlines.Line2D([], [], color="black", linestyle="-",  linewidth=1.8, label="QIPM (MNES)")
        h2 = mlines.Line2D([], [], color="black", linestyle="--", linewidth=1.8, label="QIPM (OSS)")
        ax.legend(handles=[h1, h2], loc="lower left", fontsize=9,
                  framealpha=0.95, edgecolor="#CCCCCC")

    fig.tight_layout()
    fig.subplots_adjust(top=0.78)
    fig.savefig(output, dpi=150)
    plt.close(fig)
    print(f"Saved to {output}")


# ---------------------------------------------------------------------------
# Difficulty plot
# ---------------------------------------------------------------------------

def _load_difficulty_data(
    instance_classes: list[str],
    cache_dir: Path,
    variant: str,
) -> dict[str, np.ndarray]:
    """Load s·κ products for each class for the given variant ('mnes' or 'oss')."""
    keys = BENCHMARK_VALUE_KEYS[variant]
    all_records = _iter_records(instance_classes, cache_dir)
    result: dict[str, np.ndarray] = {}
    for cls, records in all_records.items():
        values = []
        for r in records:
            s = r.get(keys.sparsity)
            k = r.get(keys.cond)
            if not _finite_number(s, positive=True) or not _finite_number(k, positive=True):
                continue
            product = _as_float(s) * _as_float(k)
            if math.isfinite(product) and product > 0:
                values.append(product)
        if values:
            result[cls] = np.array(values, dtype=np.float64)
    return result


def _build_difficulty_histogram(
    data: dict[str, np.ndarray],
) -> tuple[np.ndarray, list[str], list[np.ndarray], np.ndarray]:
    """Build shared bins, class order, values, and stacked counts."""
    all_values = np.concatenate(list(data.values()))
    bins = _difficulty_bins(all_values)
    classes_sorted = sorted(data.keys(), key=lambda cls: float(np.median(data[cls])))
    values = [data[cls] for cls in classes_sorted]
    stacked = np.zeros(N_BINS, dtype=np.float64)
    for class_values in values:
        counts, _ = np.histogram(class_values, bins=bins)
        stacked += counts
    return bins, classes_sorted, values, stacked


def plot_difficulty(
    instance_classes: list[str],
    variant: str,
    cache_dir: Path,
    output: Path,
    y_max: float | None = None,
) -> None:
    """Plot stacked s·κ histogram for one variant ('mnes' or 'oss')."""
    data = _load_difficulty_data(instance_classes, cache_dir, variant)
    if not data:
        print(f"No difficulty data found for {variant}; skipping.")
        return

    bins, classes_sorted, histogram_values, _ = _build_difficulty_histogram(data)
    lower, upper = float(bins[0]), float(bins[-1])

    plt.rcParams.update(_RCPARAMS)

    fig, ax = plt.subplots(figsize=(6, 4))

    ax.hist(
        histogram_values,
        bins=bins,
        stacked=True,
        color=[CLASS_COLORS.get(cls, "#888888") for cls in classes_sorted],
        label=[CLASS_LABELS.get(cls, cls) for cls in classes_sorted],
        edgecolor="white",
        linewidth=0.4,
    )

    ax.set_xscale("log")
    ax.set_xlim(lower, upper)
    if y_max is not None:
        ax.set_ylim(top=y_max)
    ax.set_xlabel(r"difficulty $\gamma = s \cdot \kappa$", fontsize=11, labelpad=8)
    ax.set_ylabel("number of instances", fontsize=11, labelpad=8)
    ax.set_title(variant.upper(), fontsize=12, pad=10)
    handles, labels = ax.get_legend_handles_labels()
    sorted_pairs = sorted(zip(labels, handles), key=lambda x: x[0].lower())
    sorted_labels, sorted_handles = zip(*sorted_pairs) if sorted_pairs else ([], [])
    ax.legend(sorted_handles, sorted_labels, loc="upper left", fontsize=8, framealpha=0.95, edgecolor="#CCCCCC")

    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)
    print(f"Saved to {output}")


# ---------------------------------------------------------------------------
# Fixed-cycle-time ratio plot
# ---------------------------------------------------------------------------

def _validate_cycle_time(value: float) -> float:
    cycle_time = float(value)
    if not math.isfinite(cycle_time) or cycle_time <= 0:
        raise ValueError("cycle time must be positive and finite")
    return cycle_time


def _stored_integer(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if not isinstance(value, float) or not math.isfinite(value) or not value.is_integer():
        return None
    return int(value)


def _positive_stored_number(value: object) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return value > 0
    return isinstance(value, float) and math.isfinite(value) and value > 0


def _load_ratio_data(
    instance_classes: list[str],
    cache_dir: Path,
    runtime_key: str,
    cycle_time: float,
    variants: list[str] | None = None,
) -> tuple[
    dict[str, dict[str, tuple[np.ndarray, np.ndarray]]],
    dict[str, int],
    int,
]:
    """Load aligned (total, one-preparation) ratios and skip statistics."""
    cycle_time = _validate_cycle_time(cycle_time)
    variants = list(VARIANTS) if variants is None else variants
    all_records = _iter_records(instance_classes, cache_dir)
    loaded: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
    skipped = {
        "missing_runtime": 0,
        "invalid_runtime": 0,
        "needs_refresh": 0,
        "unplottable_overflow": 0,
    }
    eligible_instances = 0

    for cls, records in all_records.items():
        class_values: dict[str, tuple[list[float], list[float]]] = {
            variant: ([], []) for variant in variants
        }
        for record in records:
            eligible_variants = [
                variant
                for variant in variants
                if _positive_stored_number(
                    record.get(BENCHMARK_VALUE_KEYS[variant].count)
                )
            ]
            if not eligible_variants:
                continue
            if runtime_key not in record:
                skipped["missing_runtime"] += 1
                continue
            runtime = record[runtime_key]
            if not _positive_stored_number(runtime):
                skipped["invalid_runtime"] += 1
                continue
            record_eligible = False
            for variant in eligible_variants:
                keys = BENCHMARK_VALUE_KEYS[variant]
                raw_count = record.get(keys.count)
                count = _stored_integer(raw_count)
                queries = _stored_integer(record.get(keys.qlsa_queries))
                repetitions = _stored_integer(record.get(keys.tomography_reps))
                if (
                    count is None
                    or queries is None
                    or repetitions is None
                    or count <= 0
                    or queries <= 0
                    or repetitions <= 0
                    or count != queries * repetitions
                ):
                    skipped["needs_refresh"] += 1
                    continue
                try:
                    total_ratio = cycle_time * count / runtime
                    prep_ratio = cycle_time * queries / runtime
                except OverflowError:
                    try:
                        total_ratio = float(
                            Fraction(count) * Fraction(cycle_time) / Fraction(runtime)
                        )
                        prep_ratio = float(
                            Fraction(queries) * Fraction(cycle_time) / Fraction(runtime)
                        )
                    except OverflowError:
                        skipped["unplottable_overflow"] += 1
                        continue
                except (TypeError, ZeroDivisionError):
                    continue
                if not math.isfinite(total_ratio) or not math.isfinite(prep_ratio):
                    skipped["unplottable_overflow"] += 1
                    continue
                total_values, prep_values = class_values[variant]
                total_values.append(total_ratio)
                prep_values.append(prep_ratio)
                record_eligible = True
            if record_eligible:
                eligible_instances += 1

        class_data = {
            variant: (
                np.array(total_values, dtype=np.float64),
                np.array(prep_values, dtype=np.float64),
            )
            for variant, (total_values, prep_values) in class_values.items()
            if total_values
        }
        if class_data:
            loaded[cls] = class_data
    return loaded, skipped, eligible_instances


def _ecdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ordered = np.sort(values)
    start = np.nextafter(ordered[0], 0.0)
    if start <= 0:
        start = ordered[0]
    x_values = np.concatenate(([start], ordered))
    percentages = 100.0 * np.arange(len(ordered) + 1) / len(ordered)
    return x_values, percentages


def _cycle_duration_label(seconds: float) -> str:
    if seconds < 1e-9:
        return f"{seconds * 1e12:g} ps"
    if seconds < 1e-6:
        return f"{seconds * 1e9:g} ns"
    return f"{seconds:g} s"


def plot_ratio(
    instance_classes: list[str],
    cache_dir: Path,
    output: Path,
    *,
    runtime_key: str,
    baseline_label: str,
    cycle_time: float = DEFAULT_CYCLE_DURATION,
    style: str = "box",
    variant: str = "both",
) -> None:
    """Plot fixed-cycle quantum/classical runtime ratios."""
    if style not in ("box", "ecdf"):
        raise ValueError("ratio style must be 'box' or 'ecdf'")
    variants = list(VARIANTS) if variant == "both" else [variant]
    data, skipped, eligible = _load_ratio_data(
        instance_classes, cache_dir, runtime_key, cycle_time, variants
    )
    print(
        f"Ratio data: {skipped['missing_runtime']} records missing the runtime key; "
        f"{skipped['invalid_runtime']} records with non-positive or non-finite runtime; "
        f"{skipped['needs_refresh']} variant records need "
        "benchmark.py --refresh-counts; "
        f"{skipped['unplottable_overflow']} variant records overflow plotting."
    )
    if not data:
        print("No ratio data found.")
        return

    plt.rcParams.update(_RCPARAMS)
    fig, axes = plt.subplots(
        1, len(variants), figsize=(6 * len(variants), 4.6), sharey=True, squeeze=False
    )
    fallback_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    classes = sorted(data, key=str.lower)
    class_colors = {
        cls: CLASS_COLORS.get(cls, fallback_colors[i % len(fallback_colors)])
        for i, cls in enumerate(classes)
    }

    for ax, active in zip(axes[0], variants):
        available = [cls for cls in classes if active in data[cls]]
        if not available:
            ax.text(0.5, 0.5, "No eligible data", transform=ax.transAxes, ha="center")
            ax.set_title(active.upper())
            continue
        if style == "box":
            positions = np.arange(len(available), dtype=float)
            total = [data[cls][active][0] for cls in available]
            prep = [data[cls][active][1] for cls in available]
            total_boxes = ax.boxplot(
                total, positions=positions - 0.18, widths=0.32, patch_artist=True,
                manage_ticks=False,
                flierprops=dict(
                    marker=".", markersize=2, markerfacecolor="#555555",
                    markeredgecolor="none", alpha=0.35,
                ),
            )
            prep_boxes = ax.boxplot(
                prep, positions=positions + 0.18, widths=0.32, patch_artist=True,
                manage_ticks=False,
                flierprops=dict(
                    marker=".", markersize=2, markerfacecolor="#555555",
                    markeredgecolor="none", alpha=0.35,
                ),
            )
            for patch, cls in zip(total_boxes["boxes"], available):
                patch.set_facecolor(class_colors[cls])
            for patch, cls in zip(prep_boxes["boxes"], available):
                patch.set_facecolor(class_colors[cls])
                patch.set_alpha(0.3)
                patch.set_hatch("///")
            ax.set_xticks(positions)
            ax.set_xticklabels(
                [CLASS_LABELS.get(cls, cls) for cls in available], rotation=25, ha="right"
            )
            ax.set_yscale("log")
            ax.set_xlabel("instance class")
        else:
            total = np.concatenate([data[cls][active][0] for cls in available])
            prep = np.concatenate([data[cls][active][1] for cls in available])
            ax.step(
                *_ecdf(total), where="post", color="#333333", linewidth=1.8,
                linestyle="-",
            )
            ax.step(
                *_ecdf(prep), where="post", color="#333333", linewidth=1.8,
                linestyle="--",
            )
            ax.set_xscale("log")
            ax.set_ylim(0, 100)
            ax.set_xlabel(r"quantum/classical runtime ratio $\rho$")
        if style == "box":
            ax.axhline(1.0, color="#555555", linestyle="--", linewidth=1.0)
            ax.text(0.02, 1.0, "parity", transform=ax.get_yaxis_transform(), va="bottom")
        else:
            ax.axvline(1.0, color="#555555", linestyle="--", linewidth=1.0)
            ax.text(1.0, 0.03, "parity", transform=ax.get_xaxis_transform(), ha="right", rotation=90)
        ax.set_title(active.upper())
        ax.grid(True, which="both", color="#E0E0E0", linewidth=0.6)

    axes[0][0].set_ylabel(
        r"quantum/classical runtime ratio $\rho$" if style == "box"
        else r"\% of instances with ratio $\leq x$"
    )
    series_handles = [
        mpatches.Patch(facecolor="#777777", label=r"total ($Q\times R$)"),
        mpatches.Patch(
            facecolor="#BBBBBB", hatch="///", alpha=0.4,
            label="one QLSA state preparation (no readout)",
        ),
    ] if style == "box" else [
        mlines.Line2D([], [], color="#333333", linestyle="-", label=r"total ($Q\times R$)"),
        mlines.Line2D(
            [], [], color="#333333", linestyle="--",
            label="one QLSA state preparation (no readout)",
        ),
    ]
    fig.suptitle("Fixed-cycle quantum/classical runtime ratio", y=0.98)
    fig.legend(
        handles=series_handles, loc="upper center", bbox_to_anchor=(0.5, 0.92),
        ncol=2, frameon=False,
    )
    fig.text(
        0.5, 0.01,
        f"Cycle duration: {_cycle_duration_label(cycle_time)}; classical baseline: "
        f"{baseline_label}; eligible instances: {eligible}",
        ha="center", fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.82))
    fig.savefig(output, dpi=150)
    plt.close(fig)
    print(f"Saved to {output}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Plot quantum advantage curves, fixed-cycle ratios, or difficulty histograms.",
    )
    parser.add_argument(
        "instance_classes",
        nargs="*",
        help="Instance class names (subfolders under cache_dir). If none given, process all.",
    )
    parser.add_argument(
        "--variant",
        choices=["mnes", "oss", "both"],
        default="both",
        help="Which QIPM variant(s) to include (default: both).",
    )
    parser.add_argument(
        "--solver",
        choices=list(RUNTIME_KEYS),
        default="highs-std",
        help="Classical solver runtime to compare against (default: highs-std). Ignored with --difficulty.",
    )
    parser.add_argument(
        "--difficulty",
        action="store_true",
        help="Plot s·κ difficulty histogram instead of advantage curves.",
    )
    parser.add_argument(
        "--ratio",
        action="store_true",
        help="Plot fixed-cycle quantum/classical runtime ratios instead of advantage curves.",
    )
    parser.add_argument(
        "--cycle-time",
        type=float,
        default=DEFAULT_CYCLE_DURATION,
        help="Assumed quantum cycle duration in seconds for --ratio (default: 8e-10).",
    )
    parser.add_argument(
        "--ratio-style",
        choices=("box", "ecdf"),
        default="box",
        help="Ratio plot style (default: box).",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Cache directory (default: cache_dir in current directory).",
    )
    args = parser.parse_args()
    if args.ratio and args.difficulty:
        parser.error("--ratio and --difficulty are mutually exclusive")
    if args.ratio:
        try:
            args.cycle_time = _validate_cycle_time(args.cycle_time)
        except ValueError as exc:
            parser.error(str(exc))

    cache_dir = resolve_cache_root(args.cache_dir)

    if args.instance_classes:
        classes = args.instance_classes
        classes_tag = "-".join(classes)
    else:
        classes = list_class_names(cache_dir) if cache_dir.is_dir() else []
        classes_tag = "all"

    if args.difficulty:
        variants = list(VARIANTS) if args.variant == "both" else [args.variant]
        if len(variants) > 1:
            peak_counts = []
            for v in variants:
                vdata = _load_difficulty_data(classes, cache_dir, v)
                if not vdata:
                    continue
                _, _, _, stacked = _build_difficulty_histogram(vdata)
                peak_counts.append(int(stacked.max()))
            y_max: float | None = max(peak_counts) * 1.1 if peak_counts else None
        else:
            y_max = None
        for variant in variants:
            plot_difficulty(
                instance_classes=classes,
                variant=variant,
                cache_dir=cache_dir,
                output=Path(f"plot_difficulty_{classes_tag}_{variant}.pdf"),
                y_max=y_max,
            )
    elif args.ratio:
        plot_ratio(
            instance_classes=classes,
            cache_dir=cache_dir,
            output=Path(
                f"plot_ratio_{classes_tag}_{args.solver}_{args.variant}_{args.ratio_style}.pdf"
            ),
            runtime_key=RUNTIME_KEYS[args.solver],
            baseline_label=args.solver,
            cycle_time=args.cycle_time,
            style=args.ratio_style,
            variant=args.variant,
        )
    else:
        plot_advantage(
            instance_classes=classes,
            variant=args.variant,
            cache_dir=cache_dir,
            output=Path(f"plot_advantage_{classes_tag}_{args.solver}_{args.variant}.pdf"),
            runtime_key=RUNTIME_KEYS[args.solver],
        )
