#!/usr/bin/env python3
"""Plot quantum advantage curves and difficulty (s·κ) histograms from benchmark data."""

from __future__ import annotations

import math
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

GATE_SPEED_RECORD = 8e-10  # seconds — update as needed
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
    numeric = float(value)
    return math.isfinite(numeric) and (numeric > 0 if positive else numeric >= 0)


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
                _finite_number(r.get(BENCHMARK_VALUE_KEYS["mnes"][0]), positive=True)
                or _finite_number(r.get(BENCHMARK_VALUE_KEYS["oss"][0]), positive=True)
            )
        ]
        if filtered:
            result[cls] = filtered
    return result


def _cycle_counts(records: list[dict], variant: str) -> np.ndarray | None:
    """Extract cycle counts for a single variant ('mnes' or 'oss')."""
    key = BENCHMARK_VALUE_KEYS[variant][0]
    vals = [r[key] for r in records if _finite_number(r.get(key), positive=True)]
    return np.array(vals, dtype=np.float64) if vals else None


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
            key = BENCHMARK_VALUE_KEYS[variant][0]
            pairs = [
                (record[key], record[runtime_key])
                for record in records
                if _finite_number(record.get(key), positive=True)
            ]
            if pairs:
                class_pairs[variant] = (
                    np.array([count for count, _ in pairs], dtype=np.float64),
                    np.array([runtime for _, runtime in pairs], dtype=np.float64),
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
    runtime_key: str = "runtime_glpk",
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
                if _finite_number(record.get(BENCHMARK_VALUE_KEYS["mnes"][0]), positive=True)
                and _finite_number(record.get(BENCHMARK_VALUE_KEYS["oss"][0]), positive=True)
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
    x_max = max(float(combined.max()), GATE_SPEED_RECORD) * 10
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

    ax.axvline(GATE_SPEED_RECORD, color="#444444", linestyle=":", linewidth=1.2)
    ax.text(
        GATE_SPEED_RECORD * 0.82, 50,
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
    _, sparsity_key, cond_key = BENCHMARK_VALUE_KEYS[variant]
    all_records = _iter_records(instance_classes, cache_dir)
    result: dict[str, np.ndarray] = {}
    for cls, records in all_records.items():
        values = []
        for r in records:
            s = r.get(sparsity_key)
            k = r.get(cond_key)
            if not _finite_number(s, positive=True) or not _finite_number(k, positive=True):
                continue
            product = float(s) * float(k)
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
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Plot quantum advantage curves or difficulty (s·κ) histograms.",
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
        default="glpk",
        help="Classical solver runtime to compare against (default: glpk). Ignored with --difficulty.",
    )
    parser.add_argument(
        "--difficulty",
        action="store_true",
        help="Plot s·κ difficulty histogram instead of advantage curves.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Cache directory (default: cache_dir in current directory).",
    )
    args = parser.parse_args()

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
    else:
        plot_advantage(
            instance_classes=classes,
            variant=args.variant,
            cache_dir=cache_dir,
            output=Path(f"plot_advantage_{classes_tag}_{args.solver}_{args.variant}.pdf"),
            runtime_key=RUNTIME_KEYS[args.solver],
        )
