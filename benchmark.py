#!/usr/bin/env python3
"""Benchmark LP instances and store model-2 QIPM screening estimates."""

from __future__ import annotations

import math
from pathlib import Path

from bounds import (
    CycleCountResult,
    DegenerateInstanceError,
    InconsistentSystemError,
    PreprocessCrashedError,
    PreprocessTimeoutError,
    RankUncertainError,
    _cycle_count_mnes,
    _cycle_count_mnes_from_basis,
    _cycle_count_oss,
    _cycle_count_oss_from_basis,
    _preprocess_basis,
)
from standard_form import load_standard_form
from store import (
    BENCHMARK_MODEL,
    BENCHMARK_MODEL_KEY,
    BENCHMARK_RESULT_KEYS,
    BENCHMARK_STATUS_KEYS,
    BENCHMARK_VALUE_KEYS,
    VARIANTS,
    atomic_write_json,
    list_class_names,
    list_instance_dirs,
    process_instance_dirs,
    purge_keys_from_ledgers,
    read_ledger,
    resolve_cache_root,
    summarize_records,
)


def _benchmark_instance_from_path(
    path: Path,
    variant: str = "both",
) -> None:
    """Load one .std instance, compute estimates, and update its .data ledger."""
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Instance file not found: {path}")
    if path.suffix.lower() != ".std":
        raise ValueError(f"Path must be .std; got {path.suffix!r}")
    if variant not in ("mnes", "oss", "both"):
        raise ValueError(f"variant must be 'mnes', 'oss', or 'both'; got {variant!r}")

    base_name = path.name[: -len(".std")]
    data_path = path.parent / (base_name + ".data")
    data, _ = read_ledger(data_path)
    active_variants = list(VARIANTS) if variant == "both" else [variant]

    # A record-level marker cannot distinguish an old inactive variant.  When
    # upgrading a ledger, discard all model-1 benchmark fields before writing
    # model 2, even if this invocation computes only one variant.
    if data.get(BENCHMARK_MODEL_KEY) != BENCHMARK_MODEL:
        for stale_variant in VARIANTS:
            for key in BENCHMARK_RESULT_KEYS[stale_variant]:
                data.pop(key, None)
    data[BENCHMARK_MODEL_KEY] = BENCHMARK_MODEL

    def _skip(status: str) -> None:
        for active in active_variants:
            for key in BENCHMARK_VALUE_KEYS[active]:
                data.pop(key, None)
            data[BENCHMARK_STATUS_KEYS[active]] = status
        atomic_write_json(data_path, data)

    try:
        _, b, A, _ = load_standard_form(path)
    except Exception as exc:  # noqa: BLE001 - persist corrupt-input status
        _skip(f"error:{type(exc).__name__}")
        return

    if A.shape[0] < 2 or A.shape[1] < 2:
        _skip("skipped_degenerate")
        return
    if A.shape[0] > 100_000:
        _skip("skipped_too_large")
        return

    def _failure_status(exc: BaseException) -> str:
        if isinstance(exc, PreprocessTimeoutError):
            return "timeout"
        if isinstance(exc, PreprocessCrashedError):
            return "crashed"
        if isinstance(exc, RankUncertainError):
            return "rank_uncertain"
        if isinstance(exc, InconsistentSystemError):
            return "inconsistent_rows"
        if isinstance(exc, DegenerateInstanceError):
            return "skipped_degenerate"
        return f"error:{type(exc).__name__}"

    def _record_failure(active: str, exc: BaseException) -> None:
        failure_status = _failure_status(exc)
        for key in BENCHMARK_VALUE_KEYS[active]:
            if failure_status == "skipped_degenerate":
                data.pop(key, None)
            else:
                data[key] = None
        data[BENCHMARK_STATUS_KEYS[active]] = failure_status

    def _record_success(active: str, result: CycleCountResult) -> None:
        cond = max(float(result.cond), 1.0)
        if not math.isfinite(cond):
            raise ArithmeticError("condition number is not finite")
        keys = BENCHMARK_VALUE_KEYS[active]
        data[keys.count] = result.count
        data[keys.count_best_known] = result.count_best_known
        data[keys.count_floor] = result.count_floor
        data[keys.sparsity] = result.sparsity
        data[keys.sparsity_method] = result.sparsity_method
        data[keys.cond] = cond
        data[keys.cond_method] = result.cond_method
        data[keys.qlsa_queries] = result.qlsa_queries
        data[keys.qlsa_queries_best_known] = result.qlsa_queries_best_known
        data[keys.qlsa_queries_floor] = result.qlsa_queries_floor
        data[keys.tomography_reps] = result.repetitions
        data[BENCHMARK_STATUS_KEYS[active]] = "ok"

    if variant == "both":
        try:
            basis = _preprocess_basis(A, b)
        except (RuntimeError, ValueError, ArithmeticError) as exc:
            _record_failure("mnes", exc)
            _record_failure("oss", exc)
        else:
            if basis.m < 2:
                _skip("skipped_degenerate")
                return
            try:
                _record_success("mnes", _cycle_count_mnes_from_basis(basis))
            except (RuntimeError, ValueError, ArithmeticError) as exc:
                _record_failure("mnes", exc)
            try:
                _record_success("oss", _cycle_count_oss_from_basis(basis))
            except (RuntimeError, ValueError, ArithmeticError) as exc:
                _record_failure("oss", exc)
    else:
        calculate = _cycle_count_mnes if variant == "mnes" else _cycle_count_oss
        try:
            _record_success(variant, calculate(A, b))
        except (RuntimeError, ValueError, ArithmeticError) as exc:
            _record_failure(variant, exc)

    atomic_write_json(data_path, data)


def benchmark_instance(
    instance_class: str,
    instance_name: str,
    cache_dir: str | Path | None = None,
    variant: str = "both",
) -> None:
    """Benchmark one instance discovered under cache_dir by its sole .std file."""
    root = resolve_cache_root(cache_dir)
    instance_dir = root / instance_class / instance_name
    if not instance_dir.is_dir():
        raise FileNotFoundError(f"Instance directory not found: {instance_dir}")
    std_files = sorted(instance_dir.glob("*.std"))
    if len(std_files) != 1:
        raise FileNotFoundError(
            f"Expected exactly one .std in {instance_dir}; found {len(std_files)}"
        )
    _benchmark_instance_from_path(std_files[0], variant=variant)


def benchmark_instance_class(
    instance_class: str,
    variant: str = "both",
    cache_dir: str | Path | None = None,
) -> None:
    """Benchmark all .std instances in one cache class."""
    root = resolve_cache_root(cache_dir)
    folder = root / instance_class
    if not folder.is_dir():
        raise FileNotFoundError(f"Instance class folder not found: {folder}")

    process_instance_dirs(
        instance_class,
        list_instance_dirs(folder),
        lambda subdir: benchmark_instance(
            instance_class, subdir.name, cache_dir=root, variant=variant
        ),
        required_glob="*.std",
        missing_message="no .std file",
    )


def benchmark_all_instance_classes(
    instance_classes: list[str] | None = None,
    variant: str = "both",
    cache_dir: str | Path | None = None,
) -> None:
    """Run the benchmark stage for selected or all cache classes."""
    root = resolve_cache_root(cache_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Cache directory not found: {root}")
    if instance_classes is None:
        instance_classes = list_class_names(root)

    for name in instance_classes:
        benchmark_instance_class(name, variant=variant, cache_dir=root)
        show_benchmark_status([name], variant=variant, cache_dir=root)


def show_benchmark_status(
    instance_classes: list[str] | None = None,
    variant: str = "both",
    cache_dir: str | Path | None = None,
) -> None:
    """Print current model-2 benchmark counts and non-ok statuses by class."""
    root = resolve_cache_root(cache_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Cache directory not found: {root}")
    if instance_classes is None:
        instance_classes = list_class_names(root)
    active_variants = list(VARIANTS) if variant == "both" else [variant]

    for cls in instance_classes:
        folder = root / cls
        if not folder.is_dir():
            print(f"{cls}: directory not found")
            continue
        subdirs = list_instance_dirs(folder)
        total = len(subdirs)
        summaries = {
            active: summarize_records(
                subdirs,
                value_key=BENCHMARK_VALUE_KEYS[active].count,
                status_key=BENCHMARK_STATUS_KEYS[active],
                ok_statuses=("ok",),
                required_model=BENCHMARK_MODEL,
            )
            for active in active_variants
        }

        parts = []
        for active in active_variants:
            count, non_ok = summaries[active]
            breakdown = ", ".join(
                f"{status}: {number}" for status, number in sorted(non_ok.items())
            )
            suffix = f" ({breakdown})" if breakdown else ""
            parts.append(f"{active}: {count}/{total}{suffix}")
        print(f"{cls}:  {'  '.join(parts)}")


def clear_benchmark_data(
    instance_classes: list[str] | None = None,
    cache_dir: str | Path | None = None,
    variant: str = "both",
) -> None:
    """Remove benchmark values and statuses from .data files."""
    root = resolve_cache_root(cache_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Cache directory not found: {root}")

    active_variants = VARIANTS if variant == "both" else (variant,)
    keys = tuple(key for active in active_variants for key in BENCHMARK_RESULT_KEYS[active])
    if variant == "both":
        keys += (BENCHMARK_MODEL_KEY,)
    search_roots = [root / name for name in instance_classes] if instance_classes else [root]
    purge_keys_from_ledgers(search_roots, keys)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Benchmark LP instances with model-2 QIPM screening estimates.",
    )
    parser.add_argument(
        "instance_classes",
        nargs="*",
        help="Instance class names (subfolders under cache_dir). If none given, process all.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Cache directory (default: cache_dir in current directory).",
    )
    parser.add_argument(
        "--variant",
        choices=["mnes", "oss", "both"],
        default="both",
        help="Which QIPM variant to run: 'mnes', 'oss', or 'both' (default).",
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Remove benchmark entries from .data files instead of benchmarking.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show current benchmark records and non-success statuses.",
    )
    args = parser.parse_args()
    if args.show:
        show_benchmark_status(
            instance_classes=args.instance_classes or None,
            variant=args.variant,
            cache_dir=args.cache_dir,
        )
    elif args.clear:
        clear_benchmark_data(
            instance_classes=args.instance_classes or None,
            cache_dir=args.cache_dir,
            variant=args.variant,
        )
    else:
        benchmark_all_instance_classes(
            instance_classes=args.instance_classes or None,
            variant=args.variant,
            cache_dir=args.cache_dir,
        )
