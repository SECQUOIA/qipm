#!/usr/bin/env python3
"""Solve LP instances (MPS or standard-form .std) with HiGHS and record solve time."""

from __future__ import annotations

import os

# Effective when solve.py starts the process before NumPy is imported elsewhere.
for _thread_env_var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_thread_env_var, "1")
del _thread_env_var

import multiprocessing
import time
from pathlib import Path

import highspy
import numpy as np
from tqdm import tqdm

from standard_form import _HIGHS_INF, load_standard_form

from store import (
    SOLVE_RESULT_KEYS,
    STANDARD_FORM_ROW_CAP,
    list_class_names,
    list_instance_dirs,
    merge_ledger,
    process_instance_dirs,
    resolve_cache_root,
    summarize_records,
)

SOLVE_TIMEOUT = 600.0  # 10 minutes per file
HIGHS_THREADS = 1
HIGHS_VERSION = highspy.Highs().version()


def _standard_form_row_count(path: Path) -> int | None:
    """Read the small stored shape member, or defer malformed archives to the solver."""
    try:
        with np.load(path, allow_pickle=False) as data:
            raw_shape = np.asarray(data["A_shape"]).ravel()
        if not np.issubdtype(raw_shape.dtype, np.integer) or len(raw_shape) != 2:
            return None
        if np.any(raw_shape < 0):
            return None
        return int(raw_shape[0])
    except Exception:  # noqa: BLE001 - the normal solve records malformed archives
        return None


def _merge_solve_result(path: Path, status: str, elapsed: float | None = None) -> None:
    suffix = path.suffix.lower()[1:]
    data_path = path.with_suffix(".data")
    runtime_key, status_key = SOLVE_RESULT_KEYS[suffix]
    values = {}
    if elapsed is not None:
        values[runtime_key] = elapsed
    values[status_key] = status
    values["highs_version"] = HIGHS_VERSION
    values["highs_threads"] = HIGHS_THREADS
    merge_ledger(
        data_path,
        values,
        remove_keys=(runtime_key,) if elapsed is None else (),
    )


def _check_highs_call(status: highspy.HighsStatus, operation: str) -> None:
    if status not in (highspy.HighsStatus.kOk, highspy.HighsStatus.kWarning):
        raise RuntimeError(f"HiGHS {operation} failed: {status}")


def _solve_mps(path: Path) -> tuple[float | None, str]:
    """Read an MPS model and solve it with HiGHS's default presolve enabled."""
    highspy.Highs.resetGlobalScheduler(True)
    h = highspy.Highs()
    _check_highs_call(h.setOptionValue("threads", HIGHS_THREADS), "set threads option")
    h.setOptionValue("log_to_console", False)
    status = h.readModel(str(path))
    _check_highs_call(status, "readModel")
    integrality = getattr(h.getLp(), "integrality_", None)
    if integrality and any(v != highspy.HighsVarType.kContinuous for v in integrality):
        raise RuntimeError(f"Non-continuous variables are not supported: {path}")
    t0 = time.perf_counter()
    status = h.run()
    elapsed = time.perf_counter() - t0
    _check_highs_call(status, "solve")
    if h.getModelStatus() != highspy.HighsModelStatus.kOptimal:
        return None, "non_optimal"
    return elapsed, "ok"


def _solve_std(path: Path) -> tuple[float | None, str]:
    """Load a validated .std LP, build a HiGHS model, and solve it."""
    highspy.Highs.resetGlobalScheduler(True)
    c, b, A, _ = load_standard_form(path)
    m, n = A.shape

    h = highspy.Highs()
    _check_highs_call(h.setOptionValue("threads", HIGHS_THREADS), "set threads option")
    h.setOptionValue("log_to_console", False)
    col_lower = np.zeros(n, dtype=np.float64)
    col_upper = np.full(n, _HIGHS_INF, dtype=np.float64)
    _check_highs_call(h.addVars(n, col_lower, col_upper), "addVars")
    _check_highs_call(
        h.changeColsCost(n, np.arange(n, dtype=np.int64), c), "changeColsCost"
    )
    row_lower = b.copy()
    row_upper = b.copy()
    num_nz = int(A.nnz)
    starts = np.asarray(A.indptr[:-1], dtype=np.int32) if m > 0 else np.array([], dtype=np.int32)
    _check_highs_call(
        h.addRows(
            m,
            row_lower,
            row_upper,
            num_nz,
            starts,
            A.indices.astype(np.int32),
            A.data.astype(np.float64),
        ),
        "addRows",
    )

    t0 = time.perf_counter()
    status = h.run()
    if status not in (highspy.HighsStatus.kOk, highspy.HighsStatus.kWarning):
        # Retry with IPM when default solver fails (e.g. badly scaled RHS); IPM often handles scaling better.
        # The timer restarts deliberately: the baseline is the successful attempt only, since a
        # practitioner aware of the default-solver failure would run IPM directly.
        _check_highs_call(h.setOptionValue("solver", "ipm"), "set solver option")
        t0 = time.perf_counter()
        status = h.run()
        elapsed = time.perf_counter() - t0
        _check_highs_call(status, "IPM solve")
        if h.getModelStatus() != highspy.HighsModelStatus.kOptimal:
            return None, "non_optimal"
        return elapsed, "ok_ipm"
    elapsed = time.perf_counter() - t0
    if h.getModelStatus() != highspy.HighsModelStatus.kOptimal:
        return None, "non_optimal"
    return elapsed, "ok"


def _solve_instance_from_path(path: Path) -> None:
    """Solve the instance at path (.mps or .std) with HiGHS; write solve time to instance .data."""
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Instance file not found: {path}")

    suffix = path.suffix.lower()
    if suffix not in (".mps", ".std"):
        raise ValueError(f"Unsupported instance format: {suffix}. Use .mps or .std.")
    solve = _solve_mps if suffix == ".mps" else _solve_std
    try:
        elapsed, solve_status = solve(path)
    except Exception as exc:  # noqa: BLE001 - record solver/data failures per instance
        _merge_solve_result(path, f"error:{type(exc).__name__}")
        return
    _merge_solve_result(path, solve_status, elapsed)


def _solve_with_timeout(path: Path) -> str:
    """Run _solve_instance_from_path in a subprocess with SOLVE_TIMEOUT.

    Returns "completed", "timeout", or "crashed" and records parent-detected failures.
    """
    # A fresh child cannot inherit a parent-sized HiGHS scheduler and applies
    # the BLAS thread defaults before NumPy is imported.
    p = multiprocessing.get_context("spawn").Process(
        target=_solve_instance_from_path, args=(path,)
    )
    p.start()
    p.join(SOLVE_TIMEOUT)
    if p.is_alive():
        p.terminate()
        p.join(5)
        if p.is_alive():
            p.kill()
            p.join()
        p.close()
        _merge_solve_result(path, "timeout")
        return "timeout"
    exitcode = p.exitcode
    p.close()
    if exitcode != 0:
        _merge_solve_result(path, "crashed")
        return "crashed"
    return "completed"


def solve_instance(
    instance_class: str,
    instance_name: str,
    cache_dir: str | Path | None = None,
    format: str = "both",
) -> None:
    """Solve the instance(s) in cache_dir/instance_class/instance_name/ with HiGHS.

    Discovers .mps and/or .std in that subdirectory according to format; writes runtime to instance .data.
    instance_class: e.g. "netlib", "miplib".
    instance_name: subfolder name (instance stem).
    cache_dir: root containing instance-class subfolders; defaults to "cache_dir".
    format: "mps" | "std" | "both" — which format to solve (default "both").
    """
    if format not in ("mps", "std", "both"):
        raise ValueError(f"format must be 'mps', 'std', or 'both'; got {format!r}")

    root = resolve_cache_root(cache_dir)
    instance_dir = root / instance_class / instance_name
    if not instance_dir.is_dir():
        raise FileNotFoundError(f"Instance directory not found: {instance_dir}")

    mps_paths = sorted(instance_dir.glob("*.mps")) if format in ("mps", "both") else []
    std_paths = sorted(instance_dir.glob("*.std")) if format in ("std", "both") else []

    for p in mps_paths:
        if _solve_with_timeout(p) == "timeout":
            tqdm.write(f"timeout: {p.name} (skipping)")

    for p in std_paths:
        row_count = _standard_form_row_count(p)
        if row_count is not None and row_count > STANDARD_FORM_ROW_CAP:
            _merge_solve_result(p, "skipped_too_large")
            continue
        if _solve_with_timeout(p) == "timeout":
            tqdm.write(f"timeout: {p.name} (skipping)")


def solve_instance_class(
    instance_class: str,
    cache_dir: str | Path | None = None,
    format: str = "both",
) -> None:
    """Solve instances in the given instance-class subfolder of cache_dir.

    instance_class: name of the subfolder (e.g. "netlib", "miplib", "clique").
    cache_dir: directory containing instance-class subfolders; defaults to "cache_dir" in the current directory.
    format: "mps" | "std" | "both" — which instance format to solve (default "both").
    """
    if format not in ("mps", "std", "both"):
        raise ValueError(f"format must be 'mps', 'std', or 'both'; got {format!r}")

    root = resolve_cache_root(cache_dir)
    folder = root / instance_class
    if not folder.is_dir():
        raise FileNotFoundError(f"Instance class folder not found: {folder}")

    subdirs = list_instance_dirs(folder)
    process_instance_dirs(
        instance_class,
        subdirs,
        lambda subdir: solve_instance(
            instance_class, subdir.name, cache_dir=root, format=format
        ),
    )


def solve_all_instance_classes(
    instance_classes: list[str] | None = None,
    cache_dir: str | Path | None = None,
    format: str = "both",
) -> None:
    """Solve instances for given or all instance classes (main entry point).

    instance_classes: optional list of instance class names (subfolder names under cache_dir).
        If None, all instance classes (all subdirectories of cache_dir) are processed.
    cache_dir: directory containing instance-class subfolders; defaults to "cache_dir" in the current directory.
    format: "mps" | "std" | "both" — which instance format to solve (default "both").
    """
    root = resolve_cache_root(cache_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Cache directory not found: {root}")

    if instance_classes is None:
        instance_classes = list_class_names(root)

    for name in instance_classes:
        solve_instance_class(name, root, format=format)
        show_solve_status([name], format=format, cache_dir=root)


def show_solve_status(
    instance_classes: list[str] | None = None,
    format: str = "both",
    cache_dir: str | Path | None = None,
) -> None:
    """Print how many instances per class have valid solve-time entries in their .data files.

    For each instance class, prints one line showing counts for the active format(s):
    "<class>  [mps: x/total]  [std: x/total]".
    """
    root = resolve_cache_root(cache_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Cache directory not found: {root}")

    if instance_classes is None:
        instance_classes = list_class_names(root)

    active_formats = ["mps", "std"] if format == "both" else [format]

    for cls in instance_classes:
        folder = root / cls
        if not folder.is_dir():
            print(f"{cls}: directory not found")
            continue

        subdirs = list_instance_dirs(folder)
        total = len(subdirs)
        summaries = {
            fmt: summarize_records(
                subdirs,
                value_key=SOLVE_RESULT_KEYS[fmt][0],
                status_key=SOLVE_RESULT_KEYS[fmt][1],
                ok_statuses=("ok", "ok_ipm"),
            )
            for fmt in active_formats
        }

        parts = []
        for fmt in active_formats:
            count, non_ok = summaries[fmt]
            breakdown = ", ".join(
                f"{status}: {number}" for status, number in sorted(non_ok.items())
            )
            suffix = f" ({breakdown})" if breakdown else ""
            parts.append(f"{fmt}: {count}/{total}{suffix}")
        parts = "  ".join(parts)
        print(f"{cls}:  {parts}")



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Solve standard-form LP instances with HiGHS and record solve time.")
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
        "--format",
        choices=("mps", "std", "both"),
        default="both",
        help="Instance format to solve: mps, std, or both (default: both).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show how many instances per class have solve-time data for the selected format(s). Other flags are ignored.",
    )
    args = parser.parse_args()
    if args.show:
        show_solve_status(
            instance_classes=args.instance_classes or None,
            format=args.format,
            cache_dir=args.cache_dir,
        )
    else:
        solve_all_instance_classes(
            instance_classes=args.instance_classes or None,
            cache_dir=args.cache_dir,
            format=args.format,
        )
