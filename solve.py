#!/usr/bin/env python3
"""Solve LP instances (MPS or standard-form .std) with HiGHS and record solve time."""

from __future__ import annotations

import json
import math
import multiprocessing
import os
import tempfile
import time
from pathlib import Path

import highspy
import numpy as np
from scipy.sparse import csr_matrix
from tqdm import tqdm

SOLVE_TIMEOUT = 600.0  # 10 minutes per file

try:
    _HIGHS_INF = highspy.kHighsInf
except AttributeError:
    _HIGHS_INF = 1e30


def _atomic_write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as tmp:
            tmp_name = tmp.name
            json.dump(data, tmp, indent=None)
        os.chmod(tmp_name, 0o644)
        os.replace(tmp_name, path)
    finally:
        if tmp_name is not None and os.path.exists(tmp_name):
            os.unlink(tmp_name)


def _merge_solve_result(path: Path, status: str, elapsed: float | None = None) -> None:
    suffix = path.suffix.lower()[1:]
    data_path = path.with_suffix(".data")
    try:
        data = json.loads(data_path.read_text()) if data_path.exists() else {}
    except (json.JSONDecodeError, UnicodeDecodeError):
        data = {}
    if not isinstance(data, dict):
        data = {}
    runtime_key = f"runtime_highs_{suffix}"
    if elapsed is None:
        data.pop(runtime_key, None)
    else:
        data[runtime_key] = elapsed
    data[f"solve_status_{suffix}"] = status
    _atomic_write_json(data_path, data)


def _check_highs_call(status: highspy.HighsStatus, operation: str) -> None:
    if status not in (highspy.HighsStatus.kOk, highspy.HighsStatus.kWarning):
        raise RuntimeError(f"HiGHS {operation} failed: {status}")


def _solve_mps(path: Path) -> tuple[float | None, str]:
    """Read an MPS model and solve it with HiGHS's default presolve enabled."""
    h = highspy.Highs()
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
    data = np.load(path)
    c = np.asarray(data["c"], dtype=np.float64).ravel()
    b = np.asarray(data["b"], dtype=np.float64).ravel()
    A_data = np.asarray(data["A_data"], dtype=np.float64).ravel()
    raw_indices = np.asarray(data["A_indices"]).ravel()
    raw_indptr = np.asarray(data["A_indptr"]).ravel()
    raw_shape = np.asarray(data["A_shape"]).ravel()

    def _integer_metadata(values: np.ndarray, name: str) -> np.ndarray:
        try:
            integer_valued = np.issubdtype(values.dtype, np.integer) or (
                np.all(np.isfinite(values)) and np.all(values == np.floor(values))
            )
        except TypeError:
            integer_valued = False
        if not integer_valued:
            raise ValueError(f".std {name} must be integer-valued")
        return np.asarray(values, dtype=np.int64)

    A_indices = _integer_metadata(raw_indices, "A_indices")
    A_indptr = _integer_metadata(raw_indptr, "A_indptr")
    A_shape = _integer_metadata(raw_shape, "A_shape")
    if A_shape.size != 2 or np.any(A_shape < 0):
        raise ValueError("A_shape must contain two non-negative dimensions")
    n, m = int(A_shape[1]), int(A_shape[0])  # A is (m, n)
    if len(c) != n or len(b) != m or len(A_indptr) != m + 1:
        raise ValueError(".std vector or CSR pointer dimensions do not match A_shape")
    if not all(np.all(np.isfinite(values)) for values in (c, b, A_data)):
        raise ValueError(".std contains non-finite numeric values")
    if len(A_indices) != len(A_data):
        raise ValueError(".std CSR indices and values have different lengths")
    if A_indptr.size and (
        A_indptr[0] != 0
        or A_indptr[-1] != len(A_data)
        or np.any(np.diff(A_indptr) < 0)
    ):
        raise ValueError(".std has invalid CSR row pointers")
    if np.any(A_indices < 0) or np.any(A_indices >= n):
        raise ValueError(".std CSR column index is out of range")
    int32_max = np.iinfo(np.int32).max
    if m > int32_max or n > int32_max or len(A_data) > int32_max:
        raise OverflowError(".std dimensions exceed HiGHS int32 sparse-index limits")
    if np.any(A_indptr > int32_max) or np.any(A_indices > int32_max):
        raise OverflowError(".std sparse indices exceed int32 range")
    A = csr_matrix((A_data, A_indices, A_indptr), shape=(m, n))

    h = highspy.Highs()
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
        # Retry with IPM when default solver fails (e.g. badly scaled RHS); IPM often handles scaling better
        _check_highs_call(h.setOptionValue("solver", "ipm"), "set solver option")
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
    p = multiprocessing.Process(target=_solve_instance_from_path, args=(path,))
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

    root = Path(cache_dir).resolve() if cache_dir is not None else Path("cache_dir").resolve()
    instance_dir = root / instance_class / instance_name
    if not instance_dir.is_dir():
        raise FileNotFoundError(f"Instance directory not found: {instance_dir}")

    mps_paths = sorted(instance_dir.glob("*.mps")) if format in ("mps", "both") else []
    std_paths = sorted(instance_dir.glob("*.std")) if format in ("std", "both") else []

    mps_timed_out = False
    for p in mps_paths:
        if _solve_with_timeout(p) == "timeout":
            tqdm.write(f"timeout: {p.name} (skipping)")
            mps_timed_out = True

    if format == "both" and mps_timed_out:
        return

    for p in std_paths:
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

    root = Path(cache_dir).resolve() if cache_dir is not None else Path("cache_dir").resolve()
    folder = root / instance_class
    if not folder.is_dir():
        raise FileNotFoundError(f"Instance class folder not found: {folder}")

    subdirs = sorted(d for d in folder.iterdir() if d.is_dir())
    for subdir in tqdm(subdirs, desc=instance_class, unit="instance"):
        try:
            solve_instance(instance_class, subdir.name, cache_dir=root, format=format)
        except Exception as exc:  # noqa: BLE001 - isolate corpus instances
            tqdm.write(f"skipping {subdir.name}: {exc}")


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
    root = Path(cache_dir).resolve() if cache_dir is not None else Path("cache_dir").resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Cache directory not found: {root}")

    if instance_classes is None:
        instance_classes = [f.name for f in sorted(root.iterdir()) if f.is_dir()]

    for name in instance_classes:
        solve_instance_class(name, root, format=format)


def show_solve_status(
    instance_classes: list[str] | None = None,
    format: str = "both",
    cache_dir: str | Path | None = None,
) -> None:
    """Print how many instances per class have valid solve-time entries in their .data files.

    For each instance class, prints one line showing counts for the active format(s):
    "<class>  [mps: x/total]  [std: x/total]".
    """
    root = Path(cache_dir).resolve() if cache_dir is not None else Path("cache_dir").resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Cache directory not found: {root}")

    if instance_classes is None:
        instance_classes = [f.name for f in sorted(root.iterdir()) if f.is_dir()]

    active_keys = {
        "mps": "runtime_highs_mps",
        "std": "runtime_highs_std",
    }
    active_formats = ["mps", "std"] if format == "both" else [format]

    for cls in instance_classes:
        folder = root / cls
        if not folder.is_dir():
            print(f"{cls}: directory not found")
            continue

        subdirs = sorted(d for d in folder.iterdir() if d.is_dir())
        total = len(subdirs)

        counts: dict[str, int] = {fmt: 0 for fmt in active_formats}
        for subdir in subdirs:
            data_files = list(subdir.glob("*.data"))
            try:
                data = json.loads(data_files[0].read_text()) if data_files else {}
            except (json.JSONDecodeError, UnicodeDecodeError):
                data = {}
            if not isinstance(data, dict):
                data = {}
            for fmt in active_formats:
                status_key = f"solve_status_{fmt}"
                value = data.get(active_keys[fmt])
                legacy_done = (
                    status_key not in data
                    and isinstance(value, (int, float))
                    and math.isfinite(value)
                )
                if data.get(status_key) in ("ok", "ok_ipm") or legacy_done:
                    counts[fmt] += 1

        parts = "  ".join(f"{fmt}: {counts[fmt]}/{total}" for fmt in active_formats)
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
