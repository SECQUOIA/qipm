#!/usr/bin/env python3
"""Transform MPS instances to standard-form LP (min c'x, Ax=b, x>=0) after presolve."""

from __future__ import annotations

import json
import os
import tempfile
import warnings
from pathlib import Path

import highspy
from tqdm import tqdm
import numpy as np
from scipy.sparse import csr_matrix

# HiGHS uses ±inf for unbounded bounds
try:
    _HIGHS_INF = highspy.kHighsInf
except AttributeError:
    _HIGHS_INF = 1e30  # fallback if constant not exposed


def _lp_to_standard_form(
    num_col: int,
    num_row: int,
    col_cost: np.ndarray,
    col_lower: np.ndarray,
    col_upper: np.ndarray,
    row_lower: np.ndarray,
    row_upper: np.ndarray,
    a_start: np.ndarray,
    a_index: np.ndarray,
    a_value: np.ndarray,
    obj_offset: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, csr_matrix, float]:
    """Convert HiGHS LP (col_lower <= x <= col_upper, row_lower <= Ax <= row_upper) to standard form.

    Standard form: min c'x  s.t.  Ax = b,  x >= 0.

    Returns (c, b, A, obj_offset) with A in CSR sparse format.
    """
    inf = _HIGHS_INF
    # Convert to numpy arrays once (avoid repeated conversion in caller)
    col_cost = np.asarray(col_cost, dtype=np.float64).ravel()
    col_lower = np.asarray(col_lower, dtype=np.float64).ravel()
    col_upper = np.asarray(col_upper, dtype=np.float64).ravel()
    row_lower = np.asarray(row_lower, dtype=np.float64).ravel()
    row_upper = np.asarray(row_upper, dtype=np.float64).ravel()
    a_start = np.asarray(a_start, dtype=np.int64).ravel()
    a_index = np.asarray(a_index, dtype=np.int64).ravel()
    a_value = np.asarray(a_value, dtype=np.float64).ravel()

    # Precompute dense row index mapping: fully-free rows (lo=-inf, hi=inf) contribute
    # no constraint in standard form and are dropped entirely.
    free_row = (row_lower <= -inf) & (row_upper >= inf)
    row_map = np.where(free_row, -1, np.cumsum(~free_row) - 1).astype(np.int64)
    m_base = int((~free_row).sum())  # number of non-free original rows

    new_col_count = 0
    c_list: list[float] = []
    b_list: list[float] = []
    extra_b_list: list[float] = []
    row_list: list[int] = []
    col_list: list[int] = []
    val_list: list[float] = []
    extra_row_list: list[int] = []
    extra_col_list: list[int] = []
    extra_val_list: list[float] = []
    row_constant = np.zeros(num_row, dtype=np.float64)
    objective_constant = float(obj_offset)

    def add_var(cost: float) -> int:
        nonlocal new_col_count
        j = new_col_count
        new_col_count += 1
        c_list.append(cost)
        return j

    # Single pass over columns: map to non-negative variables and accumulate row_constant
    for j in range(num_col):
        lj = col_lower[j]
        uj = col_upper[j]
        cj = col_cost[j]
        beg = a_start[j]
        end = a_start[j + 1]
        row_ind = a_index[beg:end]
        row_vals = a_value[beg:end]

        if lj > -inf and uj < inf:
            objective_constant += cj * lj
            j1 = add_var(cj)
            j2 = add_var(0.0)
            width = uj - lj
            r = m_base + len(extra_b_list)
            extra_b_list.append(width)
            extra_row_list.append(r)
            extra_col_list.append(j1)
            extra_val_list.append(1.0)
            extra_row_list.append(r)
            extra_col_list.append(j2)
            extra_val_list.append(1.0)
            for idx in range(len(row_ind)):
                i = row_ind[idx]
                v = row_vals[idx]
                ri = row_map[i]
                if ri >= 0:
                    row_list.append(ri)
                    col_list.append(j1)
                    val_list.append(v)
                row_constant[i] += v * lj
            continue

        if lj > -inf and uj >= inf:
            objective_constant += cj * lj
            j1 = add_var(cj)
            for idx in range(len(row_ind)):
                i = row_ind[idx]
                v = row_vals[idx]
                ri = row_map[i]
                if ri >= 0:
                    row_list.append(ri)
                    col_list.append(j1)
                    val_list.append(v)
                row_constant[i] += v * lj
            continue
        if lj <= -inf and uj < inf:
            objective_constant += cj * uj
            j1 = add_var(-cj)
            for idx in range(len(row_ind)):
                i = row_ind[idx]
                v = row_vals[idx]
                ri = row_map[i]
                if ri >= 0:
                    row_list.append(ri)
                    col_list.append(j1)
                    val_list.append(-v)
                row_constant[i] += v * uj
            continue
        # Free variable
        j_plus = add_var(cj)
        j_minus = add_var(-cj)
        for idx in range(len(row_ind)):
            i = row_ind[idx]
            v = row_vals[idx]
            ri = row_map[i]
            if ri >= 0:
                row_list.append(ri)
                col_list.append(j_plus)
                val_list.append(v)
                row_list.append(ri)
                col_list.append(j_minus)
                val_list.append(-v)

    # Add slacks and set b for original rows
    for i in range(num_row):
        lo = row_lower[i]
        hi = row_upper[i]
        ri = row_map[i]
        if lo == hi:
            b_list.append(lo - row_constant[i])
        elif lo <= -inf and hi < inf:
            j_slack = add_var(0.0)
            row_list.append(ri)
            col_list.append(j_slack)
            val_list.append(1.0)
            b_list.append(hi - row_constant[i])
        elif lo > -inf and hi >= inf:
            j_slack = add_var(0.0)
            row_list.append(ri)
            col_list.append(j_slack)
            val_list.append(-1.0)
            b_list.append(lo - row_constant[i])
        elif lo <= -inf and hi >= inf:
            # Fully free row: no constraint, skip entirely.
            pass
        else:
            j_slack = add_var(0.0)
            row_list.append(ri)
            col_list.append(j_slack)
            val_list.append(1.0)
            b_list.append(hi - row_constant[i])
            extra_b_list.append(hi - lo)
            r = m_base + len(extra_b_list) - 1
            extra_row_list.append(r)
            extra_col_list.append(j_slack)
            extra_val_list.append(1.0)
            j_s2 = add_var(0.0)
            extra_row_list.append(r)
            extra_col_list.append(j_s2)
            extra_val_list.append(1.0)

    n_std = new_col_count
    m_std = len(b_list) + len(extra_b_list)
    b_std = np.empty(m_std, dtype=np.float64)
    b_std[:len(b_list)] = b_list
    b_std[len(b_list):] = extra_b_list
    c_std = np.fromiter(c_list, dtype=np.float64, count=len(c_list))

    nnz = len(row_list) + len(extra_row_list)
    if nnz == 0:
        A_std = csr_matrix((m_std, n_std))
    else:
        row_arr = np.empty(nnz, dtype=np.int64)
        col_arr = np.empty(nnz, dtype=np.int64)
        val_arr = np.empty(nnz, dtype=np.float64)
        n1 = len(row_list)
        row_arr[:n1] = row_list
        col_arr[:n1] = col_list
        val_arr[:n1] = val_list
        row_arr[n1:] = extra_row_list
        col_arr[n1:] = extra_col_list
        val_arr[n1:] = extra_val_list
        A_std = csr_matrix((val_arr, (row_arr, col_arr)), shape=(m_std, n_std))
    return c_std, b_std, A_std, objective_constant


def _atomic_write_json(path: Path, data: dict) -> None:
    """Atomically replace a JSON file with data."""
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


_DOWNSTREAM_DATA_KEYS = (
    "cycle_count_mnes",
    "cycle_count_oss",
    "sparsity_mnes",
    "sparsity_oss",
    "cond_mnes",
    "cond_oss",
    "status_mnes",
    "status_oss",
    "runtime_highs_std",
    "solve_status_std",
)


def _read_data_object(data_path: Path) -> dict:
    """Read a JSON object, treating corrupt or non-object data as empty."""
    try:
        data = json.loads(data_path.read_text()) if data_path.exists() else {}
    except (json.JSONDecodeError, UnicodeDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _merge_transform_status(path: Path, status: str, *, retract: bool = False) -> None:
    data_path = path.with_suffix(".data")
    data = _read_data_object(data_path)
    if retract:
        for key in _DOWNSTREAM_DATA_KEYS:
            data.pop(key, None)
    data["transform_status"] = status
    _atomic_write_json(data_path, data)


def _purge_downstream_data(path: Path) -> None:
    """Atomically retract results derived from an earlier standard form."""
    data_path = path.with_suffix(".data")
    data = _read_data_object(data_path)
    for key in _DOWNSTREAM_DATA_KEYS:
        data.pop(key, None)
    _atomic_write_json(data_path, data)


def _withhold_standard_form(path: Path, status: str) -> None:
    _merge_transform_status(path, status, retract=True)
    path.with_suffix(".std").unlink(missing_ok=True)


def _atomic_write_std(path: Path, **arrays: np.ndarray | float) -> None:
    """Atomically write a compressed NPZ using the requested .std filename."""
    tmp_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as tmp:
            tmp_name = tmp.name
            np.savez_compressed(tmp, **arrays)
        os.chmod(tmp_name, 0o644)
        os.replace(tmp_name, path)
    finally:
        if tmp_name is not None and os.path.exists(tmp_name):
            os.unlink(tmp_name)


def _std_matches(path: Path, arrays: dict[str, np.ndarray | float]) -> bool:
    """Return whether an existing .std contains exactly the produced arrays."""
    if not path.is_file():
        return False
    try:
        with np.load(path, allow_pickle=False) as existing:
            return all(
                key in existing.files and np.array_equal(existing[key], np.asarray(value))
                for key, value in arrays.items()
            )
    except Exception:  # noqa: BLE001 - unreadable existing output must be replaced
        return False


def _strip_zero_rows(A: csr_matrix, b: np.ndarray) -> tuple[csr_matrix, np.ndarray] | None:
    """Drop harmless zero rows, returning None when a zero row has nonzero RHS."""
    row_nnz = np.diff(A.indptr)
    zero_rows = row_nnz == 0
    if np.any(zero_rows & (np.abs(b) > 1e-9)):
        return None
    keep = ~zero_rows
    if not keep.all():
        return A[keep], b[keep]
    return A, b


def _transform_instance_impl(path: Path) -> bool | None:
    """Read MPS file at path, presolve with HiGHS, convert to standard form, and save .std next to it."""
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"MPS file not found: {path}")
    if path.stat().st_size == 0:
        raise RuntimeError(f"MPS file is empty: {path}")

    h = highspy.Highs()
    h.setOptionValue("log_to_console", False)
    status = h.readModel(str(path))
    if status == highspy.HighsStatus.kWarning:
        warnings.warn(f"HiGHS readModel returned kWarning for {path}", stacklevel=2)
    elif status != highspy.HighsStatus.kOk:
        raise RuntimeError(f"HiGHS readModel failed: {status}")

    original_lp = h.getLp()
    if original_lp.sense_ == highspy.ObjSense.kMaximize:
        raise RuntimeError(f"Maximization model is not supported: {path}")
    integrality = getattr(original_lp, "integrality_", None)
    if integrality and any(v != highspy.HighsVarType.kContinuous for v in integrality):
        raise RuntimeError(f"Non-continuous variables are not supported: {path}")

    status = h.presolve()
    if status == highspy.HighsStatus.kWarning:
        warnings.warn(f"HiGHS presolve returned kWarning for {path}", stacklevel=2)
    elif status != highspy.HighsStatus.kOk:
        raise RuntimeError(f"HiGHS presolve failed: {status}")

    presolve_status = h.getModelPresolveStatus()
    model_status = h.getModelStatus()
    if presolve_status == highspy.HighsPresolveStatus.kReducedToEmpty:
        _withhold_standard_form(path, "reduced_to_empty")
        return
    if (
        presolve_status == highspy.HighsPresolveStatus.kInfeasible
        or model_status == highspy.HighsModelStatus.kInfeasible
    ):
        _withhold_standard_form(path, "infeasible")
        return
    if (
        presolve_status == highspy.HighsPresolveStatus.kUnboundedOrInfeasible
        or model_status == highspy.HighsModelStatus.kUnboundedOrInfeasible
    ):
        _withhold_standard_form(path, "unbounded_or_infeasible")
        return
    if model_status == highspy.HighsModelStatus.kUnbounded:
        _withhold_standard_form(path, "unbounded")
        return
    if presolve_status not in (
        highspy.HighsPresolveStatus.kReduced,
        highspy.HighsPresolveStatus.kNotReduced,
    ):
        raise RuntimeError(f"Unexpected HiGHS presolve status: {presolve_status}")

    lp = h.getPresolvedLp()
    if lp is None or (lp.num_col_ == 0 and lp.num_row_ == 0):
        _withhold_standard_form(path, "reduced_to_empty")
        return

    num_col = lp.num_col_
    num_row = lp.num_row_
    a = lp.a_matrix_
    c, b, A, obj_offset = _lp_to_standard_form(
        num_col, num_row,
        lp.col_cost_, lp.col_lower_, lp.col_upper_,
        lp.row_lower_, lp.row_upper_,
        a.start_, a.index_, a.value_,
        lp.offset_,
    )

    # Presolve normally removes empty rows. Keep this as defensive handling for
    # harmless 0 = 0 rows that a solver version or input edge case may retain.
    stripped = _strip_zero_rows(A, b)
    if stripped is None:
        _withhold_standard_form(path, "infeasible")
        return
    A, b = stripped

    arrays = {
        "c": c,
        "b": b,
        "A_data": A.data,
        "A_indices": A.indices,
        "A_indptr": A.indptr,
        "A_shape": np.array(A.shape),
        "obj_offset": np.array(obj_offset, dtype=np.float64),
    }
    out_std = path.with_suffix(".std")
    changed = not _std_matches(out_std, arrays)
    if changed:
        _purge_downstream_data(path)
        _atomic_write_std(out_std, **arrays)
    return changed


def _transform_instance_from_path(path: Path) -> None:
    """Transform one discovered MPS path, retracting stale outputs on failure."""
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"MPS file not found: {path}")
    try:
        changed = _transform_instance_impl(path)
    except Exception as exc:
        _withhold_standard_form(path, f"error:{type(exc).__name__}")
        raise
    if changed is not None:
        _merge_transform_status(path, "ok")


def transform_instance(
    instance_class: str,
    instance_name: str,
    cache_dir: str | Path | None = None,
) -> None:
    """Transform the MPS instance in cache_dir/instance_class/instance_name/ to standard form.

    Discovers the single .mps file in that subdirectory and writes .std next to it.
    instance_class: e.g. "netlib", "miplib".
    instance_name: subfolder name (instance stem).
    cache_dir: root containing instance-class subfolders; defaults to "cache_dir".
    """
    root = Path(cache_dir).resolve() if cache_dir is not None else Path("cache_dir").resolve()
    instance_dir = root / instance_class / instance_name
    if not instance_dir.is_dir():
        raise FileNotFoundError(f"Instance directory not found: {instance_dir}")
    mps_files = sorted(instance_dir.glob("*.mps"))
    if len(mps_files) != 1:
        raise FileNotFoundError(
            f"Expected exactly one .mps in {instance_dir}; found {len(mps_files)}"
        )
    _transform_instance_from_path(mps_files[0])


def transform_instance_class(
    instance_class: str,
    cache_dir: str | Path | None = None,
) -> None:
    """Transform all MPS instances in the given instance-class subfolder of cache_dir to standard form.

    instance_class: name of the subfolder (e.g. "netlib", "miplib", "clique").
    cache_dir: directory containing instance-class subfolders; defaults to "cache_dir" in the current directory.
    """
    root = Path(cache_dir).resolve() if cache_dir is not None else Path("cache_dir").resolve()
    folder = root / instance_class
    if not folder.is_dir():
        raise FileNotFoundError(f"Instance class folder not found: {folder}")

    subdirs = sorted(d for d in folder.iterdir() if d.is_dir())
    for subdir in tqdm(subdirs, desc=instance_class, unit="instance"):
        if not any(subdir.glob("*.mps")):
            tqdm.write(f"skipping {subdir.name}: no .mps file")
            continue
        try:
            transform_instance(instance_class, subdir.name, cache_dir=root)
        except Exception as exc:  # noqa: BLE001 - isolate corpus instances
            tqdm.write(f"skipping {subdir.name}: {exc}")


def transform_all_instance_classes(
    instance_classes: list[str] | None = None,
    cache_dir: str | Path | None = None,
) -> None:
    """Transform MPS instances to standard form (main entry point).

    instance_classes: optional list of instance class names (subfolder names under cache_dir).
        If None, all instance classes (all subdirectories of cache_dir) are processed.
    cache_dir: directory containing instance-class subfolders; defaults to "cache_dir" in the current directory.
    """
    root = Path(cache_dir).resolve() if cache_dir is not None else Path("cache_dir").resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Cache directory not found: {root}")

    if instance_classes is None:
        instance_classes = [f.name for f in sorted(root.iterdir()) if f.is_dir()]

    for name in instance_classes:
        transform_instance_class(name, root)


def show_transform_status(
    instance_classes: list[str] | None = None,
    cache_dir: str | Path | None = None,
) -> None:
    """Print how many instances per class have a .std file.

    For each instance class, prints "<class>: x/total".
    """
    root = Path(cache_dir).resolve() if cache_dir is not None else Path("cache_dir").resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Cache directory not found: {root}")

    if instance_classes is None:
        instance_classes = [f.name for f in sorted(root.iterdir()) if f.is_dir()]

    for cls in instance_classes:
        folder = root / cls
        if not folder.is_dir():
            print(f"{cls}: directory not found")
            continue

        subdirs = sorted(d for d in folder.iterdir() if d.is_dir())
        total = len(subdirs)
        done = sum(1 for d in subdirs if any(d.glob("*.std")))
        print(f"{cls}: {done}/{total}")


def clear_std_files(
    instance_classes: list[str] | None = None,
    cache_dir: str | Path | None = None,
) -> None:
    """Delete .std files and purge their transform, solve, and benchmark data."""
    root = Path(cache_dir).resolve() if cache_dir is not None else Path("cache_dir").resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Cache directory not found: {root}")

    search_roots = [root / name for name in instance_classes] if instance_classes else [root]
    for search_root in search_roots:
        instance_dirs = {
            path.parent
            for pattern in ("*.std", "*.data")
            for path in search_root.rglob(pattern)
        }
        for instance_dir in sorted(instance_dirs, key=str):
            for data_path in instance_dir.glob("*.data"):
                invalid_data = False
                try:
                    data = json.loads(data_path.read_text())
                except (json.JSONDecodeError, UnicodeDecodeError):
                    data = {}
                    invalid_data = True
                if not isinstance(data, dict):
                    data = {}
                    invalid_data = True
                keys = ("transform_status",) + _DOWNSTREAM_DATA_KEYS
                if invalid_data or any(key in data for key in keys):
                    for key in keys:
                        data.pop(key, None)
                    _atomic_write_json(data_path, data)
            for std_path in instance_dir.glob("*.std"):
                std_path.unlink()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Transform MPS instances to standard-form LP.")
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
        "--clear",
        action="store_true",
        help="Delete all .std files instead of transforming. Other flags are ignored.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show how many instances per class have a .std file. Other flags are ignored.",
    )
    args = parser.parse_args()
    if args.show:
        show_transform_status(
            instance_classes=args.instance_classes or None,
            cache_dir=args.cache_dir,
        )
    elif args.clear:
        clear_std_files(
            instance_classes=args.instance_classes or None,
            cache_dir=args.cache_dir,
        )
    else:
        transform_all_instance_classes(
            instance_classes=args.instance_classes or None,
            cache_dir=args.cache_dir,
        )
