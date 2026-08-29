"""Standard-form conversion and the shared .std archive codec."""

from __future__ import annotations

from pathlib import Path

import highspy
import numpy as np
from scipy.sparse import csr_matrix

from store import atomic_write_npz

_HIGHS_INF = getattr(highspy, "kHighsInf", 1e30)


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


def standard_form_arrays(
    c: np.ndarray, b: np.ndarray, A: csr_matrix, obj_offset: float
) -> dict[str, np.ndarray]:
    return {
        "c": c,
        "b": b,
        "A_data": A.data,
        "A_indices": A.indices,
        "A_indptr": A.indptr,
        "A_shape": np.array(A.shape),
        "obj_offset": np.array(obj_offset, dtype=np.float64),
    }


def _atomic_write_std(path: Path, **arrays: np.ndarray | float) -> None:
    """Atomically write the standard-form NPZ layout at a .std path."""
    # Monkeypatch seam pinned by the interrupted-write regression in test_transform.py.
    atomic_write_npz(path, arrays)


def write_standard_form(
    path: Path, c: np.ndarray, b: np.ndarray, A: csr_matrix, obj_offset: float = 0.0
) -> None:
    _atomic_write_std(path, **standard_form_arrays(c, b, A, obj_offset))


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


def load_standard_form(path: Path) -> tuple[np.ndarray, np.ndarray, csr_matrix, float]:
    """Load and validate a standard-form archive."""
    with np.load(path, allow_pickle=False) as data:
        c = np.asarray(data["c"], dtype=np.float64).ravel()
        b = np.asarray(data["b"], dtype=np.float64).ravel()
        A_data = np.asarray(data["A_data"], dtype=np.float64).ravel()
        raw_indices = np.asarray(data["A_indices"]).ravel()
        raw_indptr = np.asarray(data["A_indptr"]).ravel()
        raw_shape = np.asarray(data["A_shape"]).ravel()
        try:
            obj_offset = (
                float(np.asarray(data["obj_offset"])) if "obj_offset" in data else 0.0
            )
        except Exception:  # noqa: BLE001 - optional metadata is fully best-effort
            # Solve historically ignored this metadata. Preserve that behavior
            # for malformed archives while still returning the contract value.
            obj_offset = 0.0

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
    n, m = int(A_shape[1]), int(A_shape[0])
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
    return c, b, A, obj_offset
