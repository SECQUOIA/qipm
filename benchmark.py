#!/usr/bin/env python3
"""Benchmark LP instances: A from .std; compute cycle counts for mnes/oss and write to instance .data (JSON)."""

from __future__ import annotations

import json
import math
import multiprocessing
import os
import signal
import tempfile
import time
from pathlib import Path

import numpy as np
import sparseqr
from scipy.sparse import csr_matrix
from tqdm import tqdm

_EPSILON = 1e-1  # precision shared by QLSA and outer Newton-step count
_PREPROCESS_TIMEOUT = 600  # seconds; basis preprocessing time limit
_MNES_SM_TIMEOUT = 60     # seconds; wall-clock limit for svds("SM") in MNES
_MNES_N_PROBES = 10_000   # random probes for singular-value fallbacks
_OSS_SM_TIMEOUT = 60      # seconds; wall-clock limit for svds("SM") in OSS
_OSS_N_PROBES = 10_000    # random probes for singular-value fallbacks


class RankUncertainError(RuntimeError):
    """SPQR dropped a row whose dependence could not be certified."""


class PreprocessTimeoutError(RuntimeError):
    """Basis preprocessing exceeded its wall-clock limit."""


class PreprocessCrashedError(RuntimeError):
    """Basis preprocessing worker exited without a usable result."""


class DegenerateInstanceError(RuntimeError):
    """Preprocessing reduced the benchmark system below two rows."""


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


def cycle_count_qlsa(
    *,
    s: int,
    k: float,
    epsilon: float = 1e-1,
) -> int:
    """
    Return the QLSA Chebyshev query count.

    Args:
        s: Maximum sparsity (non-zeros per row or column) of M̂.
        k: Condition number (2-norm) of M̂.
        epsilon: Precision (default 1e-1).

    Returns:
        int: The number of queries that QLS Chebyshev makes to O_H and O_F (P_A).
    """
    if s <= 0 or k < 1 or epsilon <= 0 or not math.isfinite(k):
        raise ValueError("s and epsilon must be positive and k must be finite and at least 1")
    sk = float(s) * k
    if not math.isfinite(sk) or sk > math.sqrt(np.finfo(np.float64).max):
        raise OverflowError("s*k is too large for the QLSA cycle-count formula")
    binst = math.ceil(math.log(sk / epsilon) * sk**2)
    insqrt = binst * math.log(4 * binst / epsilon)
    j0_val = int(math.ceil(math.sqrt(insqrt)))
    return 8 * j0_val


def _preprocess_basis_worker(queue: multiprocessing.Queue, A: csr_matrix) -> None:
    """Subprocess worker for _preprocess_basis; puts result or exception into queue."""
    import sparseqr
    from scipy.sparse.linalg import lsmr
    try:
        A = csr_matrix(A, dtype=np.float64)
        m, n = A.shape

        _, _, basis_P, effective_rank = sparseqr.qr(A)
        basis_P = np.asarray(basis_P, dtype=np.intp)

        if effective_rank < m:
            _, _, P_row, row_rank = sparseqr.qr(A.T)
            if row_rank != effective_rank:
                raise RankUncertainError(
                    f"SPQR rank estimates disagree: {effective_rank} vs {row_rank}"
                )
            P_row = np.asarray(P_row, dtype=np.intp)
            kept_rows = P_row[:effective_rank]
            dropped_rows = P_row[effective_rank:m]
            A_kept = A[kept_rows, :].tocsr()
            iter_lim = max(100, min(10_000, 10 * max(A_kept.shape)))
            for row_index in dropped_rows:
                dropped = A.getrow(int(row_index)).toarray().ravel()
                try:
                    solution = lsmr(
                        A_kept.T,
                        dropped,
                        atol=1e-12,
                        btol=1e-12,
                        maxiter=iter_lim,
                    )[0]
                    residual = np.linalg.norm(A_kept.T @ solution - dropped)
                except Exception as exc:  # noqa: BLE001 - failed certification is uncertain
                    raise RankUncertainError(
                        f"Could not certify SPQR-dropped row {int(row_index)}"
                    ) from exc
                relative_residual = residual / max(np.linalg.norm(dropped), np.finfo(float).tiny)
                if not math.isfinite(relative_residual) or relative_residual > 1e-6:
                    raise RankUncertainError(
                        f"SPQR-dropped row {int(row_index)} has relative residual "
                        f"{relative_residual:.3e}"
                    )
            A = A_kept
            m = effective_rank

        B = basis_P[:m]
        N_mask = np.ones(n, dtype=bool)
        N_mask[B] = False
        N = np.where(N_mask)[0]
        n_N = len(N)

        queue.put((A, m, n, B, N, n_N, A[:, B].tocsc(), A[:, N]))
    except Exception as exc:  # noqa: BLE001
        queue.put(exc)


def _preprocess_basis(A: csr_matrix):
    """Shared preprocessing for both QIPM variants.

    Returns (A, m, n, B, N, n_N, A_B_lu, A_N) after SPQR basis selection,
    optional rank-deficiency row reduction, and LU factorisation of A_B.
    Raises RuntimeError if preprocessing exceeds _PREPROCESS_TIMEOUT seconds.
    """
    import queue as _queue
    from scipy.sparse.linalg import splu

    q: multiprocessing.Queue = multiprocessing.Queue()
    p = multiprocessing.Process(target=_preprocess_basis_worker, args=(q, A))
    p.start()
    try:
        deadline = time.monotonic() + _PREPROCESS_TIMEOUT
        result = None
        while True:
            try:
                result = q.get(timeout=0.5)
                break
            except _queue.Empty:
                if not p.is_alive():
                    try:
                        result = q.get_nowait()
                        break
                    except _queue.Empty:
                        raise PreprocessCrashedError(
                            f"Basis preprocessing worker crashed with exit code {p.exitcode}"
                        )
                if time.monotonic() >= deadline:
                    raise PreprocessTimeoutError(
                        f"Basis preprocessing exceeded {_PREPROCESS_TIMEOUT // 60}-minute time limit"
                    )
        if isinstance(result, Exception):
            raise result
        A, m, n, B, N, n_N, A_B_csc, A_N = result
        return A, m, n, B, N, n_N, splu(A_B_csc), A_N
    finally:
        q.close()
        q.cancel_join_thread()
        if p.is_alive():
            p.terminate()
            p.join(5)
            if p.is_alive():
                p.kill()
                p.join()
        else:
            p.join()
        p.close()


class _AlarmTimeout(Exception):
    pass


def _sigma_timed(operator, which: str, timeout: int) -> float | None:
    """Run svds for one extreme singular value with a SIGALRM timeout.

    Returns the singular value on convergence, or None on timeout / non-convergence.
    The SIGALRM fires at the next Python callback (i.e. the next F̄ matvec), so
    the effective timeout is ±one-matvec accurate.
    """
    from scipy.sparse.linalg import ArpackNoConvergence, svds

    def _handler(signum, frame):
        raise _AlarmTimeout()

    old = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(timeout)
    try:
        return float(svds(operator, k=1, which=which, return_singular_vectors=False)[0])
    except (_AlarmTimeout, ArpackNoConvergence):
        return None
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


def _sigma_min_random_probes(
    operator_mv, dimension: int, n_probes: int, timeout: float = 60.0
) -> float:
    """Upper bound on σ_min(F̄) via random Rayleigh-quotient probes.

    For any unit w: σ_min(F̄) ≤ ‖F̄w‖ (min-max theorem).
    Returns the minimum over n_probes random Gaussian unit vectors — a valid
    (if potentially loose) upper bound, hence usable in a condition lower bound.
    """
    rng = np.random.default_rng(0)
    ub = np.inf
    deadline = time.monotonic() + timeout
    for _ in range(n_probes):
        w = rng.standard_normal(dimension)
        w /= np.linalg.norm(w)
        ub = min(ub, float(np.linalg.norm(operator_mv(w))))
        if time.monotonic() >= deadline:
            break
    return ub


def _sigma_max_random_probes(
    operator_mv, dimension: int, n_probes: int, timeout: float = 60.0
) -> float:
    """Lower-bound σ_max using the largest norm among random unit probes."""
    rng = np.random.default_rng(1)
    lower_bound = 0.0
    deadline = time.monotonic() + timeout
    for _ in range(n_probes):
        w = rng.standard_normal(dimension)
        w /= np.linalg.norm(w)
        lower_bound = max(lower_bound, float(np.linalg.norm(operator_mv(w))))
        if time.monotonic() >= deadline:
            break
    return lower_bound


def _cycle_count_mnes_from_basis(
    A: csr_matrix,
    m: int,
    n: int,
    B: np.ndarray,
    N: np.ndarray,
    n_N: int,
    A_B_lu,
    A_N: csr_matrix,
) -> tuple[int, int, float]:
    """Compute (cycle_count, sparsity, cond) for mnes from preprocessed basis.

    Uses M̂ = I + F̄F̄ᵀ, so λᵢ(M̂) = 1 + σᵢ(F̄)². Computes σ_max and σ_min of
    F̄ = A_B⁻¹ A_N via svds on a LinearOperator; κ = (1+σ_max²)/(1+σ_min²).

    When n_N < m, F̄ has rank ≤ n_N < m, so F̄F̄ᵀ has a null space and λ_min = 1
    exactly — no second svds call needed.
    """
    from scipy.sparse.linalg import LinearOperator

    s = m  # M̂ is generically dense m×m

    if n_N == 0:
        s = 1
        k = 1.0
    elif A_N.nnz == 0 or m <= 1:
        k = 1.0
    else:
        def _fbar_mv(v: np.ndarray) -> np.ndarray:
            return A_B_lu.solve(np.asarray(A_N @ v, dtype=np.float64).ravel())

        def _fbar_rmv(u: np.ndarray) -> np.ndarray:
            return np.asarray(A_N.T @ A_B_lu.solve(u, trans="T"), dtype=np.float64).ravel()

        if n_N == 1:
            # F̄ is m×1; its only singular value is ‖F̄ e₁‖
            sigma_max = float(np.linalg.norm(_fbar_mv(np.ones(1, dtype=np.float64))))
            lam_max = 1.0 + sigma_max ** 2
            lam_min = 1.0  # n_N = 1 < m → null space of F̄F̄ᵀ is non-trivial
        else:
            F_op = LinearOperator((m, n_N), matvec=_fbar_mv, rmatvec=_fbar_rmv, dtype=np.float64)
            sigma_max = _sigma_timed(F_op, "LM", _MNES_SM_TIMEOUT)
            if sigma_max is None:
                sigma_max = _sigma_max_random_probes(
                    _fbar_mv, n_N, _MNES_N_PROBES, _MNES_SM_TIMEOUT
                )
            lam_max = 1.0 + sigma_max ** 2
            if n_N < m:
                # F̄ has rank ≤ n_N < m → λ_min(M̂) = 1 exactly
                lam_min = 1.0
            else:
                # Try svds("SM") with a wall-clock timeout; Ritz values from
                # converged run are upper bounds on σ_min (interlacing theorem),
                # giving a lower bound on κ.  On timeout or non-convergence fall
                # back to random Rayleigh-quotient probes, which are cheaper but
                # potentially looser upper bounds on σ_min.
                sigma_min = _sigma_timed(F_op, "SM", _MNES_SM_TIMEOUT)
                if sigma_min is None:
                    # Probe the left side. For unit u, ||Fbar.T u|| is an upper
                    # bound on the smallest singular value relevant to Fbar Fbar.T.
                    sigma_min = _sigma_min_random_probes(
                        _fbar_rmv, m, _MNES_N_PROBES, _MNES_SM_TIMEOUT
                    )
                lam_min = 1.0 + sigma_min ** 2

        k = max(lam_max / lam_min, 1.0)

    repetitions = math.ceil((m - 1) / _EPSILON**2)
    count = cycle_count_qlsa(s=s, k=k, epsilon=_EPSILON) * repetitions
    return count, s, k


def _cycle_count_mnes(A: csr_matrix) -> tuple[int, int, float]:
    """Return (cycle_count, sparsity, cond) for mnes.

    Computes κ(M̂) via M̂ = I + F̄F̄ᵀ, F̄ = A_B⁻¹ A_N (D_B = D_N = I); s = m.
    Uses svds on F̄: κ = (1+σ_max²)/(1+σ_min²); λ_min = 1 exactly when n_N < m.
    """
    basis = _preprocess_basis(A)
    if basis[1] < 2:
        raise DegenerateInstanceError("Preprocessing reduced A below two rows")
    return _cycle_count_mnes_from_basis(*basis)


def _cycle_count_oss_from_basis(
    A: csr_matrix,
    m: int,
    n: int,
    B: np.ndarray,
    N: np.ndarray,
    n_N: int,
    A_B_lu,
    A_N: csr_matrix,
) -> tuple[int, int, float]:
    """Compute (cycle_count, sparsity, cond) for oss from preprocessed basis.

    Computes κ(M) = σ_max(M) / σ_min(M) for M = [-Aᵀ | V] ∈ ℝⁿˣⁿ (x = s = 1).
    V ∈ ℝⁿˣ⁽ⁿ⁻ᵐ⁾ is the null-space basis built from the SPQR pivot basis B:
        V[B, :] = -A_B⁻¹ A_N,  V[N, :] = I_{n-m}.

    Both extreme singular-value calls have wall-clock timeouts and fall back to
    random norm probes whose bound directions preserve a lower bound on κ(M).

    Sparsity s = max over rows and columns of M:
    - z_y columns: nnz of column j = nnz of row j of A  → max is max row-nnz(A),
    - z_λ columns: m entries in B-rows + 1 in N-rows     → max is m + 1,
    - B-rows: col-nnz_i(A) entries from -Aᵀ + n_N dense entries from V[B,:]
                                                          → max is max_col-nnz(A_B) + n_N,
    - N-rows: col-nnz_i(A) + 1                           → dominated by the terms above.
    """
    from scipy.sparse.linalg import LinearOperator

    col_nnz = A.getnnz(axis=0)
    s_zy_cols  = int(A.getnnz(axis=1).max()) if A.nnz > 0 else 0  # z_y columns
    if n_N == 0:
        s = max(s_zy_cols, int(col_nnz.max()) if A.nnz > 0 else 0)
    else:
        s_zlam_cols = m + 1
        s_B_rows = (int(col_nnz[B].max()) + n_N) if len(B) > 0 else 0
        s = max(s_zy_cols, s_zlam_cols, s_B_rows)

    # M z = [-Aᵀ z_y + V z_λ]  (x = s = 1)
    def _matvec(z: np.ndarray) -> np.ndarray:
        z = np.asarray(z, dtype=np.float64).ravel()
        z_y, z_lam = z[:m], z[m:]
        out = -np.asarray(A.T @ z_y, dtype=np.float64).ravel()
        if n_N > 0:
            sv = np.empty(n, dtype=np.float64)
            sv[B] = -A_B_lu.solve(np.asarray(A_N @ z_lam, dtype=np.float64).ravel())
            sv[N] = z_lam
            out += sv
        return out

    # Mᵀ u: first m → -A u; last n_N → -A_Nᵀ A_B⁻ᵀ u_B + u_N
    def _rmatvec(u: np.ndarray) -> np.ndarray:
        u = np.asarray(u, dtype=np.float64).ravel()
        out = np.empty(n, dtype=np.float64)
        out[:m] = -np.asarray(A @ u, dtype=np.float64).ravel()
        if n_N > 0:
            out[m:] = -np.asarray(
                A_N.T @ A_B_lu.solve(u[B], trans="T"), dtype=np.float64
            ).ravel() + u[N]
        return out

    M_op = LinearOperator((n, n), matvec=_matvec, rmatvec=_rmatvec, dtype=np.float64)
    sigma_max = _sigma_timed(M_op, "LM", _OSS_SM_TIMEOUT)
    if sigma_max is None:
        sigma_max = _sigma_max_random_probes(
            M_op.matvec, n, _OSS_N_PROBES, _OSS_SM_TIMEOUT
        )
    sigma_min = _sigma_timed(M_op, "SM", _OSS_SM_TIMEOUT)
    if sigma_min is None:
        sigma_min = _sigma_min_random_probes(
            M_op.matvec, n, _OSS_N_PROBES, _OSS_SM_TIMEOUT
        )
    k = max(sigma_max / sigma_min, 1.0)
    repetitions = math.ceil((2 * n - 1) / _EPSILON**2)
    count = cycle_count_qlsa(s=s, k=k, epsilon=_EPSILON) * repetitions
    return count, s, k


def _cycle_count_oss(A: csr_matrix) -> tuple[int, int, float]:
    """Return (cycle_count, sparsity, cond) for oss.

    Computes κ(M) = σ_max/σ_min for M = [-Aᵀ | V] ∈ ℝⁿˣⁿ (x = s = 1).
    Uses svds on M_op with timeout + random probe fallback; result is a lower bound.
    Sparsity s = max(max row-nnz(A), m+1, max col-nnz(A_B) + n_N).
    """
    basis = _preprocess_basis(csr_matrix(A, dtype=np.float64))
    if basis[1] < 2:
        raise DegenerateInstanceError("Preprocessing reduced A below two rows")
    return _cycle_count_oss_from_basis(*basis)


def _load_standard_form(path: Path) -> csr_matrix:
    """Load A from .std standard-form LP (npz format). A returned as CSR."""
    data = np.load(path)
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
    if A_shape.size != 2:
        raise ValueError("A_shape must contain two dimensions")
    m, n = int(A_shape[0]), int(A_shape[1])
    return csr_matrix((A_data, A_indices, A_indptr), shape=(m, n))


def _benchmark_instance_from_path(
    path: Path,
    variant: str = "both",
) -> None:
    """Load A from .std; compute cycle counts, write to instance .data (JSON). path must be .std."""
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Instance file not found: {path}")
    if path.suffix.lower() != ".std":
        raise ValueError(f"Path must be .std; got {path.suffix!r}")
    if variant not in ("mnes", "oss", "both"):
        raise ValueError(f"variant must be 'mnes', 'oss', or 'both'; got {variant!r}")

    base_name = path.name[: -len(".std")]
    data_path = path.parent / (base_name + ".data")
    try:
        data = json.loads(data_path.read_text()) if data_path.exists() else {}
    except (json.JSONDecodeError, UnicodeDecodeError):
        data = {}
    if not isinstance(data, dict):
        data = {}
    active_variants = ["mnes", "oss"] if variant == "both" else [variant]

    def _skip(status: str) -> None:
        for active in active_variants:
            for key in _BENCHMARK_DATA_KEYS[active]:
                data.pop(key, None)
            data[f"status_{active}"] = status
        _atomic_write_json(data_path, data)

    try:
        A = _load_standard_form(path)
    except Exception as exc:  # noqa: BLE001 - persist corrupt-input status per instance
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
        if isinstance(exc, DegenerateInstanceError):
            return "skipped_degenerate"
        return f"error:{type(exc).__name__}"

    def _record_failure(active: str, exc: BaseException) -> None:
        failure_status = _failure_status(exc)
        for key in _BENCHMARK_DATA_KEYS[active]:
            if failure_status == "skipped_degenerate":
                data.pop(key, None)
            else:
                data[key] = None
        data[f"status_{active}"] = failure_status

    def _record_success(active: str, result: tuple[int, int, float]) -> None:
        count, sparsity, cond = result
        cond = max(float(cond), 1.0)
        if not math.isfinite(cond):
            raise ArithmeticError("condition number is not finite")
        data[f"cycle_count_{active}"] = count
        data[f"sparsity_{active}"] = sparsity
        data[f"cond_{active}"] = cond
        data[f"status_{active}"] = "ok"

    if variant == "both":
        try:
            basis = _preprocess_basis(A)
        except (RuntimeError, ValueError, ArithmeticError) as exc:
            _record_failure("mnes", exc)
            _record_failure("oss", exc)
        else:
            if basis[1] < 2:
                _skip("skipped_degenerate")
                return
            try:
                _record_success("mnes", _cycle_count_mnes_from_basis(*basis))
            except (RuntimeError, ValueError, ArithmeticError) as exc:
                _record_failure("mnes", exc)
            try:
                _record_success("oss", _cycle_count_oss_from_basis(*basis))
            except (RuntimeError, ValueError, ArithmeticError) as exc:
                _record_failure("oss", exc)
    else:
        calculate = _cycle_count_mnes if variant == "mnes" else _cycle_count_oss
        try:
            _record_success(variant, calculate(A))
        except (RuntimeError, ValueError, ArithmeticError) as exc:
            _record_failure(variant, exc)

    _atomic_write_json(data_path, data)


def benchmark_instance(
    instance_class: str,
    instance_name: str,
    cache_dir: str | Path | None = None,
    variant: str = "both",
) -> None:
    """Run cycle-count benchmark for the instance in cache_dir/instance_class/instance_name/.

    Discovers the instance by .std (exactly one). Loads A from that file; writes cycle counts to instance .data (JSON).
    instance_class: e.g. "netlib", "miplib".
    instance_name: subfolder name (instance stem).
    cache_dir: root containing instance-class subfolders; defaults to "cache_dir".
    variant: 'mnes', 'oss', or 'both' (default).
    """
    root = Path(cache_dir).resolve() if cache_dir is not None else Path("cache_dir").resolve()
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
    """Run cycle-count benchmark for all .std instances in the given instance-class subfolder of cache_dir.

    instance_class: name of the subfolder (e.g. "netlib", "miplib").
    variant: 'mnes', 'oss', or 'both' (default).
    cache_dir: directory containing instance-class subfolders; defaults to "cache_dir" in the current directory.
    """
    root = Path(cache_dir).resolve() if cache_dir is not None else Path("cache_dir").resolve()
    folder = root / instance_class
    if not folder.is_dir():
        raise FileNotFoundError(f"Instance class folder not found: {folder}")

    subdirs = sorted(d for d in folder.iterdir() if d.is_dir())
    for subdir in tqdm(subdirs, desc=instance_class, unit="instance"):
        if not any(subdir.glob("*.std")):
            tqdm.write(f"skipping {subdir.name}: no .std file")
            continue
        try:
            benchmark_instance(instance_class, subdir.name, cache_dir=root, variant=variant)
        except Exception as exc:  # noqa: BLE001 - isolate corpus instances
            tqdm.write(f"skipping {subdir.name}: {exc}")


def benchmark_all_instance_classes(
    instance_classes: list[str] | None = None,
    variant: str = "both",
    cache_dir: str | Path | None = None,
) -> None:
    """Run cycle-count benchmark for .std instances (main entry point).

    instance_classes: optional list of instance class names (subfolder names under cache_dir).
        If None, all instance classes (all subdirectories of cache_dir) are processed.
    variant: 'mnes', 'oss', or 'both' (default).
    cache_dir: directory containing instance-class subfolders; defaults to "cache_dir" in the current directory.
    """
    root = Path(cache_dir).resolve() if cache_dir is not None else Path("cache_dir").resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Cache directory not found: {root}")

    if instance_classes is None:
        instance_classes = [f.name for f in sorted(root.iterdir()) if f.is_dir()]

    for name in instance_classes:
        benchmark_instance_class(name, variant=variant, cache_dir=root)


_BENCHMARK_DATA_KEYS = {
    "mnes": ("cycle_count_mnes", "sparsity_mnes", "cond_mnes"),
    "oss":  ("cycle_count_oss", "sparsity_oss", "cond_oss"),
}


def show_benchmark_status(
    instance_classes: list[str] | None = None,
    variant: str = "both",
    cache_dir: str | Path | None = None,
) -> None:
    """Print how many instances per class have all benchmark keys present in their .data files.

    For each instance class, prints one line per active variant showing
    "<class>  [mnes: x/total]  [oss: x/total]".
    An instance counts as done when its status is ok. Legacy records count only
    when their cycle-count value is finite.
    """
    root = Path(cache_dir).resolve() if cache_dir is not None else Path("cache_dir").resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Cache directory not found: {root}")

    if instance_classes is None:
        instance_classes = [f.name for f in sorted(root.iterdir()) if f.is_dir()]

    active_variants = ["mnes", "oss"] if variant == "both" else [variant]

    for cls in instance_classes:
        folder = root / cls
        if not folder.is_dir():
            print(f"{cls}: directory not found")
            continue

        subdirs = sorted(d for d in folder.iterdir() if d.is_dir())
        total = len(subdirs)

        counts: dict[str, int] = {v: 0 for v in active_variants}
        for subdir in subdirs:
            data_files = list(subdir.glob("*.data"))
            try:
                data = json.loads(data_files[0].read_text()) if data_files else {}
            except (json.JSONDecodeError, UnicodeDecodeError):
                data = {}
            if not isinstance(data, dict):
                data = {}
            for v in active_variants:
                status_key = f"status_{v}"
                value = data.get(f"cycle_count_{v}")
                legacy_done = (
                    status_key not in data
                    and isinstance(value, (int, float))
                    and math.isfinite(value)
                )
                if data.get(status_key) == "ok" or legacy_done:
                    counts[v] += 1

        parts = "  ".join(f"{v}: {counts[v]}/{total}" for v in active_variants)
        print(f"{cls}:  {parts}")


def clear_benchmark_data(
    instance_classes: list[str] | None = None,
    cache_dir: str | Path | None = None,
    variant: str = "both",
) -> None:
    """Remove benchmark values and statuses from .data files."""
    root = Path(cache_dir).resolve() if cache_dir is not None else Path("cache_dir").resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Cache directory not found: {root}")

    value_keys = (
        _BENCHMARK_DATA_KEYS["mnes"] + _BENCHMARK_DATA_KEYS["oss"]
        if variant == "both"
        else _BENCHMARK_DATA_KEYS[variant]
    )
    active_variants = ("mnes", "oss") if variant == "both" else (variant,)
    keys = value_keys + tuple(f"status_{v}" for v in active_variants)

    search_roots = [root / name for name in instance_classes] if instance_classes else [root]
    for search_root in search_roots:
        for data_path in search_root.rglob("*.data"):
            try:
                data = json.loads(data_path.read_text())
            except (json.JSONDecodeError, UnicodeDecodeError):
                data = {}
            if not isinstance(data, dict):
                data = {}
            if any(k in data for k in keys):
                for k in keys:
                    data.pop(k, None)
                _atomic_write_json(data_path, data)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Benchmark LP instances: compute QIPM cycle counts and write to instance .data (JSON).",
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
        help="Remove benchmark entries from .data files instead of benchmarking. Other flags are ignored.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show how many instances per class have benchmark data for the selected variant(s). Other flags are ignored.",
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
