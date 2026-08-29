"""Numerical lower-bound calculations for the QIPM benchmark."""
from __future__ import annotations

from dataclasses import dataclass
import math
import multiprocessing
import signal
import time

import numpy as np
# Deliberately fail at module load when sparseqr is missing (BUGFIXES F9).
import sparseqr
from scipy.sparse import csr_matrix

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


@dataclass(frozen=True)
class BasisSelection:
    """Picklable basis-selection result sent by the preprocessing worker."""

    A: csr_matrix
    m: int
    n: int
    B: np.ndarray
    N: np.ndarray
    n_N: int
    A_B_csc: object
    A_N: csr_matrix


@dataclass(frozen=True)
class PreparedBasis:
    """Selected basis with the parent-side, non-picklable LU factorization."""

    A: csr_matrix
    m: int
    n: int
    B: np.ndarray
    N: np.ndarray
    n_N: int
    A_B_lu: object
    A_N: csr_matrix


@dataclass(frozen=True)
class CycleCountResult:
    count: int
    sparsity: int
    cond: float


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

        queue.put(BasisSelection(A, m, n, B, N, n_N, A[:, B].tocsc(), A[:, N]))
    except Exception as exc:  # noqa: BLE001
        queue.put(exc)


def _preprocess_basis(A: csr_matrix) -> PreparedBasis:
    """Shared preprocessing for both QIPM variants.

    Returns a prepared basis after SPQR basis selection,
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
        return PreparedBasis(
            result.A,
            result.m,
            result.n,
            result.B,
            result.N,
            result.n_N,
            splu(result.A_B_csc),
            result.A_N,
        )
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


def _cycle_count_mnes_from_basis(basis: PreparedBasis) -> CycleCountResult:
    """Compute the cycle count, sparsity, and condition bound for MNES.

    Uses M̂ = I + F̄F̄ᵀ, so λᵢ(M̂) = 1 + σᵢ(F̄)². Computes σ_max and σ_min of
    F̄ = A_B⁻¹ A_N via svds on a LinearOperator; κ = (1+σ_max²)/(1+σ_min²).

    When n_N < m, F̄ has rank ≤ n_N < m, so F̄F̄ᵀ has a null space and λ_min = 1
    exactly — no second svds call needed.
    """
    from scipy.sparse.linalg import LinearOperator

    m, n_N = basis.m, basis.n_N
    A_B_lu, A_N = basis.A_B_lu, basis.A_N

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
    return CycleCountResult(count, s, k)


def _cycle_count_mnes(A: csr_matrix) -> CycleCountResult:
    """Return the cycle count, sparsity, and condition bound for MNES.

    Computes κ(M̂) via M̂ = I + F̄F̄ᵀ, F̄ = A_B⁻¹ A_N (D_B = D_N = I); s = m.
    Uses svds on F̄: κ = (1+σ_max²)/(1+σ_min²); λ_min = 1 exactly when n_N < m.
    """
    basis = _preprocess_basis(A)
    if basis.m < 2:
        raise DegenerateInstanceError("Preprocessing reduced A below two rows")
    return _cycle_count_mnes_from_basis(basis)


def _cycle_count_oss_from_basis(basis: PreparedBasis) -> CycleCountResult:
    """Compute the cycle count, sparsity, and condition bound for OSS.

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

    A, m, n = basis.A, basis.m, basis.n
    B, N, n_N = basis.B, basis.N, basis.n_N
    A_B_lu, A_N = basis.A_B_lu, basis.A_N

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
    return CycleCountResult(count, s, k)


def _cycle_count_oss(A: csr_matrix) -> CycleCountResult:
    """Return the cycle count, sparsity, and condition bound for OSS.

    Computes κ(M) = σ_max/σ_min for M = [-Aᵀ | V] ∈ ℝⁿˣⁿ (x = s = 1).
    Uses svds on M_op with timeout + random probe fallback; result is a lower bound.
    Sparsity s = max(max row-nnz(A), m+1, max col-nnz(A_B) + n_N).
    """
    basis = _preprocess_basis(csr_matrix(A, dtype=np.float64))
    if basis.m < 2:
        raise DegenerateInstanceError("Preprocessing reduced A below two rows")
    return _cycle_count_oss_from_basis(basis)
