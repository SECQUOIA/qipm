"""Numerical screening estimates for the QIPM benchmark."""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import math
import multiprocessing
import signal
import time

import numpy as np
# Deliberately fail at module load when sparseqr is missing (BUGFIXES F9).
import sparseqr
from scipy.sparse import csr_matrix

# Per-solve epsilon = 0.1 is justified by iterative refinement in
# Mohammadisiahroudi et al.  The prepared-state and extracted-vector errors add
# by the triangle inequality; this quantum-favourable screen does not charge
# the extra precision that a combined 0.1 error contract would require.
_EPSILON_QLSA = 1e-1  # target l2 error of the prepared solution state
_EPSILON_TOMO = 1e-1  # target l2 error of the extracted classical vector
_PREPROCESS_TIMEOUT = 600  # seconds; basis preprocessing time limit
_MNES_SM_TIMEOUT = 60     # seconds; wall-clock limit for svds("SM") in MNES
_MNES_N_PROBES = 10_000   # random probes for singular-value fallbacks
_OSS_SM_TIMEOUT = 60      # seconds; wall-clock limit for svds("SM") in OSS
_OSS_N_PROBES = 10_000    # random probes for singular-value fallbacks
_SPARSITY_N_SAMPLES = 64  # distinct rows/columns sampled per operator block
_SPARSITY_RELATIVE_THRESHOLD = 1e-12


class RankUncertainError(RuntimeError):
    """SPQR dropped a row whose dependence could not be certified."""


class InconsistentSystemError(RuntimeError):
    """A row dependent in A is inconsistent with the same dependence in b."""


class PreprocessTimeoutError(RuntimeError):
    """Basis preprocessing exceeded its wall-clock limit."""


class PreprocessCrashedError(RuntimeError):
    """Basis preprocessing worker exited without a usable result."""


class DegenerateInstanceError(RuntimeError):
    """Preprocessing reduced the benchmark system below two rows."""


class BasisSingularError(RuntimeError):
    """SuperLU found the SPQR-selected basis to be exactly singular."""


@dataclass(frozen=True)
class BasisSelection:
    """Picklable basis-selection result sent by the preprocessing worker."""

    A: csr_matrix
    b: np.ndarray
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
    b: np.ndarray
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
    count_best_known: int
    count_floor: int
    sparsity: int
    sparsity_method: str
    cond: float
    cond_method: str
    qlsa_queries: int
    qlsa_queries_best_known: int
    qlsa_queries_floor: int
    repetitions: int


def cycle_count_qlsa(
    *,
    s: int,
    k: float,
    epsilon: float = _EPSILON_QLSA,
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
    try:
        sk = float(s) * k
        binst = math.ceil(math.log2(sk / epsilon) * sk**2)
        insqrt = binst * math.log2(4 * binst / epsilon)
        return 8 * int(math.ceil(math.sqrt(insqrt)))
    except OverflowError:
        pass

    k_num, k_den = float(k).as_integer_ratio()
    sk = Fraction(s * k_num, k_den)
    lg1 = math.log2(s * k_num) - math.log2(k_den) - math.log2(epsilon)
    binst = math.ceil(Fraction(lg1) * sk * sk)
    lg2 = math.log2(binst) + 2.0 - math.log2(epsilon)
    insqrt = Fraction(lg2) * binst
    n = max(math.ceil(insqrt), 0)
    result = math.isqrt(n)
    if result * result < n:
        result += 1
    return 8 * result


def _ceil_scaled_condition(k: float, scale: float) -> int:
    """Return ceil(k * scale) without overflowing the intermediate product."""
    if k < 1 or not math.isfinite(k) or scale <= 0 or not math.isfinite(scale):
        raise ValueError("k and scale must be finite and positive, with k at least 1")
    k_num, k_den = k.as_integer_ratio()
    return math.ceil(Fraction(k_num, k_den) * Fraction.from_float(scale))


def _ceil_condition_sqrt_s(*, k: float, s: int) -> int:
    """Return the exact ceiling of k * sqrt(s) for the stored float k."""
    if k < 1 or not math.isfinite(k) or s < 1:
        raise ValueError("k must be finite and at least 1, and s must be positive")
    p, q = k.as_integer_ratio()
    radicand = p * p * s
    q_squared = q * q
    result = math.isqrt(radicand // q_squared)
    while result * result * q_squared < radicand:
        result += 1
    return result


def _query_counts(*, s: int, k: float) -> tuple[int, int, int]:
    """Return modeled, best-known, and scaling-floor QLSA query counts.

    The modeled count is the existing Chebyshev construction.  The best-known
    line uses Dalzell's known-norm kernel-reflection shortcut,
    ceil(kappa ln(2 sqrt(2) / epsilon)).  Substituting the target-matrix kappa
    for the block-encoding kappa_BE = alpha ||M^-1|| >= kappa is deliberately
    quantum-favourable.  The floor line is ceil(kappa sqrt(s)), the proven
    worst-case sparse-access Omega(kappa sqrt(s)) scaling of Mori et al. with
    its unknown Omega constant set to one; it is not a theorem-exact or
    per-instance lower bound.
    """
    modeled = cycle_count_qlsa(s=s, k=k, epsilon=_EPSILON_QLSA)
    best_known_scale = math.log(2.0 * math.sqrt(2.0) / _EPSILON_QLSA)
    # The estimate line must not round the transcendental scale above its value.
    best_known_scale = math.nextafter(best_known_scale, 0.0)
    best_known = _ceil_scaled_condition(k, best_known_scale)
    floor = _ceil_condition_sqrt_s(k=k, s=s)
    return modeled, best_known, floor


def _tomography_repetitions(dimension: int) -> int:
    """Return the state-preparation-unitary readout repetition estimate.

    Pure-state l2 tomography with state-preparation-unitary access costs
    tilde-Theta(d / epsilon) (van Apeldoorn et al., SODA 2023, Theorems 23/52).
    We use ceil((d - 1) / epsilon_tomo), phrased as at or below the cost of every
    known readout protocol rather than as a theorem-exact lower bound because
    the asymptotic notation hides constants and polylogarithms.  Fraction
    arithmetic makes the ceiling exact for the configured decimal epsilon.
    """
    if dimension < 1:
        raise ValueError("tomography dimension must be positive")
    epsilon = Fraction(str(_EPSILON_TOMO))
    return math.ceil(Fraction(dimension - 1, 1) / epsilon)


def _sample_indices(dimension: int, rng: np.random.Generator) -> tuple[np.ndarray, bool]:
    """Return distinct seeded sample indices and whether they enumerate the block."""
    if dimension <= _SPARSITY_N_SAMPLES:
        return np.arange(dimension, dtype=np.intp), True
    indices = rng.choice(dimension, size=_SPARSITY_N_SAMPLES, replace=False)
    return np.sort(indices.astype(np.intp, copy=False)), False


def _significant_nnz(*parts: np.ndarray) -> int:
    """Count entries above one relative threshold across vector parts."""
    arrays = [np.asarray(part, dtype=np.float64).ravel() for part in parts]
    max_abs = max((float(np.max(np.abs(part))) for part in arrays if part.size), default=0.0)
    threshold = _SPARSITY_RELATIVE_THRESHOLD * max_abs
    return sum(int(np.count_nonzero(np.abs(part) > threshold)) for part in arrays)


def _mnes_sparsity(basis: PreparedBasis) -> tuple[int, str]:
    """Estimate max nnz of M-hat by exact enumeration or seeded row sampling.

    A sampled maximum and thresholding tiny entries can only undercount the
    corresponding numerically formed maximum.  The modeled and floor lines grow
    with s, while the best-known line is independent of s, so underestimating s
    never overstates a reported quantity.  This is floating-point numerical
    evidence, not certified arithmetic.
    """
    m, A_N, A_B_lu = basis.m, basis.A_N, basis.A_B_lu
    if basis.n_N == 0 or A_N.nnz == 0:
        return 1, "exact"
    rng = np.random.default_rng(0)
    rows, exact = _sample_indices(m, rng)
    sparsity = 1
    for row_index in rows:
        unit = np.zeros(m, dtype=np.float64)
        unit[int(row_index)] = 1.0
        fbar_t_row = np.asarray(
            A_N.T @ A_B_lu.solve(unit, trans="T"), dtype=np.float64
        ).ravel()
        row = A_B_lu.solve(
            np.asarray(A_N @ fbar_t_row, dtype=np.float64).ravel()
        )
        row[int(row_index)] += 1.0
        sparsity = max(sparsity, _significant_nnz(row))
    return sparsity, "exact" if exact else "sampled"


def _oss_sparsity(basis: PreparedBasis) -> tuple[int, str]:
    """Estimate max row/column nnz of M by blockwise seeded sampling.

    Each row and column is formed matrix-free from A and the selected-basis LU.
    Sampling a maximum and thresholding at 1e-12 times the vector maximum can
    only undercount the numerically formed maximum.  The modeled and floor lines
    grow with s, while the best-known line is independent of s, so the estimate
    never overstates a reported quantity.  The calculation uses ordinary
    floating-point arithmetic, not certified arithmetic.
    """
    A, m = basis.A, basis.m
    B, N, n_N = basis.B, basis.N, basis.n_N
    A_B_lu, A_N = basis.A_B_lu, basis.A_N
    rng = np.random.default_rng(0)
    basic_positions, exact_basic = _sample_indices(m, rng)
    nonbasic_positions, exact_nonbasic = _sample_indices(n_N, rng)
    sparsity = 0

    # z_y columns are negated rows of A.
    for row_index in basic_positions:
        sparsity = max(sparsity, _significant_nnz(A.getrow(int(row_index)).data))

    # z_lambda columns contain -A_B^-1 A_N e_j on B and e_j on N.
    for nonbasic_position in nonbasic_positions:
        rhs = A_N.getcol(int(nonbasic_position)).toarray().ravel()
        basic_part = -A_B_lu.solve(rhs)
        sparsity = max(sparsity, _significant_nnz(basic_part, np.ones(1)))

    A_csc = A.tocsc()
    # B-rows contain column B[p] of -A^T and row p of -A_B^-1 A_N.
    for basic_position in basic_positions:
        variable_index = int(B[int(basic_position)])
        a_part = A_csc.getcol(variable_index).data
        unit = np.zeros(m, dtype=np.float64)
        unit[int(basic_position)] = 1.0
        v_part = -np.asarray(
            A_N.T @ A_B_lu.solve(unit, trans="T"), dtype=np.float64
        ).ravel()
        sparsity = max(sparsity, _significant_nnz(a_part, v_part))

    # N-rows contain column N[j] of -A^T and one unit entry in V.
    for nonbasic_position in nonbasic_positions:
        variable_index = int(N[int(nonbasic_position)])
        a_part = A_csc.getcol(variable_index).data
        sparsity = max(sparsity, _significant_nnz(a_part, np.ones(1)))

    exact = exact_basic and exact_nonbasic
    return max(sparsity, 1), "exact" if exact else "sampled"


def _preprocess_basis_worker(
    queue: multiprocessing.Queue, A: csr_matrix, b: np.ndarray
) -> None:
    """Subprocess worker for _preprocess_basis; puts result or exception into queue."""
    import sparseqr
    from scipy.sparse.linalg import lsmr
    try:
        A = csr_matrix(A, dtype=np.float64)
        # Stored-entry counts and basis selection must see canonical structure
        # so the sparsity estimate remains an underestimate.
        A.sum_duplicates()
        A.eliminate_zeros()
        m, n = A.shape
        b = np.asarray(b, dtype=np.float64).ravel()
        if b.shape != (m,):
            raise ValueError(f"b must have shape ({m},); got {b.shape}")

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
            b_kept = b[kept_rows]
            b_scale = 1e-6 * max(float(np.linalg.norm(b, ord=np.inf)), 1.0)
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
                b_residual = abs(float(b[int(row_index)] - solution @ b_kept))
                # Componentwise backward-error scaling matches the relative
                # A-side gate when the dependence coefficients are large.
                b_tolerance = b_scale * (
                    1.0 + float(np.linalg.norm(solution, ord=1))
                )
                if not math.isfinite(b_residual) or b_residual > b_tolerance:
                    raise InconsistentSystemError(
                        f"SPQR-dropped row {int(row_index)} has right-hand-side "
                        f"residual {b_residual:.3e}"
                    )
            A = A_kept
            b = b_kept
            m = effective_rank

        B = basis_P[:m]
        N_mask = np.ones(n, dtype=bool)
        N_mask[B] = False
        N = np.where(N_mask)[0]
        n_N = len(N)

        queue.put(BasisSelection(A, b, m, n, B, N, n_N, A[:, B].tocsc(), A[:, N]))
    except Exception as exc:  # noqa: BLE001
        queue.put(exc)


def _preprocess_basis(A: csr_matrix, b: np.ndarray) -> PreparedBasis:
    """Shared preprocessing for both QIPM variants.

    Returns a prepared basis after SPQR basis selection,
    optional rank-deficiency row reduction, and LU factorisation of A_B.
    Raises RuntimeError if preprocessing exceeds _PREPROCESS_TIMEOUT seconds.
    """
    import queue as _queue
    from scipy.sparse.linalg import splu

    q: multiprocessing.Queue = multiprocessing.Queue()
    p = multiprocessing.Process(target=_preprocess_basis_worker, args=(q, A, b))
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
        try:
            A_B_lu = splu(result.A_B_csc)
        except RuntimeError as exc:
            if "singular" in str(exc).lower():
                raise BasisSingularError(str(exc)) from exc
            raise
        return PreparedBasis(
            result.A,
            result.b,
            result.m,
            result.n,
            result.B,
            result.N,
            result.n_N,
            A_B_lu,
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

    s, sparsity_method = _mnes_sparsity(basis)

    if n_N == 0:
        k = 1.0
        cond_method = "exact"
    elif A_N.nnz == 0 or m <= 1:
        k = 1.0
        cond_method = "exact"
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
            cond_method = "exact"
        else:
            F_op = LinearOperator((m, n_N), matvec=_fbar_mv, rmatvec=_fbar_rmv, dtype=np.float64)
            sigma_max = _sigma_timed(F_op, "LM", _MNES_SM_TIMEOUT)
            probed_max = sigma_max is None
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
                probed_min = sigma_min is None
                if sigma_min is None:
                    # Probe the left side. For unit u, ||Fbar.T u|| is an upper
                    # bound on the smallest singular value relevant to Fbar Fbar.T.
                    sigma_min = _sigma_min_random_probes(
                        _fbar_rmv, m, _MNES_N_PROBES, _MNES_SM_TIMEOUT
                    )
                lam_min = 1.0 + sigma_min ** 2

            if probed_max and n_N >= m and probed_min:
                cond_method = "probe_both"
            elif probed_max:
                cond_method = "probe_sigma_max"
            elif n_N >= m and probed_min:
                cond_method = "probe_sigma_min"
            else:
                cond_method = "svds"

        k = max(lam_max / lam_min, 1.0)

    repetitions = _tomography_repetitions(m)
    qlsa_queries, qlsa_queries_best_known, qlsa_queries_floor = _query_counts(s=s, k=k)
    count = qlsa_queries * repetitions
    count_best_known = qlsa_queries_best_known * repetitions
    count_floor = qlsa_queries_floor * repetitions
    return CycleCountResult(
        count, count_best_known, count_floor, s, sparsity_method, k, cond_method,
        qlsa_queries, qlsa_queries_best_known, qlsa_queries_floor, repetitions,
    )


def _cycle_count_mnes(A: csr_matrix, b: np.ndarray) -> CycleCountResult:
    """Return the cycle count, sparsity, and condition bound for MNES.

    Computes κ(M̂) via M̂ = I + F̄F̄ᵀ, F̄ = A_B⁻¹ A_N (D_B = D_N = I).
    Uses svds on F̄: κ = (1+σ_max²)/(1+σ_min²); λ_min = 1 exactly when n_N < m.
    """
    basis = _preprocess_basis(A, b)
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

    Sparsity is a seeded sampled maximum over matrix-free rows and columns (or
    exact numerical enumeration when each block fits the sample budget).
    """
    from scipy.sparse.linalg import LinearOperator

    A, m, n = basis.A, basis.m, basis.n
    B, N, n_N = basis.B, basis.N, basis.n_N
    A_B_lu, A_N = basis.A_B_lu, basis.A_N

    s, sparsity_method = _oss_sparsity(basis)

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
    probed_max = sigma_max is None
    if sigma_max is None:
        sigma_max = _sigma_max_random_probes(
            M_op.matvec, n, _OSS_N_PROBES, _OSS_SM_TIMEOUT
        )
    sigma_min = _sigma_timed(M_op, "SM", _OSS_SM_TIMEOUT)
    probed_min = sigma_min is None
    if sigma_min is None:
        sigma_min = _sigma_min_random_probes(
            M_op.matvec, n, _OSS_N_PROBES, _OSS_SM_TIMEOUT
        )
    k = max(sigma_max / sigma_min, 1.0)
    if probed_max and probed_min:
        cond_method = "probe_both"
    elif probed_max:
        cond_method = "probe_sigma_max"
    elif probed_min:
        cond_method = "probe_sigma_min"
    else:
        cond_method = "svds"
    repetitions = _tomography_repetitions(n)
    qlsa_queries, qlsa_queries_best_known, qlsa_queries_floor = _query_counts(s=s, k=k)
    count = qlsa_queries * repetitions
    count_best_known = qlsa_queries_best_known * repetitions
    count_floor = qlsa_queries_floor * repetitions
    return CycleCountResult(
        count, count_best_known, count_floor, s, sparsity_method, k, cond_method,
        qlsa_queries, qlsa_queries_best_known, qlsa_queries_floor, repetitions,
    )


def _cycle_count_oss(A: csr_matrix, b: np.ndarray) -> CycleCountResult:
    """Return the cycle count, sparsity, and condition bound for OSS.

    Computes κ(M) = σ_max/σ_min for M = [-Aᵀ | V] ∈ ℝⁿˣⁿ (x = s = 1).
    Uses svds on M_op with timeout + random probe fallback; result is a lower bound.
    Sparsity is a sampled numerical underestimate of max row/column nnz.
    """
    A = csr_matrix(A, dtype=np.float64)
    basis = _preprocess_basis(A, b)
    if basis.m < 2:
        raise DegenerateInstanceError("Preprocessing reduced A below two rows")
    return _cycle_count_oss_from_basis(basis)
