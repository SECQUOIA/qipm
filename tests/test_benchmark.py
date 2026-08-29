"""Regression tests for QLSA counts, condition bounds, and statuses."""

import json
import math
import multiprocessing
from pathlib import Path

import numpy as np
import pytest
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import lsmr, lsqr

import sparseqr
import benchmark
import bounds
from benchmark import (
    _benchmark_instance_from_path,
    clear_benchmark_data,
    show_benchmark_status,
)
from bounds import (
    CycleCountResult,
    PreparedBasis,
    RankUncertainError,
    _cycle_count_mnes_from_basis,
    _cycle_count_oss_from_basis,
    _preprocess_basis,
    _preprocess_basis_worker,
    _sigma_min_random_probes,
    cycle_count_qlsa,
)
from standard_form import load_standard_form, write_standard_form


def _write_std(path: Path, A: np.ndarray) -> None:
    sparse = csr_matrix(A, dtype=np.float64)
    write_standard_form(
        path,
        np.ones(sparse.shape[1]),
        np.ones(sparse.shape[0]),
        sparse,
    )


def _write_raw_std(path: Path, **overrides: np.ndarray) -> None:
    """Handcraft the disk layout independently of the production writer."""
    arrays = {
        "c": np.ones(2),
        "b": np.ones(2),
        "A_data": np.ones(2),
        "A_indices": np.array([0, 1]),
        "A_indptr": np.array([0, 1, 2]),
        "A_shape": np.array([2, 2]),
        "obj_offset": np.array(0.0),
    }
    arrays.update(overrides)
    with path.open("wb") as stream:
        np.savez_compressed(stream, **arrays)


def test_handcrafted_standard_form_layout_remains_compatible(tmp_path: Path) -> None:
    std_path = tmp_path / "layout.std"
    _write_raw_std(std_path)
    c, b, A, obj_offset = load_standard_form(std_path)
    np.testing.assert_array_equal(c, np.ones(2))
    np.testing.assert_array_equal(b, np.ones(2))
    np.testing.assert_array_equal(A.toarray(), np.eye(2))
    assert obj_offset == 0.0


@pytest.mark.parametrize(("s", "k", "expected"), [(1, 1.0, 32), (2, 1.0, 72)])
def test_cycle_count_exact(s: int, k: float, expected: int) -> None:
    assert cycle_count_qlsa(s=s, k=k) == expected


def test_cycle_count_monotone_and_guards_overflow() -> None:
    base = cycle_count_qlsa(s=2, k=2.0)
    assert cycle_count_qlsa(s=3, k=2.0) > base
    assert cycle_count_qlsa(s=2, k=3.0) > base
    with pytest.raises((OverflowError, ValueError, ArithmeticError)):
        cycle_count_qlsa(s=2, k=1e308)
    with pytest.raises(ValueError):
        cycle_count_qlsa(s=1, k=math.inf)


@pytest.mark.parametrize(
    "A_dense",
    [
        np.array([[2.0, 0.0, 1.0], [0.0, 3.0, 1.0]]),
        np.array([[1.0, 0.0, 2.0, 1.0, 3.0], [0.0, 1.0, 1.0, 2.0, 1.0]]),
    ],
)
def test_reported_condition_numbers_are_lower_bounds(A_dense: np.ndarray) -> None:
    basis = _preprocess_basis(csr_matrix(A_dense))
    A, m, n = basis.A, basis.m, basis.n
    B, N, n_N = basis.B, basis.N, basis.n_N
    A_B_lu, A_N = basis.A_B_lu, basis.A_N
    k_mnes = _cycle_count_mnes_from_basis(basis).cond
    F = A_B_lu.solve(A_N.toarray()) if n_N else np.empty((m, 0))
    true_mnes = np.linalg.cond(np.eye(m) + F @ F.T)
    assert 1.0 <= k_mnes <= true_mnes * (1 + 1e-8)

    k_oss = _cycle_count_oss_from_basis(basis).cond
    V = np.empty((n, n_N))
    if n_N:
        V[B, :] = -F
        V[N, :] = np.eye(n_N)
    M = np.column_stack((-A.toarray().T, V))
    true_oss = np.linalg.cond(M)
    assert 1.0 <= k_oss <= true_oss * (1 + 1e-8)


def test_wide_fbar_left_probe_is_valid_upper_bound() -> None:
    F = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 2.0, 0.0, 0.0]])
    probe = _sigma_min_random_probes(F.T.__matmul__, F.shape[0], 1000)
    true_smallest = np.linalg.svd(F, compute_uv=False)[-1]
    assert probe >= true_smallest - 1e-12


def test_mnes_production_fallback_probes_wide_fbar_on_left(monkeypatch) -> None:
    A = csr_matrix(
        np.array(
            [
                [1.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 1.0, 0.0, 0.0],
            ]
        )
    )
    basis = _preprocess_basis(A)
    assert basis.n_N > basis.m
    monkeypatch.setattr(bounds, "_sigma_timed", lambda *args: None)
    reported_k = _cycle_count_mnes_from_basis(basis).cond
    F = basis.A_B_lu.solve(basis.A_N.toarray())
    true_k = np.linalg.cond(np.eye(basis.m) + F @ F.T)
    assert reported_k <= true_k * (1 + 1e-8)


def test_empty_nonbasic_partition_uses_exact_shortcuts() -> None:
    A = csr_matrix(np.array([[1.0, 1.0], [0.0, 1.0]]))
    basis = _preprocess_basis(A)
    assert basis.n_N == 0
    mnes = _cycle_count_mnes_from_basis(basis)
    oss = _cycle_count_oss_from_basis(basis)
    s_mnes, k_mnes = mnes.sparsity, mnes.cond
    s_oss = oss.sparsity
    assert (s_mnes, k_mnes) == (1, 1.0)
    assert s_oss == 2


def test_repetition_counts_use_ceiling_at_both_sites(monkeypatch) -> None:
    A = csr_matrix(np.eye(2))
    basis = PreparedBasis(
        A,
        2,
        2,
        np.array([0, 1]),
        np.array([], dtype=int),
        0,
        None,
        csr_matrix((2, 0)),
    )
    mnes_count = _cycle_count_mnes_from_basis(basis).count
    monkeypatch.setattr(bounds, "_sigma_timed", lambda *args: 1.0)
    oss_count = _cycle_count_oss_from_basis(basis).count
    qlsa_count = cycle_count_qlsa(s=1, k=1.0)
    assert mnes_count == qlsa_count * 100
    assert oss_count == qlsa_count * 300


def test_degenerate_instance_is_skipped_without_counts(tmp_path: Path) -> None:
    std_path = tmp_path / "tiny.std"
    _write_std(std_path, np.array([[1.0]]))
    _benchmark_instance_from_path(std_path, variant="both")
    data = json.loads((tmp_path / "tiny.data").read_text())
    assert data["status_mnes"] == "skipped_degenerate"
    assert data["status_oss"] == "skipped_degenerate"
    assert "cycle_count_mnes" not in data
    assert "cycle_count_oss" not in data


@pytest.mark.parametrize("variant", ["mnes", "oss", "both"])
def test_post_spqr_degenerate_instance_is_skipped_without_counts(
    variant: str, tmp_path: Path
) -> None:
    std_path = tmp_path / "rank_one.std"
    _write_std(std_path, np.array([[1.0, 2.0], [2.0, 4.0]]))
    _benchmark_instance_from_path(std_path, variant=variant)
    data = json.loads((tmp_path / "rank_one.data").read_text())
    active = ("mnes", "oss") if variant == "both" else (variant,)
    for active_variant in active:
        assert data[f"status_{active_variant}"] == "skipped_degenerate"
        assert f"cycle_count_{active_variant}" not in data


def test_rank_reduction_certifies_duplicate_row(tmp_path: Path) -> None:
    std_path = tmp_path / "duplicate.std"
    _write_std(std_path, np.array([[5.0, 0.0], [0.0, 2.0], [0.0, 2.0]]))
    _benchmark_instance_from_path(std_path, variant="both")
    data = json.loads((tmp_path / "duplicate.data").read_text())
    assert data["status_mnes"] == "ok"
    assert data["status_oss"] == "ok"


def test_lsmr_certifies_difficult_duplicate_row(tmp_path: Path) -> None:
    rng = np.random.default_rng(3)
    k = int(rng.integers(3, 30))
    assert k == 24
    B = rng.standard_normal((k, k + 2)) * np.exp(rng.standard_normal((k, 1)) * 2)
    A = np.vstack([B, B[0]])

    sparse_A = csr_matrix(A)
    _, _, P_row, effective_rank = sparseqr.qr(sparse_A.T)
    kept = sparse_A[np.asarray(P_row)[:effective_rank], :]
    dropped = sparse_A.getrow(int(np.asarray(P_row)[effective_rank])).toarray().ravel()
    iteration_limit = max(100, min(10_000, 10 * max(kept.shape)))
    old_solution = lsqr(
        kept.T, dropped, atol=1e-10, btol=1e-10, iter_lim=iteration_limit
    )[0]
    new_solution = lsmr(
        kept.T, dropped, atol=1e-12, btol=1e-12, maxiter=iteration_limit
    )[0]
    old_relative_residual = np.linalg.norm(kept.T @ old_solution - dropped) / np.linalg.norm(dropped)
    new_relative_residual = np.linalg.norm(kept.T @ new_solution - dropped) / np.linalg.norm(dropped)
    assert old_relative_residual > 1e-8
    assert new_relative_residual < 1e-6

    std_path = tmp_path / "difficult_duplicate.std"
    _write_std(std_path, A)
    _benchmark_instance_from_path(std_path, variant="mnes")
    data = json.loads((tmp_path / "difficult_duplicate.data").read_text())
    assert data["status_mnes"] == "ok"


def test_preprocess_worker_rejects_disagreeing_spqr_ranks(monkeypatch) -> None:
    def fake_qr(matrix):
        rank = 2 if matrix.shape == (3, 4) else 1
        return None, None, np.arange(matrix.shape[1]), rank

    monkeypatch.setattr(sparseqr, "qr", fake_qr)
    queue = multiprocessing.Queue()
    try:
        _preprocess_basis_worker(queue, csr_matrix(np.ones((3, 4))))
        result = queue.get(timeout=1)
    finally:
        queue.close()
        queue.join_thread()
    assert isinstance(result, RankUncertainError)
    assert "rank estimates disagree: 2 vs 1" in str(result)


def test_badly_scaled_rank_is_recorded_uncertain(tmp_path: Path) -> None:
    std_path = tmp_path / "uncertain.std"
    _write_std(std_path, np.diag([5e14, 2.0]))
    _benchmark_instance_from_path(std_path, variant="both")
    data = json.loads((tmp_path / "uncertain.data").read_text())
    assert data["status_mnes"] == "rank_uncertain"
    assert data["status_oss"] == "rank_uncertain"


def test_too_large_instance_records_skip(tmp_path: Path) -> None:
    std_path = tmp_path / "large.std"
    A = csr_matrix(([1.0, 1.0], ([0, 100_000], [0, 1])), shape=(100_001, 2))
    _write_std(std_path, A)
    _benchmark_instance_from_path(std_path, variant="both")
    data = json.loads((tmp_path / "large.data").read_text())
    assert data["status_mnes"] == "skipped_too_large"
    assert data["status_oss"] == "skipped_too_large"
    assert "cycle_count_mnes" not in data
    assert "cycle_count_oss" not in data


def test_corrupt_std_retracts_stale_benchmark_values(tmp_path: Path) -> None:
    std_path = tmp_path / "corrupt.std"
    std_path.write_bytes(b"truncated")
    (tmp_path / "corrupt.data").write_text(
        json.dumps({"cycle_count_mnes": 10, "status_mnes": "ok"})
    )
    _benchmark_instance_from_path(std_path, variant="mnes")
    data = json.loads((tmp_path / "corrupt.data").read_text())
    assert data["status_mnes"].startswith("error:")
    assert "cycle_count_mnes" not in data


def test_fractional_indices_are_rejected_by_benchmark_loader(tmp_path: Path) -> None:
    std_path = tmp_path / "fractional.std"
    _write_raw_std(std_path, A_indices=np.array([0.9, 1.0]))
    _benchmark_instance_from_path(std_path, variant="mnes")
    data = json.loads((tmp_path / "fractional.data").read_text())
    assert data["status_mnes"] == "error:ValueError"
    assert "cycle_count_mnes" not in data


def test_out_of_range_indices_are_rejected_by_benchmark_loader(tmp_path: Path) -> None:
    std_path = tmp_path / "out_of_range.std"
    _write_raw_std(std_path, A_indices=np.array([0, 2]))
    _benchmark_instance_from_path(std_path, variant="mnes")
    data = json.loads(std_path.with_suffix(".data").read_text())
    assert data["status_mnes"] == "error:ValueError"
    assert "cycle_count_mnes" not in data


def test_benchmark_both_records_success(tmp_path: Path) -> None:
    std_path = tmp_path / "regular.std"
    _write_std(std_path, np.array([[2.0, 0.0, 1.0], [0.0, 3.0, 1.0]]))
    _benchmark_instance_from_path(std_path, variant="both")
    data = json.loads((tmp_path / "regular.data").read_text())
    for variant in ("mnes", "oss"):
        assert data[f"status_{variant}"] == "ok"
        assert data[f"cycle_count_{variant}"] > 0
        assert data[f"cond_{variant}"] >= 1.0


def test_both_mode_preserves_mnes_when_oss_fails(tmp_path: Path, monkeypatch) -> None:
    std_path = tmp_path / "partial.std"
    _write_std(std_path, np.eye(2))
    monkeypatch.setattr(
        benchmark,
        "_preprocess_basis",
        lambda A: PreparedBasis(A, 2, 2, None, None, 0, None, None),
    )
    monkeypatch.setattr(
        benchmark,
        "_cycle_count_mnes_from_basis",
        lambda basis: CycleCountResult(100, 2, 0.9),
    )

    def _fail(basis):
        raise OverflowError("too large")

    monkeypatch.setattr(benchmark, "_cycle_count_oss_from_basis", _fail)
    _benchmark_instance_from_path(std_path, variant="both")
    data = json.loads((tmp_path / "partial.data").read_text())
    assert data["status_mnes"] == "ok"
    assert data["cycle_count_mnes"] == 100
    assert data["cond_mnes"] == 1.0
    assert data["status_oss"] == "error:OverflowError"
    assert data["cycle_count_oss"] is None


def test_show_and_clear_use_status_with_legacy_fallback(tmp_path: Path, capsys) -> None:
    for name, data in {
        "ok": {"status_mnes": "ok", "cycle_count_mnes": 10},
        "failed": {"status_mnes": "timeout", "cycle_count_mnes": 10},
        "legacy": {"cycle_count_mnes": 5},
        "bad_legacy": {"cycle_count_mnes": float("inf")},
    }.items():
        instance_dir = tmp_path / "cls" / name
        instance_dir.mkdir(parents=True)
        (instance_dir / f"{name}.data").write_text(json.dumps(data))
    show_benchmark_status(["cls"], variant="mnes", cache_dir=tmp_path)
    assert "mnes: 2/4" in capsys.readouterr().out
    clear_benchmark_data(["cls"], cache_dir=tmp_path, variant="mnes")
    for data_path in (tmp_path / "cls").rglob("*.data"):
        data = json.loads(data_path.read_text())
        assert "status_mnes" not in data
        assert "cycle_count_mnes" not in data
