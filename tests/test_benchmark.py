"""Regression tests for QLSA counts, condition bounds, and statuses."""

import json
import math
import multiprocessing
from pathlib import Path

import numpy as np
import pytest
from scipy.sparse import csr_matrix
import scipy.sparse.linalg as sparse_linalg
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
    BasisSingularError,
    CycleCountResult,
    PreparedBasis,
    RankUncertainError,
    _ceil_condition_sqrt_s,
    _cycle_count_mnes_from_basis,
    _cycle_count_oss_from_basis,
    _mnes_sparsity,
    _oss_sparsity,
    _preprocess_basis,
    _preprocess_basis_worker,
    _query_counts,
    _sigma_min_random_probes,
    cycle_count_qlsa,
)
from standard_form import load_standard_form, write_standard_form
from store import read_ledger


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


def test_read_ledger_treats_oversized_integer_literal_as_invalid(tmp_path: Path) -> None:
    data_path = tmp_path / "huge.data"
    data_path.write_text('{"value":' + "1" * 5000 + "}")
    assert read_ledger(data_path) == ({}, "invalid")


@pytest.mark.parametrize(("s", "k", "expected"), [(1, 1.0, 48), (2, 1.0, 112)])
def test_cycle_count_exact(s: int, k: float, expected: int) -> None:
    assert cycle_count_qlsa(s=s, k=k) == expected


def test_cycle_count_monotone_and_total_domain() -> None:
    base = cycle_count_qlsa(s=2, k=2.0)
    assert cycle_count_qlsa(s=3, k=2.0) > base
    assert cycle_count_qlsa(s=2, k=3.0) > base
    counts = [cycle_count_qlsa(s=2, k=k) for k in (1e150, 1e152, 1e308)]
    assert counts[0] < counts[1] < counts[2]
    assert cycle_count_qlsa(s=2, k=1e308) == counts[2]


@pytest.mark.parametrize("k", [math.inf, math.nan, 0.5])
def test_cycle_count_rejects_invalid_condition(k: float) -> None:
    with pytest.raises(ValueError):
        cycle_count_qlsa(s=1, k=k)


def test_cycle_count_rejects_nonpositive_s() -> None:
    with pytest.raises(ValueError):
        cycle_count_qlsa(s=0, k=1.0)


def test_floor_query_ceiling_uses_exact_integer_arithmetic() -> None:
    assert _ceil_condition_sqrt_s(k=6.363961030678928, s=2) == 9


@pytest.mark.parametrize(("k", "expected"), [(1.0, 4), (2.0, 7), (6.0, 21)])
def test_best_known_query_count_uses_downward_rounded_scale(
    k: float, expected: int
) -> None:
    assert _query_counts(s=2, k=k)[1] == expected


@pytest.mark.parametrize(
    ("k", "expected"),
    [
        (
            1e152,
            "1156643368607391726922799121359002874832769270854683260497779543522460828168011845615410214885352995319089624375215206543922913056505313811337571739967270136",
        ),
        (
            1e308,
            "2329818942515642034590609445500944842331756808055407363830729637541832829450180033369891970399533381494251808380482478564957076520176979476103360344748718452983668529229329003743592304976338622731824286482241473055674327997466348202046124783743324073039694052880167436183769848291314198630171785954173573537860192",
        ),
    ],
)
def test_cycle_count_big_domain_regressions(k: float, expected: str) -> None:
    result = cycle_count_qlsa(s=2, k=k)
    reference = int(expected)
    assert type(result) is int
    assert abs(result - reference) / reference <= 1e-12


@pytest.mark.parametrize(
    "A_dense",
    [
        np.array([[2.0, 0.0, 1.0], [0.0, 3.0, 1.0]]),
        np.array([[1.0, 0.0, 2.0, 1.0, 3.0], [0.0, 1.0, 1.0, 2.0, 1.0]]),
    ],
)
def test_reported_condition_numbers_are_lower_bounds(A_dense: np.ndarray) -> None:
    basis = _preprocess_basis(csr_matrix(A_dense), np.zeros(A_dense.shape[0]))
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
    basis = _preprocess_basis(A, np.zeros(A.shape[0]))
    assert basis.n_N > basis.m
    monkeypatch.setattr(bounds, "_sigma_timed", lambda *args: None)
    result = _cycle_count_mnes_from_basis(basis)
    reported_k = result.cond
    F = basis.A_B_lu.solve(basis.A_N.toarray())
    true_k = np.linalg.cond(np.eye(basis.m) + F @ F.T)
    assert reported_k <= true_k * (1 + 1e-8)
    assert result.cond_method == "probe_both"


def test_empty_nonbasic_partition_uses_exact_shortcuts() -> None:
    A = csr_matrix(np.array([[1.0, 1.0], [0.0, 1.0]]))
    basis = _preprocess_basis(A, np.zeros(A.shape[0]))
    assert basis.n_N == 0
    mnes = _cycle_count_mnes_from_basis(basis)
    oss = _cycle_count_oss_from_basis(basis)
    s_mnes, k_mnes = mnes.sparsity, mnes.cond
    s_oss = oss.sparsity
    assert (s_mnes, k_mnes) == (1, 1.0)
    assert mnes.cond_method == "exact"
    assert s_oss == 2


def test_repetition_counts_and_totals_use_model_2_dimensions(monkeypatch) -> None:
    A = csr_matrix(np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]]))
    basis = _preprocess_basis(A, np.zeros(A.shape[0]))
    mnes = _cycle_count_mnes_from_basis(basis)
    monkeypatch.setattr(bounds, "_sigma_timed", lambda *args: 1.0)
    oss = _cycle_count_oss_from_basis(basis)
    assert mnes.repetitions == 10 * (basis.m - 1)
    assert oss.repetitions == 10 * (basis.n - 1)
    for result in (mnes, oss):
        assert result.count == result.qlsa_queries * result.repetitions
        assert result.count_best_known == (
            result.qlsa_queries_best_known * result.repetitions
        )
        assert result.count_floor == result.qlsa_queries_floor * result.repetitions
        assert result.count_floor <= result.count


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
    _write_std(std_path, np.array([[1.0, 2.0], [1.0, 2.0]]))
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
        _preprocess_basis_worker(queue, csr_matrix(np.ones((3, 4))), np.zeros(3))
        result = queue.get(timeout=1)
    finally:
        queue.close()
        queue.join_thread()
    assert isinstance(result, RankUncertainError)
    assert "rank estimates disagree: 2 vs 1" in str(result)


def test_b_consistency_tolerance_scales_with_large_certificate(monkeypatch) -> None:
    A = csr_matrix(np.array([
        [1e-8, 0.0],
        [0.0, 1e-8],
        [1.0, -1.0],
    ]))

    def fake_qr(matrix):
        if matrix.shape == (3, 2):
            return None, None, np.array([0, 1]), 2
        return None, None, np.array([0, 1, 2]), 2

    monkeypatch.setattr(sparseqr, "qr", fake_qr)
    monkeypatch.setattr(
        sparse_linalg,
        "lsmr",
        lambda *args, **kwargs: (np.array([1e8 + 100.0, -1e8]),),
    )
    queue = multiprocessing.Queue()
    try:
        _preprocess_basis_worker(queue, A, np.array([1.0, 1.0, 0.0]))
        result = queue.get(timeout=1)
    finally:
        queue.close()
        queue.join_thread()
    assert not isinstance(result, Exception)


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


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (BasisSingularError("Factor is exactly singular"), "basis_singular"),
        (RuntimeError("other preprocessing failure"), "error:RuntimeError"),
    ],
)
def test_preprocessing_runtime_failures_have_specific_status(
    error: RuntimeError,
    expected: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    std_path = tmp_path / "failure.std"
    _write_std(std_path, np.eye(2))

    def fail(A, b):
        raise error

    monkeypatch.setattr(benchmark, "_preprocess_basis", fail)
    _benchmark_instance_from_path(std_path, variant="both")

    data = json.loads(std_path.with_suffix(".data").read_text())
    assert data["status_mnes"] == expected
    assert data["status_oss"] == expected


def test_preprocess_wraps_only_singular_superlu_failure(monkeypatch) -> None:
    def singular(matrix):
        raise RuntimeError("Factor is exactly singular")

    monkeypatch.setattr(sparse_linalg, "splu", singular)
    with pytest.raises(BasisSingularError, match="exactly singular"):
        _preprocess_basis(csr_matrix(np.eye(2)), np.ones(2))

    def other_failure(matrix):
        raise RuntimeError("some other superlu failure")

    monkeypatch.setattr(sparse_linalg, "splu", other_failure)
    with pytest.raises(RuntimeError, match="some other superlu failure") as caught:
        _preprocess_basis(csr_matrix(np.eye(2)), np.ones(2))
    assert not isinstance(caught.value, BasisSingularError)


def test_corrupt_std_retracts_stale_benchmark_values(tmp_path: Path) -> None:
    std_path = tmp_path / "corrupt.std"
    std_path.write_bytes(b"truncated")
    (tmp_path / "corrupt.data").write_text(
        json.dumps({
            "cycle_count_mnes": 10,
            "qlsa_queries_mnes": 5,
            "tomography_reps_mnes": 2,
            "status_mnes": "ok",
        })
    )
    _benchmark_instance_from_path(std_path, variant="mnes")
    data = json.loads((tmp_path / "corrupt.data").read_text())
    assert data["status_mnes"].startswith("error:")
    assert "cycle_count_mnes" not in data
    assert "qlsa_queries_mnes" not in data
    assert "tomography_reps_mnes" not in data


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


def test_benchmark_both_records_model_2_hierarchy(tmp_path: Path) -> None:
    std_path = tmp_path / "regular.std"
    _write_std(std_path, np.array([[2.0, 0.0, 1.0], [0.0, 3.0, 1.0]]))
    _benchmark_instance_from_path(std_path, variant="both")
    data = json.loads((tmp_path / "regular.data").read_text())
    assert data["benchmark_model"] == 2
    for variant in ("mnes", "oss"):
        assert data[f"status_{variant}"] == "ok"
        assert data[f"cond_{variant}"] >= 1.0
        assert data[f"sparsity_method_{variant}"] in {"exact", "sampled"}
        repetitions = data[f"tomography_reps_{variant}"]
        for suffix in ("", "_best_known", "_floor"):
            count = data[f"cycle_count{suffix}_{variant}"]
            queries = data[f"qlsa_queries{suffix}_{variant}"]
            assert type(count) is int
            assert count == queries * repetitions
        assert data[f"cycle_count_floor_{variant}"] <= data[f"cycle_count_{variant}"]


def test_both_mode_preserves_mnes_when_oss_fails(tmp_path: Path, monkeypatch) -> None:
    std_path = tmp_path / "partial.std"
    _write_std(std_path, np.eye(2))
    monkeypatch.setattr(
        benchmark,
        "_preprocess_basis",
        lambda A, b: PreparedBasis(
            A, b, 2, 2, np.array([0, 1]), np.array([], dtype=int), 0,
            None, csr_matrix((2, 0)),
        ),
    )
    monkeypatch.setattr(
        benchmark,
        "_cycle_count_mnes_from_basis",
        lambda basis: CycleCountResult(
            count=100,
            count_best_known=20,
            count_floor=8,
            sparsity=2,
            sparsity_method="exact",
            cond=0.9,
            cond_method="exact",
            qlsa_queries=25,
            qlsa_queries_best_known=5,
            qlsa_queries_floor=2,
            repetitions=4,
        ),
    )

    def _fail(basis):
        raise OverflowError("too large")

    monkeypatch.setattr(benchmark, "_cycle_count_oss_from_basis", _fail)
    _benchmark_instance_from_path(std_path, variant="both")
    data = json.loads((tmp_path / "partial.data").read_text())
    assert data["status_mnes"] == "ok"
    assert data["cycle_count_mnes"] == 100
    assert data["cycle_count_best_known_mnes"] == 20
    assert data["cond_mnes"] == 1.0
    assert data["sparsity_method_mnes"] == "exact"
    assert data["status_oss"] == "error:OverflowError"
    assert data["cycle_count_oss"] is None
    assert data["cycle_count_floor_oss"] is None
    assert data["sparsity_method_oss"] is None


def test_show_and_clear_require_model_2(tmp_path: Path, capsys) -> None:
    for name, data in {
        "ok": {
            "benchmark_model": 2,
            "status_mnes": "ok",
            "cycle_count_mnes": 10,
            "qlsa_queries_mnes": 5,
            "tomography_reps_mnes": 2,
        },
        "old_ok": {"status_mnes": "ok", "cycle_count_mnes": 10},
        "legacy": {"cycle_count_mnes": 5},
        "failed": {"benchmark_model": 2, "status_mnes": "timeout"},
    }.items():
        instance_dir = tmp_path / "cls" / name
        instance_dir.mkdir(parents=True)
        (instance_dir / f"{name}.data").write_text(json.dumps(data))
    show_benchmark_status(["cls"], variant="mnes", cache_dir=tmp_path)
    assert "mnes: 1/4 (outdated_model: 2, timeout: 1)" in capsys.readouterr().out
    clear_benchmark_data(["cls"], cache_dir=tmp_path, variant="mnes")
    for data_path in (tmp_path / "cls").rglob("*.data"):
        data = json.loads(data_path.read_text())
        assert "status_mnes" not in data
        assert "cycle_count_mnes" not in data
        assert "qlsa_queries_mnes" not in data
        assert "tomography_reps_mnes" not in data


def test_model_upgrade_single_variant_discards_other_stale_variant(tmp_path: Path) -> None:
    std_path = tmp_path / "upgrade.std"
    _write_std(std_path, np.array([[2.0, 0.0, 1.0], [0.0, 3.0, 1.0]]))
    std_path.with_suffix(".data").write_text(json.dumps({
        "status_oss": "ok",
        "cycle_count_oss": 123,
        "qlsa_queries_oss": 123,
        "tomography_reps_oss": 1,
    }))
    _benchmark_instance_from_path(std_path, variant="mnes")
    data = json.loads(std_path.with_suffix(".data").read_text())
    assert data["benchmark_model"] == 2
    assert data["status_mnes"] == "ok"
    assert "status_oss" not in data
    assert "cycle_count_oss" not in data


def test_clear_model_marker_depends_on_variant_scope(tmp_path: Path) -> None:
    instance_dir = tmp_path / "cls" / "item"
    instance_dir.mkdir(parents=True)
    data_path = instance_dir / "item.data"
    data_path.write_text(json.dumps({
        "benchmark_model": 2,
        "status_mnes": "ok",
        "cycle_count_mnes": 10,
        "status_oss": "ok",
        "cycle_count_oss": 20,
    }))
    clear_benchmark_data(["cls"], cache_dir=tmp_path, variant="mnes")
    data = json.loads(data_path.read_text())
    assert data["benchmark_model"] == 2
    assert "status_mnes" not in data
    assert data["status_oss"] == "ok"

    clear_benchmark_data(["cls"], cache_dir=tmp_path, variant="both")
    data = json.loads(data_path.read_text())
    assert "benchmark_model" not in data
    assert "status_oss" not in data


def _dense_max_significant_nnz(matrix: np.ndarray) -> int:
    counts = []
    for vector in [*matrix, *matrix.T]:
        threshold = 1e-12 * np.max(np.abs(vector))
        counts.append(int(np.count_nonzero(np.abs(vector) > threshold)))
    return max(counts)


def _dense_operators(basis: PreparedBasis) -> tuple[np.ndarray, np.ndarray]:
    F = basis.A_B_lu.solve(basis.A_N.toarray())
    mnes = np.eye(basis.m) + F @ F.T
    V = np.empty((basis.n, basis.n_N))
    V[basis.B, :] = -F
    V[basis.N, :] = np.eye(basis.n_N)
    return mnes, np.column_stack((-basis.A.toarray().T, V))


def _structured_sparse_fixture(m: int) -> csr_matrix:
    rng = np.random.default_rng(17 + m)
    banded = np.eye(m) + np.diag(np.linspace(0.1, 0.4, m - 1), 1)
    dense = rng.normal(size=(m, 4))
    dense[:, 0] *= 4.0
    sparse_column = np.zeros((m, 1))
    sparse_column[[0, m // 2, m - 1], 0] = (2.0, -1.0, 3.0)
    return csr_matrix(np.column_stack((banded, dense, sparse_column)))


def test_exact_sparsity_matches_dense_operators() -> None:
    rng = np.random.default_rng(7)
    A = np.column_stack((np.eye(5), rng.normal(size=(5, 4))))
    basis = _preprocess_basis(csr_matrix(A), np.zeros(5))
    mnes_dense, oss_dense = _dense_operators(basis)
    assert _mnes_sparsity(basis) == (
        _dense_max_significant_nnz(mnes_dense), "exact"
    )
    assert _oss_sparsity(basis) == (
        _dense_max_significant_nnz(oss_dense), "exact"
    )


def test_sampled_sparsity_does_not_exceed_dense_maximum() -> None:
    rng = np.random.default_rng(9)
    A = np.column_stack((np.eye(70), rng.normal(size=(70, 8))))
    basis = _preprocess_basis(csr_matrix(A), np.zeros(70))
    mnes_dense, oss_dense = _dense_operators(basis)
    mnes_s, mnes_method = _mnes_sparsity(basis)
    oss_s, oss_method = _oss_sparsity(basis)
    assert mnes_method == oss_method == "sampled"
    assert mnes_s <= _dense_max_significant_nnz(mnes_dense)
    assert oss_s <= _dense_max_significant_nnz(oss_dense)


@pytest.mark.parametrize(("m", "method"), [(8, "exact"), (70, "sampled")])
def test_structured_sparsity_matches_dense_operator_direction(
    m: int, method: str
) -> None:
    A = _structured_sparse_fixture(m)
    basis = _preprocess_basis(A, np.zeros(m))
    assert not np.array_equal(basis.B, np.arange(m))
    mnes_dense, oss_dense = _dense_operators(basis)
    mnes_s, mnes_method = _mnes_sparsity(basis)
    oss_s, oss_method = _oss_sparsity(basis)
    assert mnes_method == oss_method == method
    if method == "exact":
        assert mnes_s == _dense_max_significant_nnz(mnes_dense)
        assert oss_s == _dense_max_significant_nnz(oss_dense)
    else:
        assert mnes_s <= _dense_max_significant_nnz(mnes_dense)
        assert oss_s <= _dense_max_significant_nnz(oss_dense)


def test_preprocessing_canonicalizes_duplicate_csr_entries() -> None:
    A = csr_matrix(
        (
            np.array([0.2, 0.2, 0.2, 0.2, 0.2, 1.0]),
            np.array([0, 0, 0, 0, 0, 1]),
            np.array([0, 5, 6]),
        ),
        shape=(2, 2),
    )
    assert A.nnz == 6
    basis = _preprocess_basis(A, np.zeros(2))
    assert basis.A.nnz == 2
    assert _oss_sparsity(basis) == (1, "exact")


def test_mnes_zero_nonbasic_matrix_has_unit_sparsity() -> None:
    A = csr_matrix(np.column_stack((np.eye(3), np.zeros((3, 2)))))
    basis = _preprocess_basis(A, np.zeros(3))
    assert basis.A_N.nnz == 0
    result = _cycle_count_mnes_from_basis(basis)
    assert (result.sparsity, result.sparsity_method) == (1, "exact")


def test_inconsistent_dropped_row_is_recorded(tmp_path: Path) -> None:
    A = np.array([[5.0, 0.0], [0.0, 2.0], [0.0, 2.0]])
    std_path = tmp_path / "inconsistent.std"
    sparse = csr_matrix(A)
    write_standard_form(
        std_path,
        np.ones(sparse.shape[1]),
        np.array([3.0, 4.0, 5.0]),
        sparse,
    )
    _benchmark_instance_from_path(std_path, variant="both")
    data = json.loads((tmp_path / "inconsistent.data").read_text())
    assert data["status_mnes"] == "inconsistent_rows"
    assert data["status_oss"] == "inconsistent_rows"
