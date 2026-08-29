"""Regression tests for presolve and standard-form conversion."""

import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from scipy.sparse import csr_matrix

import highspy
import transform
from standard_form import (
    _HIGHS_INF,
    _lp_to_standard_form,
    _strip_zero_rows,
    load_standard_form,
)
from transform import (
    clear_std_files,
    transform_instance,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures"
SURVIVING_STEMS = ["surviving_mixed", "surviving_range"]
DOWNSTREAM_DATA = {
    "cycle_count_mnes": 10,
    "sparsity_mnes": 2,
    "cond_mnes": 3.0,
    "status_mnes": "ok",
    "cycle_count_oss": 20,
    "sparsity_oss": 4,
    "cond_oss": 5.0,
    "status_oss": "ok",
    "runtime_highs_std": 0.1,
    "solve_status_std": "ok",
}
SURVIVING_DATA = {
    "runtime_glpk": 1.0,
    "runtime_highs_mps": 0.2,
    "solve_status_mps": "ok",
}


def _load_std(path: Path) -> tuple[np.ndarray, np.ndarray, csr_matrix, float]:
    return load_standard_form(path)


def _highs_objective(path: Path) -> float:
    highs = highspy.Highs()
    highs.setOptionValue("log_to_console", False)
    assert highs.readModel(str(path)) in (highspy.HighsStatus.kOk, highspy.HighsStatus.kWarning)
    assert highs.run() in (highspy.HighsStatus.kOk, highspy.HighsStatus.kWarning)
    assert highs.getModelStatus() == highspy.HighsModelStatus.kOptimal
    return float(highs.getObjectiveValue())


def _std_objective(path: Path) -> float:
    c, b, A, offset = _load_std(path)
    m, n = A.shape
    highs = highspy.Highs()
    highs.setOptionValue("log_to_console", False)
    highs.addVars(n, np.zeros(n), np.full(n, highspy.kHighsInf))
    highs.changeColsCost(n, np.arange(n, dtype=np.int64), c)
    highs.addRows(
        m,
        b,
        b,
        A.nnz,
        np.asarray(A.indptr[:-1], dtype=np.int32),
        np.asarray(A.indices, dtype=np.int32),
        A.data,
    )
    assert highs.run() in (highspy.HighsStatus.kOk, highspy.HighsStatus.kWarning)
    assert highs.getModelStatus() == highspy.HighsModelStatus.kOptimal
    return float(highs.getObjectiveValue()) + offset


@pytest.mark.parametrize("stem", SURVIVING_STEMS)
def test_transform_survives_presolve_and_preserves_objective(stem: str, tmp_path: Path) -> None:
    instance_dir = tmp_path / "cls" / stem
    instance_dir.mkdir(parents=True)
    mps_path = instance_dir / f"{stem}.mps"
    shutil.copy(FIXTURES / f"{stem}.mps", mps_path)

    transform_instance("cls", stem, cache_dir=tmp_path)

    std_path = instance_dir / f"{stem}.std"
    assert std_path.is_file()
    _, _, A, _ = _load_std(std_path)
    highs = highspy.Highs()
    highs.setOptionValue("log_to_console", False)
    highs.readModel(str(mps_path))
    original_lp = highs.getLp()
    highs.presolve()
    presolved_lp = highs.getPresolvedLp()
    matrix = presolved_lp.a_matrix_
    _, expected_b, expected_A, _ = _lp_to_standard_form(
        presolved_lp.num_col_,
        presolved_lp.num_row_,
        presolved_lp.col_cost_,
        presolved_lp.col_lower_,
        presolved_lp.col_upper_,
        presolved_lp.row_lower_,
        presolved_lp.row_upper_,
        matrix.start_,
        matrix.index_,
        matrix.value_,
        presolved_lp.offset_,
    )
    expected_stripped = _strip_zero_rows(expected_A, expected_b)
    assert expected_stripped is not None
    assert A.shape == expected_stripped[0].shape
    if stem == "surviving_mixed":
        assert (presolved_lp.num_row_, presolved_lp.num_col_) != (
            original_lp.num_row_,
            original_lp.num_col_,
        )
    assert A.shape[0] >= 2 and A.shape[1] >= 2
    assert json.loads((instance_dir / f"{stem}.data").read_text())["transform_status"] == "ok"
    assert _std_objective(std_path) == pytest.approx(_highs_objective(mps_path), abs=1e-8)


def test_transform_records_reduced_to_empty(tmp_path: Path) -> None:
    instance_dir = tmp_path / "cls" / "equality"
    instance_dir.mkdir(parents=True)
    shutil.copy(FIXTURES / "equality.mps", instance_dir / "equality.mps")
    transform_instance("cls", "equality", cache_dir=tmp_path)
    assert not (instance_dir / "equality.std").exists()
    assert json.loads((instance_dir / "equality.data").read_text())["transform_status"] == "reduced_to_empty"


def test_zero_row_with_nonzero_rhs_is_infeasible(tmp_path: Path) -> None:
    instance_dir = tmp_path / "cls" / "badrow"
    instance_dir.mkdir(parents=True)
    (instance_dir / "badrow.mps").write_text(
        """NAME BADROW
ROWS
 N OBJ
 E BAD
COLUMNS
    X OBJ 1
RHS
    RHS1 BAD 1
BOUNDS
 LO BND X 0
ENDATA
"""
    )
    transform_instance("cls", "badrow", cache_dir=tmp_path)
    assert not (instance_dir / "badrow.std").exists()
    assert json.loads((instance_dir / "badrow.data").read_text())["transform_status"] == "infeasible"


def test_strip_zero_rows_checks_rhs() -> None:
    A = csr_matrix(np.array([[1.0, 0.0], [0.0, 0.0]]))
    assert _strip_zero_rows(A, np.array([1.0, 2.0])) is None
    stripped = _strip_zero_rows(A, np.array([1.0, 0.0]))
    assert stripped is not None
    stripped_A, stripped_b = stripped
    np.testing.assert_allclose(stripped_A.toarray(), [[1.0, 0.0]])
    np.testing.assert_allclose(stripped_b, [1.0])


def test_transform_recovers_from_corrupt_existing_data(tmp_path: Path) -> None:
    stem = "surviving_mixed"
    instance_dir = tmp_path / "cls" / stem
    instance_dir.mkdir(parents=True)
    shutil.copy(FIXTURES / f"{stem}.mps", instance_dir / f"{stem}.mps")
    data_path = instance_dir / f"{stem}.data"
    data_path.write_text("{corrupt")

    transform_instance("cls", stem, cache_dir=tmp_path)

    assert (instance_dir / f"{stem}.std").is_file()
    assert json.loads(data_path.read_text())["transform_status"] == "ok"


def test_transform_recovers_from_invalid_utf8_existing_data(tmp_path: Path) -> None:
    stem = "surviving_mixed"
    instance_dir = tmp_path / "cls" / stem
    instance_dir.mkdir(parents=True)
    shutil.copy(FIXTURES / f"{stem}.mps", instance_dir / f"{stem}.mps")
    data_path = instance_dir / f"{stem}.data"
    data_path.write_bytes(b"\xff\xfe{")

    transform_instance("cls", stem, cache_dir=tmp_path)

    assert (instance_dir / f"{stem}.std").is_file()
    assert json.loads(data_path.read_text())["transform_status"] == "ok"


def _seed_all_result_keys(data_path: Path) -> None:
    data = json.loads(data_path.read_text())
    data.update(DOWNSTREAM_DATA)
    data.update(SURVIVING_DATA)
    data_path.write_text(json.dumps(data))


def _read_std_arrays(std_path: Path) -> dict[str, np.ndarray]:
    with np.load(std_path, allow_pickle=False) as data:
        return {key: data[key].copy() for key in data.files}


def test_unchanged_retransform_preserves_all_downstream_data(tmp_path: Path) -> None:
    stem = "surviving_mixed"
    instance_dir = tmp_path / "cls" / stem
    instance_dir.mkdir(parents=True)
    mps_path = instance_dir / f"{stem}.mps"
    shutil.copy(FIXTURES / f"{stem}.mps", mps_path)
    transform_instance("cls", stem, cache_dir=tmp_path)
    data_path = instance_dir / f"{stem}.data"
    _seed_all_result_keys(data_path)

    transform_instance("cls", stem, cache_dir=tmp_path)
    data = json.loads(data_path.read_text())
    assert data == {"transform_status": "ok", **DOWNSTREAM_DATA, **SURVIVING_DATA}


def test_changed_retransform_purges_exact_downstream_data(tmp_path: Path) -> None:
    stem = "surviving_mixed"
    instance_dir = tmp_path / "cls" / stem
    instance_dir.mkdir(parents=True)
    mps_path = instance_dir / f"{stem}.mps"
    shutil.copy(FIXTURES / f"{stem}.mps", mps_path)
    transform_instance("cls", stem, cache_dir=tmp_path)
    std_path = instance_dir / f"{stem}.std"
    before = _read_std_arrays(std_path)
    data_path = instance_dir / f"{stem}.data"
    _seed_all_result_keys(data_path)

    mps_path.write_text(mps_path.read_text().replace("X1        OBJ        1.0", "X1        OBJ        1.25"))
    transform_instance("cls", stem, cache_dir=tmp_path)
    after = _read_std_arrays(std_path)
    assert any(not np.array_equal(before[key], after[key]) for key in before)
    data = json.loads(data_path.read_text())
    assert data == {"transform_status": "ok", **SURVIVING_DATA}


def test_changed_transform_purges_before_interrupted_std_write(
    tmp_path: Path, monkeypatch
) -> None:
    stem = "surviving_mixed"
    instance_dir = tmp_path / "cls" / stem
    instance_dir.mkdir(parents=True)
    mps_path = instance_dir / f"{stem}.mps"
    shutil.copy(FIXTURES / f"{stem}.mps", mps_path)
    transform_instance("cls", stem, cache_dir=tmp_path)
    std_path = instance_dir / f"{stem}.std"
    old_std_bytes = std_path.read_bytes()
    data_path = instance_dir / f"{stem}.data"
    _seed_all_result_keys(data_path)
    mps_path.write_text(
        mps_path.read_text().replace("X1        OBJ        1.0", "X1        OBJ        1.25")
    )

    def interrupt_write(path, **arrays):
        raise KeyboardInterrupt

    monkeypatch.setattr(transform, "_atomic_write_std", interrupt_write)
    with pytest.raises(KeyboardInterrupt):
        transform_instance("cls", stem, cache_dir=tmp_path)

    assert std_path.read_bytes() == old_std_bytes
    data = json.loads(data_path.read_text())
    assert data == {"transform_status": "ok", **SURVIVING_DATA}


def test_failed_rerun_retracts_std_and_downstream_data(tmp_path: Path) -> None:
    stem = "surviving_mixed"
    instance_dir = tmp_path / "cls" / stem
    instance_dir.mkdir(parents=True)
    mps_path = instance_dir / f"{stem}.mps"
    shutil.copy(FIXTURES / f"{stem}.mps", mps_path)
    transform_instance("cls", stem, cache_dir=tmp_path)
    data_path = instance_dir / f"{stem}.data"
    _seed_all_result_keys(data_path)
    mps_path.write_text("not an MPS file")

    with pytest.raises(RuntimeError):
        transform_instance("cls", stem, cache_dir=tmp_path)

    assert not (instance_dir / f"{stem}.std").exists()
    data = json.loads(data_path.read_text())
    assert data == {"transform_status": "error:RuntimeError", **SURVIVING_DATA}


def test_clear_std_files_purges_downstream_data(tmp_path: Path) -> None:
    instance_dir = tmp_path / "cls" / "item"
    instance_dir.mkdir(parents=True)
    (instance_dir / "item.std").write_bytes(b"std")
    data_path = instance_dir / "item.data"
    data_path.write_text(
        json.dumps(
            {
                "transform_status": "ok",
                **DOWNSTREAM_DATA,
                **SURVIVING_DATA,
            }
        )
    )
    corrupt_dir = tmp_path / "cls" / "corrupt"
    corrupt_dir.mkdir()
    (corrupt_dir / "corrupt.std").write_bytes(b"std")
    corrupt_data = corrupt_dir / "corrupt.data"
    corrupt_data.write_bytes(b"\xff\xfe{")
    clear_std_files(["cls"], cache_dir=tmp_path)
    assert not (instance_dir / "item.std").exists()
    assert json.loads(data_path.read_text()) == SURVIVING_DATA
    assert not (corrupt_dir / "corrupt.std").exists()
    assert json.loads(corrupt_data.read_text()) == {}


def test_clear_std_files_interleaves_purge_and_unlink_by_instance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in ("b", "a"):
        instance_dir = tmp_path / "cls" / name
        instance_dir.mkdir(parents=True)
        (instance_dir / f"{name}.data").write_text(
            json.dumps({"transform_status": "ok"})
        )
        (instance_dir / f"{name}.std").write_bytes(b"std")

    real_unlink = Path.unlink
    unlinked_instances = []

    def checked_unlink(path: Path, *args, **kwargs) -> None:
        if path.suffix == ".std":
            data = json.loads(path.with_suffix(".data").read_text())
            assert "transform_status" not in data
            if path.parent.name == "a":
                pending_data = json.loads(
                    (tmp_path / "cls" / "b" / "b.data").read_text()
                )
                assert "transform_status" in pending_data
            unlinked_instances.append(path.parent.name)
        real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", checked_unlink)
    clear_std_files(["cls"], cache_dir=tmp_path)
    assert unlinked_instances == ["a", "b"]


@pytest.mark.parametrize(
    ("lower", "upper", "expected_c", "expected_b", "expected_A", "expected_offset"),
    [
        (2.0, 5.0, [3.0, 0.0], [3.0], [[1.0, 1.0]], 7.0),
        (2.0, _HIGHS_INF, [3.0], [], [], 7.0),
        (-_HIGHS_INF, 5.0, [-3.0], [], [], 16.0),
        (-_HIGHS_INF, _HIGHS_INF, [3.0, -3.0], [], [], 1.0),
        (2.0, 2.0, [3.0, 0.0], [0.0], [[1.0, 1.0]], 7.0),
    ],
    ids=["bounded", "lower", "upper", "free", "fixed"],
)
def test_column_conversion_branches(
    lower, upper, expected_c, expected_b, expected_A, expected_offset
) -> None:
    c, b, A, offset = _lp_to_standard_form(
        1,
        0,
        np.array([3.0]),
        np.array([lower]),
        np.array([upper]),
        np.array([]),
        np.array([]),
        np.array([0, 0]),
        np.array([], dtype=int),
        np.array([]),
        1.0,
    )
    np.testing.assert_allclose(c, expected_c)
    np.testing.assert_allclose(b, expected_b)
    expected = np.asarray(expected_A).reshape(len(expected_b), len(expected_c))
    np.testing.assert_allclose(A.toarray(), expected)
    assert offset == pytest.approx(expected_offset)


@pytest.mark.parametrize(
    ("lower", "upper", "expected_A"),
    [(-_HIGHS_INF, 5.0, [[-4.0]]), (-_HIGHS_INF, _HIGHS_INF, [[4.0, -4.0]])],
    ids=["upper_nonzero_coefficient", "free_nonzero_coefficient"],
)
def test_column_signs_with_nonzero_coefficients(lower, upper, expected_A) -> None:
    _, _, A, _ = _lp_to_standard_form(
        1,
        1,
        np.array([3.0]),
        np.array([lower]),
        np.array([upper]),
        np.array([0.0]),
        np.array([0.0]),
        np.array([0, 1]),
        np.array([0]),
        np.array([4.0]),
    )
    np.testing.assert_allclose(A.toarray(), expected_A)


@pytest.mark.parametrize(
    ("lower", "upper", "expected_b", "expected_A"),
    [
        (2.0, 2.0, [2.0], np.empty((1, 0))),
        (-_HIGHS_INF, 2.0, [2.0], [[1.0]]),
        (2.0, _HIGHS_INF, [2.0], [[-1.0]]),
        (1.0, 3.0, [3.0, 2.0], [[1.0, 0.0], [1.0, 1.0]]),
        (-_HIGHS_INF, _HIGHS_INF, [], np.empty((0, 0))),
    ],
    ids=["equality", "less", "greater", "range", "free"],
)
def test_row_conversion_branches(lower, upper, expected_b, expected_A) -> None:
    c, b, A, offset = _lp_to_standard_form(
        0,
        1,
        np.array([]),
        np.array([]),
        np.array([]),
        np.array([lower]),
        np.array([upper]),
        np.array([0]),
        np.array([], dtype=int),
        np.array([]),
    )
    np.testing.assert_allclose(c, np.zeros(A.shape[1]))
    np.testing.assert_allclose(b, expected_b)
    np.testing.assert_allclose(A.toarray(), expected_A)
    assert offset == 0.0
