"""Regression tests for HiGHS solve statuses and timing."""

import json
import shutil
import zipfile
from pathlib import Path

import numpy as np
import pytest
from scipy.sparse import csr_matrix

import standard_form
import solve
from solve import (
    HIGHS_THREADS,
    HIGHS_VERSION,
    _merge_solve_result,
    _solve_instance_from_path,
    show_solve_status,
    solve_instance,
)
from standard_form import write_standard_form
from transform import transform_instance

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def _write_std(path: Path, A: np.ndarray, b: np.ndarray, c: np.ndarray) -> None:
    sparse = csr_matrix(A, dtype=np.float64)
    write_standard_form(path, c, b, sparse)


@pytest.mark.parametrize("stem", ["surviving_mixed", "surviving_range"])
def test_solve_mps_and_std_write_ok_statuses(stem: str, tmp_path: Path) -> None:
    instance_dir = tmp_path / "cls" / stem
    instance_dir.mkdir(parents=True)
    shutil.copy(FIXTURES / f"{stem}.mps", instance_dir / f"{stem}.mps")
    transform_instance("cls", stem, cache_dir=tmp_path)

    solve_instance("cls", stem, cache_dir=tmp_path, format="both")

    data = json.loads((instance_dir / f"{stem}.data").read_text())
    assert data["solve_status_mps"] in ("ok", "ok_ipm")
    assert data["solve_status_std"] in ("ok", "ok_ipm")
    assert data["runtime_highs_mps"] >= 0.0
    assert data["runtime_highs_std"] >= 0.0
    assert data["highs_version"] == HIGHS_VERSION
    assert data["highs_threads"] == 1


@pytest.mark.parametrize("suffix", ["mps", "std"])
def test_solve_resets_highs_scheduler_after_in_process_transform(
    suffix: str, tmp_path: Path
) -> None:
    stem = "surviving_mixed"
    instance_dir = tmp_path / "cls" / stem
    instance_dir.mkdir(parents=True)
    shutil.copy(FIXTURES / f"{stem}.mps", instance_dir / f"{stem}.mps")
    transform_instance("cls", stem, cache_dir=tmp_path)

    _solve_instance_from_path(instance_dir / f"{stem}.{suffix}")

    data = json.loads((instance_dir / f"{stem}.data").read_text())
    assert data[f"solve_status_{suffix}"] in ("ok", "ok_ipm")


def test_infeasible_std_records_non_optimal_without_runtime(tmp_path: Path) -> None:
    std_path = tmp_path / "infeasible.std"
    _write_std(
        std_path,
        np.array([[1.0, 0.0], [1.0, 0.0]]),
        np.array([1.0, 2.0]),
        np.array([1.0, 1.0]),
    )
    _solve_instance_from_path(std_path)
    data = json.loads((tmp_path / "infeasible.data").read_text())
    assert data["solve_status_std"] == "non_optimal"
    assert "runtime_highs_std" not in data


def test_infeasible_mps_records_non_optimal_without_runtime(tmp_path: Path) -> None:
    mps_path = tmp_path / "infeasible.mps"
    mps_path.write_text(
        """NAME INFEASIBLE
ROWS
 N OBJ
 E R1
 E R2
COLUMNS
    X OBJ 1 R1 1
    X R2 1
RHS
    RHS1 R1 1 R2 2
BOUNDS
 LO BND X 0
ENDATA
"""
    )
    _solve_instance_from_path(mps_path)
    data = json.loads(mps_path.with_suffix(".data").read_text())
    assert data["solve_status_mps"] == "non_optimal"
    assert "runtime_highs_mps" not in data


def test_invalid_std_records_error_status(tmp_path: Path) -> None:
    std_path = tmp_path / "invalid.std"
    _write_std(std_path, np.eye(2), np.ones(2), np.ones(3))
    _solve_instance_from_path(std_path)
    data = json.loads((tmp_path / "invalid.data").read_text())
    assert data["solve_status_std"] == "error:ValueError"
    assert "runtime_highs_std" not in data


@pytest.mark.parametrize("field", ["A_indices", "A_indptr", "A_shape"])
def test_fractional_structural_metadata_is_rejected(field: str, tmp_path: Path) -> None:
    std_path = tmp_path / f"fractional_{field}.std"
    arrays = {
        "c": np.ones(2),
        "b": np.ones(2),
        "A_data": np.ones(2),
        "A_indices": np.array([0, 1]),
        "A_indptr": np.array([0, 1, 2]),
        "A_shape": np.array([2, 2]),
    }
    arrays[field] = arrays[field].astype(float)
    arrays[field].flat[0] += 0.5
    with std_path.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    _solve_instance_from_path(std_path)
    data = json.loads(std_path.with_suffix(".data").read_text())
    assert data["solve_status_std"] == "error:ValueError"
    assert "runtime_highs_std" not in data


def test_solve_result_recovers_from_corrupt_existing_json(tmp_path: Path) -> None:
    std_path = tmp_path / "valid.std"
    _write_std(std_path, np.eye(2), np.ones(2), np.ones(2))
    std_path.with_suffix(".data").write_text("not json")
    _solve_instance_from_path(std_path)
    data = json.loads(std_path.with_suffix(".data").read_text())
    assert data["solve_status_std"] in ("ok", "ok_ipm")


def test_solve_ignores_corrupt_optional_obj_offset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    std_path = tmp_path / "corrupt_offset.std"
    _write_std(std_path, np.eye(2), np.ones(2), np.ones(2))
    real_load = standard_form.np.load

    class CorruptOffsetArchive:
        def __init__(self, archive):
            self.archive = archive

        def __enter__(self):
            self.archive.__enter__()
            return self

        def __exit__(self, *args):
            return self.archive.__exit__(*args)

        def __contains__(self, key):
            return key in self.archive

        def __getitem__(self, key):
            if key == "obj_offset":
                raise zipfile.BadZipFile("corrupt optional member")
            return self.archive[key]

    monkeypatch.setattr(
        standard_form.np,
        "load",
        lambda *args, **kwargs: CorruptOffsetArchive(real_load(*args, **kwargs)),
    )
    _solve_instance_from_path(std_path)
    data = json.loads(std_path.with_suffix(".data").read_text())
    assert data["solve_status_std"] in ("ok", "ok_ipm")


def test_show_solve_status_accepts_ok_ipm_and_unhashable_status(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    for name, data in {
        "ipm": {"solve_status_std": "ok_ipm"},
        "unhashable": {"solve_status_std": []},
    }.items():
        instance_dir = tmp_path / "cls" / name
        instance_dir.mkdir(parents=True)
        (instance_dir / f"{name}.data").write_text(json.dumps(data))

    show_solve_status(["cls"], format="std", cache_dir=tmp_path)
    assert "std: 1/2 (malformed: 1)" in capsys.readouterr().out


def test_both_mode_mps_timeout_retracts_std_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    instance_dir = tmp_path / "cls" / "item"
    instance_dir.mkdir(parents=True)
    mps_path = instance_dir / "item.mps"
    std_path = instance_dir / "item.std"
    mps_path.write_text("unused")
    std_path.write_text("unused")
    (instance_dir / "item.data").write_text(json.dumps({
        "runtime_highs_std": 12.5,
        "solve_status_std": "ok",
    }))
    calls = []

    def fake_solve(path: Path) -> str:
        calls.append(path)
        return "timeout"

    monkeypatch.setattr(solve, "_solve_with_timeout", fake_solve)
    solve_instance("cls", "item", cache_dir=tmp_path, format="both")

    data = json.loads((instance_dir / "item.data").read_text())
    assert calls == [mps_path]
    assert data["solve_status_std"] == "skipped_mps_timeout"
    assert "runtime_highs_std" not in data


def test_fresh_solve_ledger_writes_runtime_before_status(tmp_path: Path) -> None:
    std_path = tmp_path / "ordered.std"
    _merge_solve_result(std_path, "ok", 1.25)
    data = json.loads(std_path.with_suffix(".data").read_text())
    assert data == {
        "runtime_highs_std": 1.25,
        "solve_status_std": "ok",
        "highs_version": HIGHS_VERSION,
        "highs_threads": HIGHS_THREADS,
    }
    assert list(data)[:2] == ["runtime_highs_std", "solve_status_std"]


def test_solve_instance_file_not_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Instance directory not found"):
        solve_instance("x", "nonexistent", cache_dir=tmp_path)


def test_solve_instance_unsupported_format(tmp_path: Path) -> None:
    bad_path = tmp_path / "dummy.xyz"
    bad_path.write_text("not an instance")
    with pytest.raises(ValueError, match="Unsupported instance format"):
        _solve_instance_from_path(bad_path)
