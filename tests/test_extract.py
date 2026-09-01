"""Regression tests for extraction data safety."""

import json
import zipfile
from pathlib import Path

import pytest
import highspy

from extract import (
    _contained_path,
    _copy_zip_mps_to_cache,
    _write_data_files_from_evaluation,
)

MPS = b"""NAME TEST
ROWS
 N OBJ
 L R1
COLUMNS
 X OBJ 1 R1 1
RHS
 RHS1 R1 1
BOUNDS
 LO BND X 0
ENDATA
"""


def test_evaluation_merge_preserves_existing_data(tmp_path: Path) -> None:
    clone = tmp_path / "clone"
    archive_dir = clone / "benchmark" / "01_evaluation" / "easy_steepest" / "run"
    archive_dir.mkdir(parents=True)
    archive = archive_dir / "results_compressed.zip"
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr(
            "foo.data",
            json.dumps(
                {"runtime_primal": 1.25, "file_path": "mps/netlib/min/foo.mps"}
            ),
        )
        output.writestr(
            "corrupt.data",
            json.dumps(
                {"runtime_primal": 2.5, "file_path": "mps/netlib/min/corrupt.mps"}
            ),
        )
        output.writestr(
            "bar.data",
            json.dumps(
                {"runtime_primal": 3.5, "file_path": "mps/netlib/min/bar.mps"}
            ),
        )
        output.writestr(
            "huge.data",
            '{"runtime_primal":' + "1" * 5000
            + ',"file_path":"mps/netlib/min/huge.mps"}',
        )

    cache = tmp_path / "cache"
    instance_dir = cache / "netlib" / "foo"
    instance_dir.mkdir(parents=True)
    data_path = instance_dir / "foo.data"
    data_path.write_text(json.dumps({"cycle_count_mnes": 123}))
    corrupt_dir = cache / "netlib" / "corrupt"
    corrupt_dir.mkdir(parents=True)
    corrupt_path = corrupt_dir / "corrupt.data"
    corrupt_path.write_bytes(b"\xff\xfe{")

    _write_data_files_from_evaluation(clone, cache)

    assert json.loads(data_path.read_text()) == {
        "cycle_count_mnes": 123,
        "runtime_glpk": 1.25,
    }
    assert json.loads(corrupt_path.read_text()) == {"runtime_glpk": 2.5}
    assert json.loads((cache / "netlib" / "bar" / "bar.data").read_text()) == {
        "runtime_glpk": 3.5
    }
    assert not (cache / "netlib" / "huge").exists()


def test_containment_guard_rejects_escape(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="escapes cache root"):
        _contained_path(tmp_path / "cache", "..", "outside")


def test_zip_extraction_preserves_objective_sense_and_skips_empty(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "models.zip"
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr("models/min/minimum.mps", MPS)
        output.writestr("models/max/maximum.mps", MPS)
        output.writestr("models/min/empty.mps", b"")

    stale_empty = tmp_path / "cache" / "netlib" / "empty" / "empty.mps"
    stale_empty.parent.mkdir(parents=True)
    stale_empty.write_bytes(MPS)

    with pytest.warns(UserWarning, match="models/min/empty.mps"):
        _copy_zip_mps_to_cache(archive, tmp_path / "cache", "netlib")

    minimum = tmp_path / "cache" / "netlib" / "minimum" / "minimum.mps"
    maximum = tmp_path / "cache" / "netlib" / "maximum" / "maximum.mps"
    assert minimum.read_bytes() == MPS
    assert not stale_empty.exists()
    highs = highspy.Highs()
    highs.setOptionValue("log_to_console", False)
    assert highs.readModel(str(maximum)) == highspy.HighsStatus.kOk
    assert highs.getLp().sense_ == highspy.ObjSense.kMaximize


def test_zip_extraction_adds_max_sense_before_rows_without_name(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "models.zip"
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr("models/max/nameless.mps", MPS.split(b"\n", 1)[1])

    _copy_zip_mps_to_cache(archive, tmp_path / "cache", "netlib")

    path = tmp_path / "cache" / "netlib" / "nameless" / "nameless.mps"
    highs = highspy.Highs()
    highs.setOptionValue("log_to_console", False)
    assert highs.readModel(str(path)) == highspy.HighsStatus.kOk
    assert highs.getLp().sense_ == highspy.ObjSense.kMaximize


def test_zip_extraction_rejects_explicit_objsense(tmp_path: Path) -> None:
    explicit_min = MPS.replace(
        b"NAME TEST\n", b"NAME TEST\nOBJSENSE\n MIN\n"
    )
    archive = tmp_path / "models.zip"
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr("models/max/explicit.mps", explicit_min)

    with pytest.raises(
        RuntimeError,
        match=r"models/max/explicit\.mps.*objective-sense declaration",
    ):
        _copy_zip_mps_to_cache(archive, tmp_path / "cache", "netlib")


def test_zip_extraction_rejects_explicit_objsens(
    tmp_path: Path,
) -> None:
    ignored_max = MPS.replace(
        b"ENDATA\n", b"ENDATA\nOBJSENS\n MAX\n"
    )
    archive = tmp_path / "models.zip"
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr("models/max/ignored.mps", ignored_max)

    with pytest.raises(
        RuntimeError,
        match=r"models/max/ignored\.mps.*objective-sense declaration",
    ):
        _copy_zip_mps_to_cache(archive, tmp_path / "cache", "netlib")


def test_empty_entry_does_not_remove_file_written_in_same_run(tmp_path: Path) -> None:
    archive = tmp_path / "models.zip"
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr("first/min/same.mps", MPS)
        output.writestr("second/min/same.mps", b"")

    with pytest.warns(UserWarning, match="second/min/same.mps"):
        _copy_zip_mps_to_cache(archive, tmp_path / "cache", "netlib")

    path = tmp_path / "cache" / "netlib" / "same" / "same.mps"
    assert path.read_bytes() == MPS


def test_duplicate_zip_member_reads_each_entry_by_info(tmp_path: Path) -> None:
    archive = tmp_path / "models.zip"
    entry = "models/min/duplicate.mps"
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr(entry, MPS)
        with pytest.warns(UserWarning, match="Duplicate name"):
            output.writestr(entry, b"")

    with pytest.warns(UserWarning, match=entry):
        _copy_zip_mps_to_cache(archive, tmp_path / "cache", "netlib")

    path = tmp_path / "cache" / "netlib" / "duplicate" / "duplicate.mps"
    assert path.read_bytes() == MPS


def test_zip_extraction_rejects_min_max_stem_collision(tmp_path: Path) -> None:
    archive = tmp_path / "models.zip"
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr("models/min/same.mps", MPS)
        output.writestr("models/max/same.mps", MPS)

    with pytest.raises(RuntimeError, match=r"netlib/same.*both min/ and max/"):
        _copy_zip_mps_to_cache(archive, tmp_path / "cache", "netlib")
