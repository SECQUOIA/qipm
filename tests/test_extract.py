"""Regression tests for extraction data safety."""

import json
import zipfile
from pathlib import Path

import pytest

from extract import _contained_path, _write_data_files_from_evaluation


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
