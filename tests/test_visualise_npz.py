"""Tests for the diagnostic standard-form archive visualizer."""

from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix

from standard_form import standard_form_arrays
from tools.visualise_npz import visualise


def test_visualiser_dumps_raw_members_when_archive_has_unexpected_member(
    tmp_path: Path, capsys
) -> None:
    path = tmp_path / "extra.std"
    arrays = standard_form_arrays(
        np.ones(2), np.ones(2), csr_matrix(np.eye(2)), 0.0
    )
    with path.open("wb") as stream:
        np.savez_compressed(stream, **arrays, unexpected=np.array([3.0]))

    visualise(path)

    output = capsys.readouterr().out
    assert "Validation error: unexpected members: unexpected" in output
    assert "unexpected: shape (1,), dtype float64" in output
    assert "[3.]" in output


def test_visualiser_dumps_raw_members_after_validation_failure(
    tmp_path: Path, capsys
) -> None:
    path = tmp_path / "missing_c.std"
    with path.open("wb") as stream:
        np.savez_compressed(stream, b=np.array([2.0]))

    visualise(path)

    output = capsys.readouterr().out
    assert "Validation error: KeyError" in output
    assert "b: shape (1,), dtype float64" in output
    assert "[2.]" in output
