#!/usr/bin/env python3
"""Visualise a .std NPZ archive in human-readable form on the console."""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from standard_form import load_standard_form

_STANDARD_MEMBERS = {
    "c",
    "b",
    "A_data",
    "A_indices",
    "A_indptr",
    "A_shape",
    "obj_offset",
}


def _format_array(arr: np.ndarray, max_entries: int = 50) -> str:
    arr = np.asarray(arr).ravel()
    if arr.size == 0:
        return "[]"
    if arr.size <= max_entries:
        return np.array2string(arr, precision=6, suppress_small=True)
    head = np.array2string(arr[: max_entries // 2], precision=6, suppress_small=True)
    tail = np.array2string(
        arr[-(max_entries - max_entries // 2) :], precision=6, suppress_small=True
    )
    return f"{head[:-1]} ... {tail[1:]}  # length {arr.size}"


def _print_standard_form(path: Path, c, b, A, obj_offset: float) -> None:
    m, n = A.shape
    print(f"# {path.name} — Standard form (std)")
    print()
    print("Dimensions: m (rows) =", m, ", n (cols) =", n)
    print()
    print("c (objective, length n):")
    print(_format_array(c))
    print()
    print("b (RHS, length m):")
    print(_format_array(b))
    print()
    print("objective offset:")
    print(obj_offset)
    print()
    print("A (m×n, CSR): nnz =", A.nnz)
    print("A dense:")
    np.set_printoptions(precision=6, suppress=True, linewidth=120)
    print(A.toarray())
    np.set_printoptions()


def _print_generic_npz(data: np.lib.npyio.NpzFile, path: Path) -> None:
    """Print raw archive members, retaining readable data around corrupt members."""
    print(f"# {path.name} — raw npz contents")
    print()
    for key in sorted(data.files):
        try:
            arr = np.asarray(data[key])
        except Exception as exc:  # noqa: BLE001 - diagnose individual corrupt members
            print(f"{key}: error reading member: {type(exc).__name__}: {exc}")
            print()
            continue
        print(f"{key}: shape {arr.shape}, dtype {arr.dtype}")
        if arr.size <= 100:
            np.set_printoptions(precision=6, suppress=True, linewidth=100)
            print(arr)
            np.set_printoptions()
        else:
            print(_format_array(arr.ravel(), max_entries=30))
        print()


def _print_raw_archive(path: Path) -> None:
    try:
        with np.load(path, allow_pickle=False) as data:
            _print_generic_npz(data, path)
    except Exception as exc:  # noqa: BLE001 - report archive-level corruption
        print(f"Raw archive error: {type(exc).__name__}: {exc}")


def visualise(path: Path) -> None:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")

    try:
        c, b, A, obj_offset = load_standard_form(path)
    except Exception as exc:  # noqa: BLE001 - fall back for diagnostic use
        print(f"Validation error: {type(exc).__name__}: {exc}")
        print()
        _print_raw_archive(path)
        return

    with np.load(path, allow_pickle=False) as data:
        unexpected = sorted(set(data.files) - _STANDARD_MEMBERS)
    if unexpected:
        print(f"Validation error: unexpected members: {', '.join(unexpected)}")
        print()
        _print_raw_archive(path)
        return

    _print_standard_form(path, c, b, A, obj_offset)


def main() -> None:
    parser = argparse.ArgumentParser(description="Print a .std archive in human-readable form.")
    parser.add_argument("file", type=Path, help="Path to a .std NPZ archive")
    args = parser.parse_args()
    try:
        visualise(args.file)
    except Exception as exc:  # noqa: BLE001 - command-line diagnostic tool
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
