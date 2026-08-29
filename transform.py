#!/usr/bin/env python3
"""Transform MPS instances to standard-form LP (min c'x, Ax=b, x>=0) after presolve."""

from __future__ import annotations

import warnings
from pathlib import Path

import highspy
from standard_form import (
    _atomic_write_std,
    _lp_to_standard_form,
    _std_matches,
    _strip_zero_rows,
    standard_form_arrays,
)
from store import (
    STD_DERIVED_KEYS,
    TRANSFORM_STATUS_KEY,
    list_class_names,
    list_instance_dirs,
    merge_ledger,
    process_instance_dirs,
    purge_keys_from_ledger,
    resolve_cache_root,
)


def _merge_transform_status(path: Path, status: str, *, retract: bool = False) -> None:
    merge_ledger(
        path.with_suffix(".data"),
        {TRANSFORM_STATUS_KEY: status},
        remove_keys=STD_DERIVED_KEYS if retract else (),
    )


def _purge_downstream_data(path: Path) -> None:
    """Retract results derived from an earlier standard form."""
    merge_ledger(path.with_suffix(".data"), remove_keys=STD_DERIVED_KEYS)


def _withhold_standard_form(path: Path, status: str) -> None:
    _merge_transform_status(path, status, retract=True)
    path.with_suffix(".std").unlink(missing_ok=True)


def _transform_instance_impl(path: Path) -> bool | None:
    """Read MPS file at path, presolve with HiGHS, convert to standard form, and save .std next to it."""
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"MPS file not found: {path}")
    if path.stat().st_size == 0:
        raise RuntimeError(f"MPS file is empty: {path}")

    h = highspy.Highs()
    h.setOptionValue("log_to_console", False)
    status = h.readModel(str(path))
    if status == highspy.HighsStatus.kWarning:
        warnings.warn(f"HiGHS readModel returned kWarning for {path}", stacklevel=2)
    elif status != highspy.HighsStatus.kOk:
        raise RuntimeError(f"HiGHS readModel failed: {status}")

    original_lp = h.getLp()
    if original_lp.sense_ == highspy.ObjSense.kMaximize:
        raise RuntimeError(f"Maximization model is not supported: {path}")
    integrality = getattr(original_lp, "integrality_", None)
    if integrality and any(v != highspy.HighsVarType.kContinuous for v in integrality):
        raise RuntimeError(f"Non-continuous variables are not supported: {path}")

    status = h.presolve()
    if status == highspy.HighsStatus.kWarning:
        warnings.warn(f"HiGHS presolve returned kWarning for {path}", stacklevel=2)
    elif status != highspy.HighsStatus.kOk:
        raise RuntimeError(f"HiGHS presolve failed: {status}")

    presolve_status = h.getModelPresolveStatus()
    model_status = h.getModelStatus()
    if presolve_status == highspy.HighsPresolveStatus.kReducedToEmpty:
        _withhold_standard_form(path, "reduced_to_empty")
        return
    if (
        presolve_status == highspy.HighsPresolveStatus.kInfeasible
        or model_status == highspy.HighsModelStatus.kInfeasible
    ):
        _withhold_standard_form(path, "infeasible")
        return
    if (
        presolve_status == highspy.HighsPresolveStatus.kUnboundedOrInfeasible
        or model_status == highspy.HighsModelStatus.kUnboundedOrInfeasible
    ):
        _withhold_standard_form(path, "unbounded_or_infeasible")
        return
    if model_status == highspy.HighsModelStatus.kUnbounded:
        _withhold_standard_form(path, "unbounded")
        return
    if presolve_status not in (
        highspy.HighsPresolveStatus.kReduced,
        highspy.HighsPresolveStatus.kNotReduced,
    ):
        raise RuntimeError(f"Unexpected HiGHS presolve status: {presolve_status}")

    lp = h.getPresolvedLp()
    if lp is None or (lp.num_col_ == 0 and lp.num_row_ == 0):
        _withhold_standard_form(path, "reduced_to_empty")
        return

    num_col = lp.num_col_
    num_row = lp.num_row_
    a = lp.a_matrix_
    c, b, A, obj_offset = _lp_to_standard_form(
        num_col, num_row,
        lp.col_cost_, lp.col_lower_, lp.col_upper_,
        lp.row_lower_, lp.row_upper_,
        a.start_, a.index_, a.value_,
        lp.offset_,
    )

    # Presolve normally removes empty rows. Keep this as defensive handling for
    # harmless 0 = 0 rows that a solver version or input edge case may retain.
    stripped = _strip_zero_rows(A, b)
    if stripped is None:
        _withhold_standard_form(path, "infeasible")
        return
    A, b = stripped

    arrays = standard_form_arrays(c, b, A, obj_offset)
    out_std = path.with_suffix(".std")
    changed = not _std_matches(out_std, arrays)
    if changed:
        _purge_downstream_data(path)
        _atomic_write_std(out_std, **arrays)
    return changed


def _transform_instance_from_path(path: Path) -> None:
    """Transform one discovered MPS path, retracting stale outputs on failure."""
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"MPS file not found: {path}")
    try:
        changed = _transform_instance_impl(path)
    except Exception as exc:
        _withhold_standard_form(path, f"error:{type(exc).__name__}")
        raise
    if changed is not None:
        _merge_transform_status(path, "ok")


def transform_instance(
    instance_class: str,
    instance_name: str,
    cache_dir: str | Path | None = None,
) -> None:
    """Transform the MPS instance in cache_dir/instance_class/instance_name/ to standard form.

    Discovers the single .mps file in that subdirectory and writes .std next to it.
    instance_class: e.g. "netlib", "miplib".
    instance_name: subfolder name (instance stem).
    cache_dir: root containing instance-class subfolders; defaults to "cache_dir".
    """
    root = resolve_cache_root(cache_dir)
    instance_dir = root / instance_class / instance_name
    if not instance_dir.is_dir():
        raise FileNotFoundError(f"Instance directory not found: {instance_dir}")
    mps_files = sorted(instance_dir.glob("*.mps"))
    if len(mps_files) != 1:
        raise FileNotFoundError(
            f"Expected exactly one .mps in {instance_dir}; found {len(mps_files)}"
        )
    _transform_instance_from_path(mps_files[0])


def transform_instance_class(
    instance_class: str,
    cache_dir: str | Path | None = None,
) -> None:
    """Transform all MPS instances in the given instance-class subfolder of cache_dir to standard form.

    instance_class: name of the subfolder (e.g. "netlib", "miplib", "clique").
    cache_dir: directory containing instance-class subfolders; defaults to "cache_dir" in the current directory.
    """
    root = resolve_cache_root(cache_dir)
    folder = root / instance_class
    if not folder.is_dir():
        raise FileNotFoundError(f"Instance class folder not found: {folder}")

    process_instance_dirs(
        instance_class,
        list_instance_dirs(folder),
        lambda subdir: transform_instance(instance_class, subdir.name, cache_dir=root),
        required_glob="*.mps",
        missing_message="no .mps file",
    )


def transform_all_instance_classes(
    instance_classes: list[str] | None = None,
    cache_dir: str | Path | None = None,
) -> None:
    """Transform MPS instances to standard form (main entry point).

    instance_classes: optional list of instance class names (subfolder names under cache_dir).
        If None, all instance classes (all subdirectories of cache_dir) are processed.
    cache_dir: directory containing instance-class subfolders; defaults to "cache_dir" in the current directory.
    """
    root = resolve_cache_root(cache_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Cache directory not found: {root}")

    if instance_classes is None:
        instance_classes = list_class_names(root)

    for name in instance_classes:
        transform_instance_class(name, root)


def show_transform_status(
    instance_classes: list[str] | None = None,
    cache_dir: str | Path | None = None,
) -> None:
    """Print how many instances per class have a .std file.

    For each instance class, prints "<class>: x/total".
    """
    root = resolve_cache_root(cache_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Cache directory not found: {root}")

    if instance_classes is None:
        instance_classes = list_class_names(root)

    for cls in instance_classes:
        folder = root / cls
        if not folder.is_dir():
            print(f"{cls}: directory not found")
            continue

        subdirs = list_instance_dirs(folder)
        total = len(subdirs)
        done = sum(1 for d in subdirs if any(d.glob("*.std")))
        print(f"{cls}: {done}/{total}")


def clear_std_files(
    instance_classes: list[str] | None = None,
    cache_dir: str | Path | None = None,
) -> None:
    """Delete .std files and purge their transform, solve, and benchmark data."""
    root = resolve_cache_root(cache_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Cache directory not found: {root}")

    search_roots = [root / name for name in instance_classes] if instance_classes else [root]
    keys = (TRANSFORM_STATUS_KEY,) + STD_DERIVED_KEYS
    for search_root in search_roots:
        instance_dirs = {
            path.parent
            for pattern in ("*.std", "*.data")
            for path in search_root.rglob(pattern)
        }
        for instance_dir in sorted(instance_dirs, key=str):
            for data_path in instance_dir.glob("*.data"):
                purge_keys_from_ledger(data_path, keys, sanitize_invalid=True)
            for std_path in instance_dir.glob("*.std"):
                std_path.unlink()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Transform MPS instances to standard-form LP.")
    parser.add_argument(
        "instance_classes",
        nargs="*",
        help="Instance class names (subfolders under cache_dir). If none given, process all.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Cache directory (default: cache_dir in current directory).",
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Delete all .std files instead of transforming. Other flags are ignored.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show how many instances per class have a .std file. Other flags are ignored.",
    )
    args = parser.parse_args()
    if args.show:
        show_transform_status(
            instance_classes=args.instance_classes or None,
            cache_dir=args.cache_dir,
        )
    elif args.clear:
        clear_std_files(
            instance_classes=args.instance_classes or None,
            cache_dir=args.cache_dir,
        )
    else:
        transform_all_instance_classes(
            instance_classes=args.instance_classes or None,
            cache_dir=args.cache_dir,
        )
