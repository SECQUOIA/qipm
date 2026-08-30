"""Shared cache traversal, ledger, and replacement-write helpers."""

from __future__ import annotations

import json
import math
import os
import tempfile
from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any, Literal, NamedTuple

import numpy as np
from tqdm import tqdm

VARIANTS = ("mnes", "oss")
BENCHMARK_MODEL = 2
BENCHMARK_MODEL_KEY = "benchmark_model"


class BenchmarkValueKeys(NamedTuple):
    count: str
    count_best_known: str
    count_floor: str
    sparsity: str
    sparsity_method: str
    cond: str
    cond_method: str
    qlsa_queries: str
    qlsa_queries_best_known: str
    qlsa_queries_floor: str
    tomography_reps: str


TRANSFORM_STATUS_KEY = "transform_status"
SOLVE_RESULT_KEYS = {
    "mps": ("runtime_highs_mps", "solve_status_mps"),
    "std": ("runtime_highs_std", "solve_status_std"),
}
BENCHMARK_VALUE_KEYS = {
    variant: BenchmarkValueKeys(
        f"cycle_count_{variant}",
        f"cycle_count_best_known_{variant}",
        f"cycle_count_floor_{variant}",
        f"sparsity_{variant}",
        f"sparsity_method_{variant}",
        f"cond_{variant}",
        f"cond_method_{variant}",
        f"qlsa_queries_{variant}",
        f"qlsa_queries_best_known_{variant}",
        f"qlsa_queries_floor_{variant}",
        f"tomography_reps_{variant}",
    )
    for variant in VARIANTS
}
BENCHMARK_STATUS_KEYS = {variant: f"status_{variant}" for variant in VARIANTS}
BENCHMARK_RESULT_KEYS = {
    variant: BENCHMARK_VALUE_KEYS[variant] + (BENCHMARK_STATUS_KEYS[variant],)
    for variant in VARIANTS
}

# Results whose meaning depends on the current .std. MPS solve results are
# intentionally excluded because they remain valid when standard form changes.
STD_DERIVED_KEYS = (
    SOLVE_RESULT_KEYS["std"]
    + tuple(key for variant in VARIANTS for key in BENCHMARK_RESULT_KEYS[variant])
    + (BENCHMARK_MODEL_KEY,)
)

LedgerState = Literal["missing", "invalid", "valid"]


def _atomic_write(path: Path, mode: str, writer: Callable[[Any], None]) -> None:
    """Replace path through one mode-0644 temporary sibling."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode=mode, dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as tmp:
            tmp_name = tmp.name
            writer(tmp)
        os.chmod(tmp_name, 0o644)
        os.replace(tmp_name, path)
    finally:
        if tmp_name is not None and os.path.exists(tmp_name):
            os.unlink(tmp_name)


def atomic_write_bytes(path: Path, content: bytes) -> None:
    _atomic_write(path, "wb", lambda stream: stream.write(content))


def atomic_write_json(path: Path, data: Mapping[str, Any]) -> None:
    _atomic_write(path, "w", lambda stream: json.dump(data, stream, indent=None))


def atomic_write_npz(path: Path, arrays: Mapping[str, Any]) -> None:
    _atomic_write(path, "w+b", lambda stream: np.savez_compressed(stream, **arrays))


def read_ledger(path: Path) -> tuple[dict[str, Any], LedgerState]:
    """Read a ledger, distinguishing missing, invalid, and valid objects."""
    if not path.exists():
        return {}, "missing"
    try:
        data = json.loads(path.read_text())
    except (ValueError, UnicodeDecodeError):
        return {}, "invalid"
    if not isinstance(data, dict):
        return {}, "invalid"
    return data, "valid"


def merge_ledger(
    path: Path,
    values: Mapping[str, Any] | None = None,
    *,
    remove_keys: Iterable[str] = (),
) -> None:
    """Read, merge, and replace a ledger under the pipeline's single-writer assumption."""
    data, _ = read_ledger(path)
    for key in remove_keys:
        data.pop(key, None)
    if values is not None:
        data.update(values)
    atomic_write_json(path, data)


def purge_keys_from_ledger(
    data_path: Path,
    keys: Iterable[str],
    *,
    sanitize_invalid: bool = False,
) -> None:
    """Remove keys from one ledger, optionally replacing invalid content."""
    keys = tuple(keys)
    data, state = read_ledger(data_path)
    if (sanitize_invalid and state == "invalid") or any(key in data for key in keys):
        for key in keys:
            data.pop(key, None)
        atomic_write_json(data_path, data)


def purge_keys_from_ledgers(
    search_roots: Iterable[Path],
    keys: Iterable[str],
    *,
    sanitize_invalid: bool = False,
) -> None:
    """Remove keys from discovered ledgers, optionally replacing invalid ledgers."""
    keys = tuple(keys)
    for search_root in search_roots:
        for data_path in search_root.rglob("*.data"):
            purge_keys_from_ledger(
                data_path, keys, sanitize_invalid=sanitize_invalid
            )


def resolve_cache_root(cache_dir: str | Path | None) -> Path:
    return Path(cache_dir).resolve() if cache_dir is not None else Path("cache_dir").resolve()


def list_class_names(root: Path) -> list[str]:
    return [entry.name for entry in sorted(root.iterdir()) if entry.is_dir()]


def list_instance_dirs(folder: Path) -> list[Path]:
    return sorted(entry for entry in folder.iterdir() if entry.is_dir())


def process_instance_dirs(
    instance_class: str,
    instance_dirs: Iterable[Path],
    process: Callable[[Path], None],
    *,
    required_glob: str | None = None,
    missing_message: str | None = None,
) -> None:
    """Run one class with progress and per-instance exception isolation."""
    for instance_dir in tqdm(instance_dirs, desc=instance_class, unit="instance"):
        if required_glob is not None and not any(instance_dir.glob(required_glob)):
            tqdm.write(f"skipping {instance_dir.name}: {missing_message}")
            continue
        try:
            process(instance_dir)
        except Exception as exc:  # noqa: BLE001 - isolate corpus instances
            tqdm.write(f"skipping {instance_dir.name}: {exc}")


def stored_finite_number(value: object) -> bool:
    """Return whether a stored int or float is finite, excluding booleans."""
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return True
    return isinstance(value, float) and math.isfinite(value)


def summarize_records(
    instance_dirs: Iterable[Path],
    *,
    value_key: str,
    status_key: str,
    ok_statuses: Iterable[str],
    required_model: int | None = None,
) -> tuple[int, Counter[str]]:
    """Summarize successful statuses and label all other records."""
    ok_statuses = tuple(ok_statuses)
    count = 0
    non_ok: Counter[str] = Counter()
    for instance_dir in instance_dirs:
        data_files = sorted(instance_dir.glob("*.data"))
        data, state = read_ledger(data_files[0]) if data_files else ({}, "missing")
        if state == "invalid":
            non_ok["invalid"] += 1
            continue
        value = data.get(value_key)
        legacy_done = (
            status_key not in data
            and stored_finite_number(value)
        )
        status = data.get(status_key)
        if isinstance(status, str) and status in ok_statuses:
            if required_model is not None and data.get(BENCHMARK_MODEL_KEY) != required_model:
                non_ok["outdated_model"] += 1
            else:
                count += 1
        elif legacy_done:
            if required_model is not None:
                non_ok["outdated_model"] += 1
            else:
                count += 1
        elif status_key not in data:
            non_ok["absent"] += 1
        elif isinstance(status, str):
            non_ok[status] += 1
        else:
            non_ok["malformed"] += 1
    return count, non_ok
