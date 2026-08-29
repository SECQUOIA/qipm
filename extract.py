#!/usr/bin/env python3
"""Clone simplex-benchmarks with Git LFS, extract data into cache_dir, then remove the temp clone."""

import argparse
from datetime import datetime, timezone
import json
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

from store import atomic_write_bytes, atomic_write_json, merge_ledger

REPO_URL = "https://gitlab.ibr.cs.tu-bs.de/alg/simplex-benchmarks"
CACHE_DIR = Path("cache_dir")

INSTANCE_CLASSES = (
    "independent_set",
    "vertex_cover",
    "clique",
    "max_flow",
    "netlib",
    "miplib",
    "stochlp",
    "misc",
)

# Zip basename (no .zip) -> cache subfolder. max_flow and random_directed_graphs both -> max_flow.
ZIP_TO_CLASS = {
    "stochlp": "stochlp",
    "netlib": "netlib",
    "miplib": "miplib",
    "misc": "misc",
    "max_flow": "max_flow",
    "random_directed_graphs": "max_flow",
}

# Zips whose instances are classified by filename: clq->clique, is->independent_set, vc->vertex_cover
ZIP_BY_FILENAME = ("clq_mis_vc_dimacs", "clq_mis_vc_random")

EVAL_SUBDIRS = ("easy_steepest", "hard_steepest")
EVAL_ROOT = Path("benchmark/01_evaluation")


def _contained_path(root: Path, *parts: str) -> Path:
    """Resolve a destination and reject names that escape the cache root."""
    root = root.resolve()
    destination = root.joinpath(*parts).resolve()
    if not destination.is_relative_to(root):
        raise ValueError(f"Archive-derived path escapes cache root: {destination}")
    return destination


def _cache_class_and_stem_from_file_path(file_path: str) -> tuple[str, str] | None:
    """Derive (cache_class, instance_stem) from repo-relative file_path (e.g. mps/netlib/min/foo.mps)."""
    normalized = file_path.replace("\\", "/").strip("/")
    parts = normalized.split("/")
    if len(parts) < 2 or not parts[-1].lower().endswith(".mps"):
        return None
    zip_name = parts[1]
    stem = Path(parts[-1]).stem
    if zip_name in ZIP_TO_CLASS:
        return (ZIP_TO_CLASS[zip_name], stem)
    if zip_name in ZIP_BY_FILENAME:
        cache_class = _class_from_filename(parts[-1])
        if cache_class is None:
            return None
        return (cache_class, stem)
    return None


def _class_from_filename(filename: str) -> str | None:
    """Return cache class from filename: 'clq'->clique, 'is'->independent_set, 'vc'->vertex_cover."""
    lower = filename.lower()
    if "clq" in lower:
        return "clique"
    if "is" in lower:
        return "independent_set"
    if "vc" in lower:
        return "vertex_cover"
    return None


def _is_mps_in_min_max(entry_path: str) -> bool:
    """True if path contains a min/ or max/ segment and ends with .mps."""
    normalized = entry_path.replace("\\", "/").strip("/")
    if not normalized.lower().endswith(".mps"):
        return False
    parts = normalized.split("/")
    return len(parts) >= 2 and ("min" in parts or "max" in parts)


def _copy_zip_mps_to_cache(zip_path: Path, cache_path: Path, cache_class: str) -> None:
    """Extract .mps files under min/ or max/ from zip into cache_path/cache_class/<stem>/."""
    with zipfile.ZipFile(zip_path, "r") as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            if not _is_mps_in_min_max(info.filename):
                continue
            name = Path(info.filename).name
            stem = Path(info.filename).stem
            dest = _contained_path(cache_path, cache_class, stem, name)
            atomic_write_bytes(dest, zf.read(info.filename))


def _copy_zip_mps_by_filename(zip_path: Path, cache_path: Path) -> None:
    """Extract .mps from zip into clique/independent_set/vertex_cover by filename; one subfolder per instance."""
    with zipfile.ZipFile(zip_path, "r") as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            if not _is_mps_in_min_max(info.filename):
                continue
            filename = Path(info.filename).name
            stem = Path(info.filename).stem
            cache_class = _class_from_filename(filename)
            if cache_class is None:
                continue
            dest = _contained_path(cache_path, cache_class, stem, filename)
            atomic_write_bytes(dest, zf.read(info.filename))


def _write_data_files_from_evaluation(clone_path: Path, cache_path: Path) -> None:
    """Read benchmark/01_evaluation/*/.../*_compressed.zip; each entry is a .data file (JSON).
    Write per-instance .data files (JSON with key runtime_glpk) into the instance subfolder."""
    eval_root = clone_path / EVAL_ROOT
    if not eval_root.is_dir():
        return
    for subdir_name in EVAL_SUBDIRS:
        level1 = eval_root / subdir_name
        if not level1.is_dir():
            continue
        for level2 in level1.iterdir():
            if not level2.is_dir():
                continue
            for zip_entry in level2.glob("*_compressed.zip"):
                with zipfile.ZipFile(zip_entry, "r") as zf:
                    for info in zf.infolist():
                        if info.is_dir():
                            continue
                        if not info.filename.lower().endswith(".data"):
                            continue
                        try:
                            data = json.loads(zf.read(info.filename).decode())
                        except (json.JSONDecodeError, UnicodeDecodeError):
                            continue
                        runtime_primal = data.get("runtime_primal")
                        file_path = data.get("file_path")
                        if runtime_primal is None or not file_path:
                            continue
                        resolved = _cache_class_and_stem_from_file_path(file_path)
                        if resolved is None:
                            continue
                        cache_class, stem = resolved
                        if stem.lower().endswith(".zip"):
                            continue
                        instance_dir = _contained_path(cache_path, cache_class, stem)
                        instance_dir.mkdir(parents=True, exist_ok=True)
                        data_path = _contained_path(instance_dir, f"{stem}.data")
                        merge_ledger(data_path, {"runtime_glpk": runtime_primal})


def extract(clone_path: Path, cache_path: Path) -> None:
    """Copy or transform relevant data from clone_path into cache_path."""
    for name in INSTANCE_CLASSES:
        (cache_path / name).mkdir(parents=True, exist_ok=True)

    mps_dir = clone_path / "mps"
    if not mps_dir.is_dir():
        return

    for zip_basename, cache_class in ZIP_TO_CLASS.items():
        zip_path = mps_dir / f"{zip_basename}.zip"
        if zip_path.exists():
            _copy_zip_mps_to_cache(zip_path, cache_path, cache_class)

    for zip_basename in ZIP_BY_FILENAME:
        zip_path = mps_dir / f"{zip_basename}.zip"
        if zip_path.exists():
            _copy_zip_mps_by_filename(zip_path, cache_path)

    _write_data_files_from_evaluation(clone_path, cache_path)


def require_cmd(name: str) -> None:
    """Exit with a clear message if command is not on PATH."""
    if shutil.which(name) is None:
        print(f"Error: '{name}' is required but not found on PATH.", file=sys.stderr)
        sys.exit(1)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Clone simplex-benchmarks with Git LFS, extract data into cache_dir, then remove temp clone."
    )
    parser.add_argument(
        "clone_path",
        nargs="?",
        default=None,
        type=Path,
        help="If provided, use this path as the cloned repo (skip clone and LFS pull).",
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Do not delete the temporary clone (for debugging)",
    )
    args = parser.parse_args()

    cache_path = CACHE_DIR.resolve()
    cache_path.mkdir(parents=True, exist_ok=True)

    if args.clone_path is not None:
        clone_path = args.clone_path.resolve()
        if not clone_path.is_dir():
            print(f"Error: clone path is not a directory: {clone_path}", file=sys.stderr)
            return 1
        extract(clone_path, cache_path)
        return 0

    require_cmd("git")
    require_cmd("git-lfs")

    temp_dir = tempfile.mkdtemp(prefix="qipm-clone-")
    clone_path = Path(temp_dir)

    try:
        clone_cmd = [
            "git", "clone", "--depth", "1", "--branch", "main",
            REPO_URL,
            str(clone_path),
        ]
        subprocess.run(clone_cmd, check=True, capture_output=False)

        subprocess.run(
            ["git", "lfs", "pull"],
            cwd=clone_path,
            check=True,
            capture_output=False,
        )

        extract(clone_path, cache_path)
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=clone_path,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        atomic_write_json(
            cache_path / "extract_manifest.json",
            {
                "commit": commit,
                "extracted_at": datetime.now(timezone.utc).isoformat(),
            },
        )
    finally:
        if args.keep_temp:
            print(f"Kept temporary clone at: {clone_path}", file=sys.stderr)
        else:
            shutil.rmtree(temp_dir, ignore_errors=False)

    return 0


if __name__ == "__main__":
    sys.exit(main())
