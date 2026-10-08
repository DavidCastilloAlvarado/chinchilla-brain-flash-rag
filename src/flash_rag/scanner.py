"""File discovery and content hashing for documents/ and workspace dirs."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from .config import MAX_FILE_BYTES, SUPPORTED_SUFFIXES

# Directory names that are never indexed inside workspace dirs (build output,
# dependencies, caches, VCS). Hidden dirs are already skipped everywhere.
JUNK_DIR_NAMES = frozenset(
    {
        "node_modules",
        "__pycache__",
        "dist",
        "build",
        "out",
        "target",
        "vendor",
        "coverage",
        ".next",
        ".nuxt",
        ".cache",
        ".turbo",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        ".idea",
        ".vscode",
        "tmp",
        "temp",
    }
)


@dataclass(frozen=True)
class FileRecord:
    rel_path: str  # display path (relative to documents/ or the project root)
    abs_path: Path
    size: int
    mtime: float
    sha256: str


def scan_files(docs_dir: Path) -> tuple[list[FileRecord], list[str]]:
    """Return ``(indexable_files, skipped_reasons)`` for *docs_dir*.

    Only ``SUPPORTED_SUFFIXES`` are indexed; hidden files/dirs are ignored;
    empty or oversized files are reported in *skipped_reasons*.
    """
    records: list[FileRecord] = []
    skipped: list[str] = []
    if not docs_dir.is_dir():
        return records, skipped

    for path in sorted(docs_dir.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(docs_dir)
        if any(part.startswith(".") for part in rel.parts):
            continue
        if path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        stat = path.stat()
        if stat.st_size == 0:
            skipped.append(f"{rel.as_posix()}: empty file")
            continue
        if stat.st_size > MAX_FILE_BYTES:
            skipped.append(
                f"{rel.as_posix()}: {stat.st_size} bytes exceeds {MAX_FILE_BYTES} limit"
            )
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        records.append(
            FileRecord(
                rel_path=rel.as_posix(),
                abs_path=path,
                size=stat.st_size,
                mtime=stat.st_mtime,
                sha256=digest,
            )
        )
    return records, skipped


def scan_workspace_dir(ws_dir: Path, base: Path) -> tuple[list[FileRecord], list[str]]:
    """Scan a workspace directory (recursively) for indexable files.

    *rel_path* is relative to *base* (the project root) when the file is under
    it, otherwise the absolute path — so search results stay readable.
    Hidden files/dirs and junk dirs (``JUNK_DIR_NAMES``) are ignored.
    """
    records: list[FileRecord] = []
    skipped: list[str] = []
    if not ws_dir.is_dir():
        return records, skipped

    for path in sorted(ws_dir.rglob("*")):
        if not path.is_file():
            continue
        try:
            rel = path.relative_to(base)
            rel_str = rel.as_posix()
        except ValueError:  # workspace dir outside the project root
            rel_str = str(path)
        rel_parts = path.relative_to(ws_dir).parts
        if any(part.startswith(".") for part in rel_parts):
            continue
        if any(part in JUNK_DIR_NAMES for part in rel_parts):
            continue
        if path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        stat = path.stat()
        if stat.st_size == 0:
            skipped.append(f"{rel_str}: empty file")
            continue
        if stat.st_size > MAX_FILE_BYTES:
            skipped.append(
                f"{rel_str}: {stat.st_size} bytes exceeds {MAX_FILE_BYTES} limit"
            )
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        records.append(
            FileRecord(
                rel_path=rel_str,
                abs_path=path,
                size=stat.st_size,
                mtime=stat.st_mtime,
                sha256=digest,
            )
        )
    return records, skipped
