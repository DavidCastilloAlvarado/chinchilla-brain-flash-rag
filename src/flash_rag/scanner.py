"""File discovery and content hashing for the documents/ tree."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from .config import MAX_FILE_BYTES, SUPPORTED_SUFFIXES


@dataclass(frozen=True)
class FileRecord:
    rel_path: str  # posix-style path relative to documents/
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
