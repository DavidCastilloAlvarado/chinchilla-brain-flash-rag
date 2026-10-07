"""Reporting over the local index (``db-report``)."""

from __future__ import annotations

from pathlib import Path

from .config import SUPPORTED_SUFFIXES, Config
from .index import NotInitialized, _load_manifest
from .scanner import scan_files
from .store import Store


def _dir_key(rel_path: str) -> str:
    parts = Path(rel_path).parts
    return parts[0] + "/" if len(parts) > 1 else "(root)"


def _index_bytes(data_dir: Path) -> int:
    total = 0
    lancedb_dir = data_dir / "lancedb"
    if lancedb_dir.is_dir():
        for p in lancedb_dir.rglob("*"):
            if p.is_file():
                total += p.stat().st_size
    return total


def build_report(cfg: Config) -> dict:
    """Aggregate file/chunk/token stats from the index and the documents tree."""
    store = Store(cfg.data_dir, cfg.model)
    if not store.exists():
        raise NotInitialized()
    meta = store.read_meta() or {}

    by_file: dict[str, dict] = {}
    total_chunks = 0
    total_tokens = 0
    if store.count() > 0:
        lance_table = store.table().to_lance().to_table(columns=["file_path", "tokens"])
        grouped = lance_table.group_by("file_path").aggregate(
            [("tokens", "sum"), ("tokens", "count")]
        )
        for row in grouped.to_pylist():
            by_file[row["file_path"]] = {
                "chunks": int(row["tokens_count"]),
                "tokens": int(row["tokens_sum"]),
            }
            total_chunks += int(row["tokens_count"])
            total_tokens += int(row["tokens_sum"])

    by_dir: dict[str, dict] = {}
    for path, st in by_file.items():
        key = _dir_key(path)
        d = by_dir.setdefault(key, {"files": 0, "chunks": 0, "tokens": 0})
        d["files"] += 1
        d["chunks"] += st["chunks"]
        d["tokens"] += st["tokens"]

    # Files on disk that are NOT in the index (unsupported types, empty, oversized).
    indexed_paths = set(by_file)
    unindexed: list[dict] = []
    if cfg.docs_dir.is_dir():
        for path in sorted(cfg.docs_dir.rglob("*")):
            if not path.is_file():
                continue
            rel = path.relative_to(cfg.docs_dir)
            if any(part.startswith(".") for part in rel.parts):
                continue
            if rel.as_posix() in indexed_paths:
                continue
            if path.suffix.lower() not in SUPPORTED_SUFFIXES:
                reason = f"unsupported suffix {path.suffix.lower() or '(none)'}"
            else:
                reason = "skipped (empty or oversized)"
            unindexed.append({"file": rel.as_posix(), "reason": reason})

    return {
        "model": meta.get("model"),
        "dim": meta.get("dim"),
        "created": meta.get("created"),
        "updated": meta.get("updated"),
        "files_indexed": len(by_file),
        "files_unindexed": len(unindexed),
        "chunks": total_chunks,
        "total_tokens": total_tokens,
        "index_bytes": _index_bytes(cfg.data_dir),
        "by_dir": by_dir,
        "by_file": [
            {"file": p, **st}
            for p, st in sorted(by_file.items(), key=lambda kv: -kv[1]["tokens"])
        ],
        "unindexed": unindexed,
    }
