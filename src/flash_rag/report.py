"""Reporting over the local index (``db-report``)."""

from __future__ import annotations

from pathlib import Path

from .config import SUPPORTED_SUFFIXES, Config
from .index import NotInitialized
from .pdf import extract_pdf_pages
from .scanner import JUNK_DIR_NAMES
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
    roots: list[tuple[Path, bool]] = [(cfg.docs_dir, False)]
    roots.extend((ws, True) for ws in cfg.workspace_dirs if ws.is_dir())
    for root, is_ws in roots:
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            rel_parts = path.relative_to(root).parts
            if any(part.startswith(".") for part in rel_parts):
                continue
            if is_ws and any(part in JUNK_DIR_NAMES for part in rel_parts):
                continue
            if is_ws:
                try:
                    rel_str = path.relative_to(cfg.root).as_posix()
                except ValueError:
                    rel_str = str(path)
            else:
                rel_str = path.relative_to(root).as_posix()
            if rel_str in indexed_paths:
                continue
            if path.suffix.lower() == ".pdf":
                # Probe the actual reason (cheap: only unindexed files are checked).
                ex = extract_pdf_pages(path)
                reason = ex.skipped[0] if ex.skipped else "no extractable text"
            elif path.suffix.lower() not in SUPPORTED_SUFFIXES:
                reason = f"unsupported suffix {path.suffix.lower() or '(none)'}"
            else:
                reason = "skipped (empty, oversized, or undecodable)"
            unindexed.append({"file": rel_str, "reason": reason})

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
