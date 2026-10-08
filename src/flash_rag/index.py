"""Index building (db-init) and incremental refresh (db-refresh)."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

from .chunker import chunk_file, chunk_text
from .config import EMBED_BATCH_SIZE, Config
from .embedder import make_embedder
from .pdf import extract_pdf_pages
from .scanner import FileRecord, scan_files, scan_workspace_dir
from .store import Store


class NotInitialized(RuntimeError):
    """The vector index does not exist yet."""


class AlreadyInitialized(RuntimeError):
    """The vector index already exists (use db-refresh, or db-init --force)."""


class ModelMismatch(RuntimeError):
    """The stored index was built with a different embedding model."""


@dataclass
class IndexStats:
    files_added: int = 0
    files_updated: int = 0
    files_removed: int = 0
    files_unchanged: int = 0
    chunks_added: int = 0
    chunks_removed: int = 0
    skipped: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# manifest (file hash -> index state) for incremental refresh
# ---------------------------------------------------------------------------


def _load_manifest(cfg: Config) -> dict:
    if cfg.manifest_path.is_file():
        try:
            return json.loads(cfg.manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
    return {}


def _save_manifest(cfg: Config, manifest: dict) -> None:
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    cfg.manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )


def _scan_all(cfg: Config) -> tuple[list[FileRecord], list[str]]:
    """Scan documents/ plus every configured workspace dir (deduped by file)."""
    records, skipped = scan_files(cfg.docs_dir)
    seen = {rec.abs_path for rec in records}
    for ws in cfg.workspace_dirs:
        if not ws.is_dir():
            skipped.append(f"workspace dir {ws}: not found (skipped)")
            continue
        ws_records, ws_skipped = scan_workspace_dir(ws, cfg.root)
        skipped.extend(ws_skipped)
        for rec in ws_records:
            if rec.abs_path not in seen:
                seen.add(rec.abs_path)
                records.append(rec)
    return records, skipped


def _read_file(rec: FileRecord) -> str | None:
    try:
        return rec.abs_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None


def _chunk_rows(rec: FileRecord, text: str) -> list[dict]:
    rows = []
    for c in chunk_file(text, rec.rel_path):
        rows.append(
            {
                "id": f"{rec.rel_path}#{c.index}",
                "file_path": rec.rel_path,
                "section": c.section,
                "page": -1,
                "text": c.text,
                "start": c.start,
                "end": c.end,
                "tokens": c.tokens,
                "file_hash": rec.sha256,
                "mtime": rec.mtime,
            }
        )
    return rows


def _pdf_rows(rec: FileRecord, pages: dict[int, str]) -> list[dict]:
    """One or more chunks per PDF page (long pages are split), all tagged with the page number."""
    rows = []
    idx = 0
    for page_num in sorted(pages):
        for c in chunk_text(pages[page_num]):
            rows.append(
                {
                    "id": f"{rec.rel_path}#{idx}",
                    "file_path": rec.rel_path,
                    "section": "",
                    "page": page_num,
                    "text": c.text,
                    "start": c.start,
                    "end": c.end,
                    "tokens": c.tokens,
                    "file_hash": rec.sha256,
                    "mtime": rec.mtime,
                }
            )
            idx += 1
    return rows


def _embed_texts(
    cfg: Config, texts: list[str], console: Console
) -> tuple[list[list[float]], int]:
    """Embed *texts* in batches, loading the model once (with a note).

    Returns ``(vectors, dim)``.
    """
    console.print(
        f"Loading embedding model [bold]{cfg.model}[/] "
        "(downloaded from Hugging Face on first use, then cached)…"
    )
    embedder = make_embedder(cfg)
    dim = embedder.dim
    vectors: list[list[float]] = []
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total} batches"),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("embedding", total=(len(texts) + EMBED_BATCH_SIZE - 1) // EMBED_BATCH_SIZE)
        for i in range(0, len(texts), EMBED_BATCH_SIZE):
            vectors.extend(embedder.embed(texts[i : i + EMBED_BATCH_SIZE]))
            progress.advance(task)
    return vectors, dim


def _prepare(
    cfg: Config, records: list[FileRecord], console: Console
) -> tuple[list[tuple[FileRecord, list[dict], list[str]]], list[str]]:
    """Chunk *records*; returns (per-file rows+embed texts, skipped reasons)."""
    prepared: list[tuple[FileRecord, list[dict], list[str]]] = []
    skipped: list[str] = []
    for rec in records:
        if rec.rel_path.lower().endswith(".pdf"):
            extraction = extract_pdf_pages(rec.abs_path)
            skipped.extend(f"{rec.rel_path}: {s}" for s in extraction.skipped)
            if not extraction.pages:
                continue
            rows = _pdf_rows(rec, extraction.pages)
            if not rows:
                skipped.append(f"{rec.rel_path}: no chunkable content")
                continue
            texts = [
                f"{rec.rel_path} (page {r['page']})\n{r['text']}" for r in rows
            ]
        else:
            text = _read_file(rec)
            if text is None:
                skipped.append(f"{rec.rel_path}: could not decode as UTF-8")
                continue
            rows = _chunk_rows(rec, text)
            if not rows:
                skipped.append(f"{rec.rel_path}: no chunkable content")
                continue
            texts = [
                f"{rec.rel_path} > {r['section']}\n{r['text']}" if r["section"] else r["text"]
                for r in rows
            ]
        prepared.append((rec, rows, texts))
    return prepared, skipped


def _attach_vectors(
    prepared: list[tuple[FileRecord, list[dict], list[str]]],
    vectors: list[list[float]],
) -> list[dict]:
    out: list[dict] = []
    vi = 0
    for _rec, rows, _texts in prepared:
        for r in rows:
            r = dict(r)
            r["vector"] = vectors[vi]
            vi += 1
            out.append(r)
    return out


def _manifest_entry(rec: FileRecord, n_chunks: int) -> dict:
    return {
        "sha256": rec.sha256,
        "mtime": rec.mtime,
        "size": rec.size,
        "chunks": n_chunks,
    }


# ---------------------------------------------------------------------------
# public commands
# ---------------------------------------------------------------------------


def build(cfg: Config, force: bool = False, console: Console | None = None) -> IndexStats:
    """Map, chunk and embed every supported file (first-time setup)."""
    console = console or Console()
    store = Store(cfg.data_dir, cfg.model)
    if store.exists() and not force:
        raise AlreadyInitialized()
    if store.exists():
        store.drop()

    stats = IndexStats()
    records, skipped = _scan_all(cfg)
    stats.skipped = list(skipped)

    console.print(f"Chunking [bold]{len(records)}[/] file(s) in {cfg.docs_dir}…")
    prepared, skipped2 = _prepare(cfg, records, console)
    stats.skipped.extend(skipped2)
    all_texts = [t for _rec, _rows, texts in prepared for t in texts]
    vectors, dim = _embed_texts(cfg, all_texts, console)
    rows = _attach_vectors(prepared, vectors)

    store.create(rows, dim)
    manifest = {rec.rel_path: _manifest_entry(rec, len(rows_)) for rec, rows_, _t in prepared}
    _save_manifest(cfg, manifest)
    store.write_meta(dim, created=True)

    stats.files_added = len(prepared)
    stats.chunks_added = len(rows)
    return stats


def refresh(cfg: Config, console: Console | None = None) -> IndexStats:
    """Incrementally index new/changed files and drop deleted ones."""
    console = console or Console()
    store = Store(cfg.data_dir, cfg.model)
    if not store.exists():
        raise NotInitialized()
    meta = store.read_meta()
    if meta and meta.get("model") != cfg.model:
        raise ModelMismatch(
            f"Index was built with model '{meta.get('model')}' but the configured "
            f"model is '{cfg.model}'. Rebuild with: uv run db-init --force"
        )

    old_manifest = _load_manifest(cfg)
    records, skipped = _scan_all(cfg)
    new_by_path = {rec.rel_path: rec for rec in records}

    stats = IndexStats()
    stats.skipped = list(skipped)

    to_process: list[FileRecord] = []
    for rec in records:
        old = old_manifest.get(rec.rel_path)
        if old is None:
            to_process.append(rec)
            stats.files_added += 1
        elif old.get("sha256") != rec.sha256:
            to_process.append(rec)
            stats.files_updated += 1
        else:
            stats.files_unchanged += 1

    for path, old in old_manifest.items():
        if path not in new_by_path:
            store.delete_file(path)
            stats.files_removed += 1
            stats.chunks_removed += int(old.get("chunks", 0))

    if to_process:
        console.print(f"Chunking [bold]{len(to_process)}[/] changed file(s)…")
        prepared, skipped2 = _prepare(cfg, to_process, console)
        stats.skipped.extend(skipped2)
        all_texts = [t for _rec, _rows, texts in prepared for t in texts]
        vectors, dim = _embed_texts(cfg, all_texts, console)
        rows = _attach_vectors(prepared, vectors)

        by_file: dict[str, list[dict]] = {}
        for row in rows:
            by_file.setdefault(row["file_path"], []).append(row)
        for rec in to_process:
            store.delete_file(rec.rel_path)  # drop stale chunks first
            file_rows = by_file.get(rec.rel_path, [])
            if file_rows:
                store.add(file_rows, dim)
                old_manifest[rec.rel_path] = _manifest_entry(rec, len(file_rows))
                stats.chunks_added += len(file_rows)
            else:
                old_manifest.pop(rec.rel_path, None)

    for path in list(old_manifest):
        if path not in new_by_path:
            old_manifest.pop(path, None)
    _save_manifest(cfg, old_manifest)
    dim = _dim_of(rows) if to_process else (meta or {}).get("dim", 0)
    store.write_meta(dim, created=False)
    return stats


def _dim_of(rows: list[dict] | None) -> int:
    if rows:
        v = rows[0].get("vector")
        if v:
            return len(v)
    return 0
