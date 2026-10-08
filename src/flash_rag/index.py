"""Index building (db-init) and incremental refresh (db-refresh)."""

from __future__ import annotations

import json
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
    """Atomically write the manifest (tmp + rename) — it is the resume checkpoint."""
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    tmp = cfg.manifest_path.with_name(cfg.manifest_path.name + ".tmp")
    tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(cfg.manifest_path)


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


def _group_prepared(
    prepared: list[tuple[FileRecord, list[dict], list[str]]],
) -> list[list[tuple[FileRecord, list[dict], list[str]]]]:
    """Group files so each group's total chunk count is <= EMBED_BATCH_SIZE.

    Keeps embedding batches full (many small files -> one batch) while
    bounding rework after an interruption to at most one group.
    """
    groups: list[list[tuple[FileRecord, list[dict], list[str]]]] = []
    cur: list[tuple[FileRecord, list[dict], list[str]]] = []
    count = 0
    for item in prepared:
        n = len(item[1])
        if cur and count + n > EMBED_BATCH_SIZE:
            groups.append(cur)
            cur, count = [], 0
        cur.append(item)
        count += n
    if cur:
        groups.append(cur)
    return groups


def _commit_groups(
    cfg: Config,
    store: Store,
    prepared: list[tuple[FileRecord, list[dict], list[str]]],
    manifest: dict,
    dim: int | None,
    console: Console,
    stats: IndexStats,
) -> int:
    """Embed and commit prepared files group by group, checkpointing the manifest.

    Per group: embed -> per-file delete+add -> manifest update -> atomic manifest
    save. An interrupted run resumes from the manifest: committed files are
    skipped, and at most one group's worth of embedding is redone.
    """
    console.print(
        f"Loading embedding model [bold]{cfg.model}[/] "
        "(downloaded from Hugging Face on first use, then cached)…"
    )
    embedder = make_embedder(cfg)
    dim = dim or embedder.dim
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total} files"),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("indexing", total=len(prepared))
        for group in _group_prepared(prepared):
            texts = [t for _rec, _rows, ts in group for t in ts]
            vectors: list[list[float]] = []
            for i in range(0, len(texts), EMBED_BATCH_SIZE):
                vectors.extend(embedder.embed(texts[i : i + EMBED_BATCH_SIZE]))
            vi = 0
            for rec, rows, _ts in group:
                file_rows = []
                for r in rows:
                    r = dict(r)
                    r["vector"] = vectors[vi]
                    vi += 1
                    file_rows.append(r)
                store.delete_file(rec.rel_path)  # idempotent: drops stale/dup rows
                store.add(file_rows, dim)
                manifest[rec.rel_path] = _manifest_entry(rec, len(file_rows))
                stats.chunks_added += len(file_rows)
                progress.advance(task)
            _save_manifest(cfg, manifest)  # checkpoint: committed files survive a crash
    return dim


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
    """Map, chunk, embed and commit every supported file (first-time setup).

    Checkpointed per file group: as soon as a group is embedded it is written to
    the store and registered in the manifest, so an interrupted run can be
    resumed by re-running db-init (or db-refresh) without re-embedding finished
    files.
    """
    console = console or Console()
    store = Store(cfg.data_dir, cfg.model)
    if store.exists() and not force:
        manifest = _load_manifest(cfg)
        if manifest:
            records, _ = _scan_all(cfg)
            on_disk = {rec.rel_path: rec.sha256 for rec in records}
            pending = [
                p for p, h in on_disk.items() if manifest.get(p, {}).get("sha256") != h
            ]
            stale = [p for p in manifest if p not in on_disk]
            if not pending and not stale:
                raise AlreadyInitialized()
            console.print(
                "[yellow]Index exists but is incomplete — resuming: "
                "already-indexed files are skipped.[/]"
            )
            return refresh(cfg, console=console)
        raise AlreadyInitialized()
    if store.exists():
        if force:
            n = len(_load_manifest(cfg))
            if n:
                console.print(
                    f"[red]--force: dropping existing index — {n} already-indexed "
                    f"file(s) will be re-embedded from scratch.[/]"
                )
        store.drop()

    stats = IndexStats()
    records, skipped = _scan_all(cfg)
    stats.skipped = list(skipped)

    console.print(f"Chunking [bold]{len(records)}[/] file(s) in {cfg.docs_dir}…")
    prepared, skipped2 = _prepare(cfg, records, console)
    stats.skipped.extend(skipped2)
    stats.files_added = len(prepared)

    manifest: dict = {}
    if prepared:
        dim = _commit_groups(cfg, store, prepared, manifest, None, console, stats)
    else:
        dim = make_embedder(cfg).dim
        store.create([], dim)
    _save_manifest(cfg, manifest)
    store.write_meta(dim, created=True)
    return stats


def refresh(cfg: Config, console: Console | None = None) -> IndexStats:
    """Incrementally index new/changed files and drop deleted ones.

    Checkpointed per file group (see ``_commit_groups``): an interrupted run can
    be resumed by re-running db-refresh — committed files are skipped and at most
    one group's worth of embedding is redone.
    """
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

    for path, old in list(old_manifest.items()):
        if path not in new_by_path:
            store.delete_file(path)
            old_manifest.pop(path, None)
            stats.files_removed += 1
            stats.chunks_removed += int(old.get("chunks", 0))
    if stats.files_removed:
        _save_manifest(cfg, old_manifest)  # checkpoint deletions too

    if to_process:
        console.print(f"Chunking [bold]{len(to_process)}[/] changed file(s)…")
        prepared, skipped2 = _prepare(cfg, to_process, console)
        stats.skipped.extend(skipped2)
        prepared_paths = {rec.rel_path for rec, _rows, _texts in prepared}
        for rec in to_process:
            if rec.rel_path not in prepared_paths:
                old_manifest.pop(rec.rel_path, None)  # not indexable — forget it
        if prepared:
            dim = _commit_groups(
                cfg, store, prepared, old_manifest, (meta or {}).get("dim"), console, stats
            )
        else:
            dim = (meta or {}).get("dim", 0)
        _save_manifest(cfg, old_manifest)
        store.write_meta(dim, created=False)
    else:
        store.write_meta((meta or {}).get("dim", 0), created=False)
    return stats
