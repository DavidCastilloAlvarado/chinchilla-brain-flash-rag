"""CLI entry points: ``search``, ``db-init``, ``db-refresh``, ``db-status``,
``db-report``, ``check_freshness_and_refresh``.

Each command is a standalone console script (see pyproject [project.scripts]),
so agents and humans can call them directly:

    uv run search "query" [--json] [-k 5] [--path prefix] [--full]
    uv run db-init [--force]
    uv run db-refresh
    uv run db-status
    uv run db-report [--json]
    uv run check_freshness_and_refresh [--json]

``search`` first runs a freshness check that happens at most once per day
(state: ``.data/freshness_state.json``): when the index's last update is
older than ``FLASH_RAG_AUTO_REFRESH_MAX_AGE_DAYS`` (default 7 days), an
incremental refresh runs before the search.

Exit codes: 0 = ok, 1 = error, 2 = index not initialized yet.
"""

from __future__ import annotations

import json

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from . import __version__
from .config import Config, load_config
from .freshness import index_age_days, maybe_auto_refresh
from .index import (
    AlreadyInitialized,
    ModelMismatch,
    NotInitialized,
    build,
    refresh,
)
from .search import run_search
from .store import Store
from .report import build_report

console = Console()
err_console = Console(stderr=True)

EXIT_NOT_INITIALIZED = 2


def _fail(message: str, code: int = 1) -> None:
    err_console.print(f"[bold red]✗[/] {message}")
    raise typer.Exit(code)


def _print_not_initialized() -> None:
    err_console.print(
        Panel.fit(
            "[bold yellow]The knowledge base is not initialized yet.[/]\n\n"
            "Run:\n"
            "  [bold]uv run db-init[/]\n\n"
            "This scans [bold]documents/[/], chunks and embeds every .md/.txt/.pdf file "
            "locally (one-time; the embedding model is downloaded from Hugging "
            "Face on first use).",
            title="flash-rag",
            border_style="yellow",
        )
    )


def _snippet(text: str, limit: int = 400) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[: limit - 1].rstrip() + "…"


def _print_results(data: dict, full: bool) -> None:
    if not data["results"]:
        console.print("[yellow]No results.[/]")
        return
    for r in data["results"]:
        page = r.get("page")
        loc = r["file"] + (f" › p. {page}" if page else "")
        loc += f"  ›  {r['section']}" if r["section"] else ""
        console.print(f"[bold]{r['rank']}.[/] [dim]{r['score']:.3f}[/] [bold cyan]{loc}[/]")
        body = r["text"] if full else _snippet(r["text"])
        console.print("   " + body.replace("\n", "\n   "))
        console.print()


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------

_search_app = typer.Typer(add_completion=False)


@_search_app.command()
def _search_cmd(
    query: str = typer.Argument(..., help="Natural-language query."),
    top_k: int = typer.Option(5, "--top-k", "-k", min=1, max=50, help="Number of results."),
    as_json: bool = typer.Option(False, "--json", help="Machine-readable JSON output (for agents)."),
    path: str = typer.Option(None, "--path", help="Only results whose file path starts with this prefix."),
    full: bool = typer.Option(False, "--full", help="Print full chunk text instead of a snippet."),
    vector_only: bool = typer.Option(
        False, "--vector-only", help="Disable the BM25 full-text leg (pure vector search)."
    ),
) -> None:
    """Hybrid search (BM25 + vector) over documents/ using the local index."""
    cfg = load_config()
    _maybe_auto_refresh(cfg, as_json=as_json)
    try:
        data = run_search(cfg, query, top_k, path, vector_only=vector_only)
    except NotInitialized:
        _print_not_initialized()
        raise typer.Exit(EXIT_NOT_INITIALIZED)
    except Exception as exc:  # noqa: BLE001 - report any failure cleanly
        _fail(f"search failed: {exc}")
    if as_json:
        print(json.dumps(data, indent=2, ensure_ascii=False))
        return
    _print_results(data, full)


def _maybe_auto_refresh(cfg: Config, as_json: bool) -> None:
    """Pre-search freshness hook.

    At most once per ``FLASH_RAG_AUTO_REFRESH_CHECK_INTERVAL_HOURS`` (default
    24 h, state in ``.data/freshness_state.json``), checks the index age and
    runs the incremental refresh first when the index is older than
    ``FLASH_RAG_AUTO_REFRESH_MAX_AGE_DAYS`` (default 7 days). A failed
    refresh fails the search (exit 1) with an actionable message instead of
    silently serving an unrefreshed index. With ``--json`` all of this goes
    to stderr so stdout stays pure JSON.
    """
    out = err_console if as_json else console
    try:
        maybe_auto_refresh(cfg, out)
    except ModelMismatch as exc:
        _fail(f"auto-refresh failed: {exc}")
    except Exception as exc:  # noqa: BLE001 - report any failure cleanly
        _fail(
            f"auto-refresh failed: {exc}. Retry with "
            f"[bold]uv run db-refresh[/] (the daily check is marked done and "
            f"runs again in ~24 h)."
        )


def search() -> None:
    _search_app()


# ---------------------------------------------------------------------------
# db-init
# ---------------------------------------------------------------------------

_init_app = typer.Typer(add_completion=False)


@_init_app.command()
def _init_cmd(
    force: bool = typer.Option(
        False, "--force", help="Drop the existing index and rebuild from scratch."
    ),
) -> None:
    """Map, chunk and embed every supported file in documents/ (first-time setup)."""
    cfg = load_config()
    if not cfg.docs_dir.is_dir():
        _fail(f"documents directory not found: {cfg.docs_dir}")
    try:
        stats = build(cfg, force=force)
    except AlreadyInitialized:
        _fail(
            "Index already exists. Use [bold]uv run db-refresh[/] to update it, "
            "or [bold]uv run db-init --force[/] to rebuild from scratch."
        )
    except Exception as exc:  # noqa: BLE001
        _fail(f"db-init failed: {exc}")

    console.print(
        f"[bold green]✓[/] Indexed [bold]{stats.files_added}[/] file(s), "
        f"[bold]{stats.chunks_added}[/] chunk(s) → {cfg.data_dir}"
    )
    if stats.skipped:
        console.print(f"[yellow]{len(stats.skipped)} file(s) skipped:[/]")
        for reason in stats.skipped:
            console.print(f"  [dim]- {reason}[/]")
    console.print("Search with: [bold]uv run search \"your query\"[/]")


def db_init() -> None:
    _init_app()


# ---------------------------------------------------------------------------
# db-refresh
# ---------------------------------------------------------------------------

_refresh_app = typer.Typer(add_completion=False)


@_refresh_app.command()
def _refresh_cmd() -> None:
    """Incrementally index new/changed files and remove deleted ones."""
    cfg = load_config()
    try:
        stats = refresh(cfg)
    except NotInitialized:
        _print_not_initialized()
        raise typer.Exit(EXIT_NOT_INITIALIZED)
    except ModelMismatch as exc:
        _fail(str(exc))
    except Exception as exc:  # noqa: BLE001
        _fail(f"db-refresh failed: {exc}")

    console.print(
        f"[bold green]✓[/] +{stats.files_added} new, ~{stats.files_updated} updated, "
        f"-{stats.files_removed} removed, {stats.files_unchanged} unchanged "
        f"({stats.chunks_added} chunks added, {stats.chunks_removed} removed)"
    )
    if stats.skipped:
        console.print(f"[yellow]{len(stats.skipped)} file(s) skipped:[/]")
        for reason in stats.skipped:
            console.print(f"  [dim]- {reason}[/]")


def db_refresh() -> None:
    _refresh_app()


# ---------------------------------------------------------------------------
# db-status
# ---------------------------------------------------------------------------

_status_app = typer.Typer(add_completion=False)


@_status_app.command()
def _status_cmd() -> None:
    """Show index status: model, file/chunk counts, last update, freshness check."""
    cfg = load_config()
    store = Store(cfg.data_dir, cfg.model)
    meta = store.read_meta()
    if not store.exists():
        _print_not_initialized()
        raise typer.Exit(EXIT_NOT_INITIALIZED)

    manifest = {}
    if cfg.manifest_path.is_file():
        try:
            manifest = json.loads(cfg.manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            pass

    freshness = {}
    if cfg.freshness_state_path.is_file():
        try:
            freshness = json.loads(cfg.freshness_state_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            pass

    table = Table(show_header=False, box=None, pad_edge=False)
    table.add_column(justify="right", style="cyan")
    table.add_column()
    table.add_row("version", __version__)
    table.add_row("model", str(meta.get("model", "?")))
    table.add_row("backend", cfg.backend)
    table.add_row("dim", str(meta.get("dim", "?")))
    table.add_row("files indexed", str(len(manifest)))
    table.add_row("chunks", str(store.count()))
    table.add_row("created", str(meta.get("created", "?")))
    table.add_row("updated", str(meta.get("updated", "?")))
    if freshness:
        table.add_row(
            "freshness check",
            f"last {freshness.get('last_checked', '?')}, "
            f"stale={'yes' if freshness.get('stale') else 'no'}, "
            f"auto-refreshed={'yes' if freshness.get('refreshed') else 'no'}",
        )
    table.add_row("data dir", str(cfg.data_dir))
    if cfg.workspace_dirs:
        table.add_row("workspace dirs", ", ".join(str(ws) for ws in cfg.workspace_dirs))
    console.print(table)


def db_status() -> None:
    _status_app()


# ---------------------------------------------------------------------------
# db-report
# ---------------------------------------------------------------------------

_report_app = typer.Typer(add_completion=False)


def _print_report(data: dict) -> None:
    console.print("[bold]Knowledge base report[/]")
    unindexed_note = (
        f", [yellow]{data['files_unindexed']} unindexed[/]"
        if data["files_unindexed"]
        else ""
    )
    console.print(f"  model      {data['model']} ({data['dim']}-dim)")
    console.print(f"  files      {data['files_indexed']} indexed{unindexed_note}")
    console.print(f"  chunks     {data['chunks']}")
    console.print(f"  tokens     {data['total_tokens']:,}")
    console.print(f"  index size {data['index_bytes'] / (1024 * 1024):.1f} MB")
    console.print(f"  updated    {data['updated']}")
    console.print()

    t = Table(title="By directory")
    t.add_column("dir")
    t.add_column("files", justify="right")
    t.add_column("chunks", justify="right")
    t.add_column("tokens", justify="right")
    for d, st in sorted(data["by_dir"].items()):
        t.add_row(d, str(st["files"]), str(st["chunks"]), f"{st['tokens']:,}")
    console.print(t)

    t2 = Table(title="By file")
    t2.add_column("file")
    t2.add_column("chunks", justify="right")
    t2.add_column("tokens", justify="right")
    for row in data["by_file"]:
        t2.add_row(row["file"], str(row["chunks"]), f"{row['tokens']:,}")
    console.print(t2)

    if data["unindexed"]:
        console.print(f"[yellow]{len(data['unindexed'])} unindexed file(s):[/]")
        for u in data["unindexed"][:10]:
            console.print(f"  [dim]- {u['file']} ({u['reason']})[/]")
        if len(data["unindexed"]) > 10:
            console.print(f"  [dim]… and {len(data['unindexed']) - 10} more[/]")


@_report_app.command()
def _report_cmd(
    as_json: bool = typer.Option(
        False, "--json", help="Machine-readable JSON output (for agents)."
    ),
) -> None:
    """Report: file/chunk/token totals plus per-directory and per-file breakdown."""
    cfg = load_config()
    try:
        data = build_report(cfg)
    except NotInitialized:
        _print_not_initialized()
        raise typer.Exit(EXIT_NOT_INITIALIZED)
    except Exception as exc:  # noqa: BLE001
        _fail(f"db-report failed: {exc}")
    if as_json:
        print(json.dumps(data, indent=2, ensure_ascii=False))
        return
    _print_report(data)


def db_report() -> None:
    _report_app()


# ---------------------------------------------------------------------------
# check_freshness_and_refresh
# ---------------------------------------------------------------------------

FRESHNESS_MAX_AGE_DAYS = 3.0

_fresh_app = typer.Typer(add_completion=False)


def _run_refresh(cfg) -> None:
    """Run an incremental refresh and print its stats (shared with db-refresh)."""
    try:
        stats = refresh(cfg)
    except ModelMismatch as exc:
        _fail(str(exc))
    except Exception as exc:  # noqa: BLE001
        _fail(f"refresh failed: {exc}")
    console.print(
        f"[bold green]✓[/] +{stats.files_added} new, ~{stats.files_updated} updated, "
        f"-{stats.files_removed} removed, {stats.files_unchanged} unchanged "
        f"({stats.chunks_added} chunks added, {stats.chunks_removed} removed)"
    )
    if stats.skipped:
        console.print(f"[yellow]{len(stats.skipped)} file(s) skipped:[/]")
        for reason in stats.skipped:
            console.print(f"  [dim]- {reason}[/]")


@_fresh_app.command()
def _fresh_cmd(
    as_json: bool = typer.Option(False, "--json", help="Machine-readable JSON output (for agents)."),
) -> None:
    """Check index age (3-day rule) and refresh automatically when stale.

    Responds ``db fresh: true`` or ``db fresh: false``; when stale it runs
    the incremental refresh internally and ends with ``db fresh: true``.
    """
    cfg = load_config()
    store = Store(cfg.data_dir, cfg.model)
    if not store.exists():
        _print_not_initialized()
        raise typer.Exit(EXIT_NOT_INITIALIZED)

    meta = store.read_meta()
    age = index_age_days(meta)
    fresh = age is not None and age <= FRESHNESS_MAX_AGE_DAYS

    if as_json:
        result: dict = {
            "db_fresh": fresh,
            "updated": (meta or {}).get("updated"),
            "age_days": round(age, 2) if age is not None else None,
            "max_age_days": FRESHNESS_MAX_AGE_DAYS,
            "refreshed": False,
        }
        if not fresh:
            _run_refresh(cfg)
            result["refreshed"] = True
            result["db_fresh"] = True
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return

    if fresh:
        console.print("db fresh: true")
        return

    console.print("db fresh: false")
    _run_refresh(cfg)
    console.print("db fresh: true")


def check_freshness_and_refresh() -> None:
    _fresh_app()


if __name__ == "__main__":
    search()
