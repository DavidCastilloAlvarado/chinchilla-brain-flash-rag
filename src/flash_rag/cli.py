"""CLI entry points: ``search``, ``db-init``, ``db-refresh``, ``db-status``.

Each command is a standalone console script (see pyproject [project.scripts]),
so agents and humans can call them directly:

    uv run search "query" [--json] [-k 5] [--path prefix] [--full]
    uv run db-init [--force]
    uv run db-refresh
    uv run db-status

Exit codes: 0 = ok, 1 = error, 2 = index not initialized yet.
"""

from __future__ import annotations

import json

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from . import __version__
from .config import load_config
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
) -> None:
    """Semantic search over documents/ using the local vector index."""
    cfg = load_config()
    try:
        data = run_search(cfg, query, top_k, path)
    except NotInitialized:
        _print_not_initialized()
        raise typer.Exit(EXIT_NOT_INITIALIZED)
    except Exception as exc:  # noqa: BLE001 - report any failure cleanly
        _fail(f"search failed: {exc}")
    if as_json:
        print(json.dumps(data, indent=2, ensure_ascii=False))
        return
    _print_results(data, full)


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
    """Show index status: model, file/chunk counts, last update."""
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

    table = Table(show_header=False, box=None, pad_edge=False)
    table.add_column(justify="right", style="cyan")
    table.add_column()
    table.add_row("version", __version__)
    table.add_row("model", str(meta.get("model", "?")))
    table.add_row("dim", str(meta.get("dim", "?")))
    table.add_row("files indexed", str(len(manifest)))
    table.add_row("chunks", str(store.count()))
    table.add_row("created", str(meta.get("created", "?")))
    table.add_row("updated", str(meta.get("updated", "?")))
    table.add_row("data dir", str(cfg.data_dir))
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


if __name__ == "__main__":
    search()
