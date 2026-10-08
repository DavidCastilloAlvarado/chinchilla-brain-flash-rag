"""Pre-search freshness check: auto-refresh the index when it is stale.

``search`` is the hottest command, so it carries a freshness guarantee.
Before searching, at most once per ``auto_refresh_check_interval_hours``
(default 24 h), we look at the index's ``updated`` timestamp (from
``meta.json``). If the index is older than ``auto_refresh_max_age_days``
(default 7 days), we run the same incremental refresh that ``db-refresh``
runs, and only then let the search proceed.

The "last checked" marker is persisted in
``<data_dir>/freshness_state.json`` so the check itself happens at most once
per interval, even when search is invoked many times. A refresh that fails
still marks the day as checked (so we do not hammer a failing refresh on
every query), and the error propagates to the caller, which fails the search
with a clear, actionable message.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from rich.console import Console

from .config import Config
from .index import IndexStats, refresh
from .store import Store


def _now() -> datetime:
    return datetime.now(timezone.utc)


def index_age_days(meta: dict | None) -> float | None:
    """Age of the index in days, from the ``updated`` meta timestamp.

    Returns ``None`` when the meta is missing or the timestamp is absent or
    unparseable (callers treat that as stale).
    """
    if not meta:
        return None
    raw = meta.get("updated")
    if not raw:
        return None
    try:
        updated = datetime.fromisoformat(str(raw))
    except ValueError:
        return None
    if updated.tzinfo is None:
        updated = updated.replace(tzinfo=timezone.utc)
    return max(0.0, (_now() - updated).total_seconds() / 86400)


def _state_path(cfg: Config) -> Path:
    return cfg.freshness_state_path


def _load_state(cfg: Config) -> dict:
    path = _state_path(cfg)
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _save_state(cfg: Config, state: dict) -> None:
    """Atomically write the check state (tmp + rename)."""
    path = _state_path(cfg)
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
    tmp.replace(path)


def _parse_ts(raw: object) -> datetime | None:
    if not raw:
        return None
    try:
        ts = datetime.fromisoformat(str(raw))
    except ValueError:
        return None
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts


def _checked_recently(state: dict, cfg: Config) -> bool:
    """True when a freshness check already ran within the configured interval."""
    last = _parse_ts(state.get("last_checked"))
    if last is None:
        return False
    elapsed_h = (_now() - last).total_seconds() / 3600.0
    return elapsed_h < cfg.auto_refresh_check_interval_hours


def _print_stats(stats: IndexStats, console: Console) -> None:
    console.print(
        f"[bold green]✓[/] +{stats.files_added} new, ~{stats.files_updated} updated, "
        f"-{stats.files_removed} removed, {stats.files_unchanged} unchanged "
        f"({stats.chunks_added} chunks added, {stats.chunks_removed} removed)"
    )
    if stats.skipped:
        console.print(f"[yellow]{len(stats.skipped)} file(s) skipped:[/]")
        for reason in stats.skipped:
            console.print(f"  [dim]- {reason}[/]")


def maybe_auto_refresh(cfg: Config, console: Console) -> bool:
    """Run the once-per-interval freshness check before a search.

    Returns ``True`` when the index was refreshed, ``False`` when no refresh
    was needed or the check was skipped (already checked within the interval,
    or the index does not exist yet — in which case ``search`` reports
    "not initialized" as usual).

    Raises when a triggered refresh fails (e.g. model mismatch); the caller
    is expected to fail the search with an actionable message. The "checked"
    marker is persisted before the refresh starts, so a failing refresh is
    not retried on every subsequent search within the interval.
    """
    store = Store(cfg.data_dir, cfg.model)
    if not store.exists():
        return False

    state = _load_state(cfg)
    if _checked_recently(state, cfg):
        return False

    # A check is due: record it now so it runs at most once per interval,
    # even if the refresh that follows turns out to fail.
    now = _now()
    meta = store.read_meta()
    age = index_age_days(meta)
    stale = age is None or age > cfg.auto_refresh_max_age_days
    _save_state(
        cfg,
        {
            "last_checked": now.isoformat(timespec="seconds"),
            "index_age_days": round(age, 3) if age is not None else None,
            "max_age_days": cfg.auto_refresh_max_age_days,
            "check_interval_hours": cfg.auto_refresh_check_interval_hours,
            "stale": stale,
            "refreshed": False,
        },
    )

    if not stale:
        return False

    if age is None:
        console.print(
            "[yellow]Index age unknown (no valid 'updated' timestamp) — "
            "refreshing before search…[/]"
        )
    else:
        console.print(
            f"[yellow]Index is {age:.1f} days old (limit "
            f"{cfg.auto_refresh_max_age_days:g}) — refreshing before search…[/]"
        )

    stats = refresh(cfg, console=console)  # may raise: state is already marked checked
    _save_state(
        cfg,
        {
            **_load_state(cfg),
            "refreshed": True,
        },
    )
    _print_stats(stats, console)
    return True
