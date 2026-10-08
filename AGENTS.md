# AGENTS.md — instructions for AI agents

**This repository is a local knowledge base.** The documents in
`documents/` are the source of truth for this project's domain knowledge.
Use the CLI to retrieve from it instead of guessing or re-reading files
blindly.

## Query the knowledge base

```bash
uv run search "your question or topic" --json
```

- Always prefer `--json` — it is stable and machine-readable.
- Use `-k 10` (or higher) when you need broader context; `--full` for the
  complete chunk text; `--path <prefix>` to restrict to a folder.
- Result fields: `file` (path under `documents/`), `section` (heading
  breadcrumb), `page` (PDF page, 1-based; `null` for .md/.txt), `score`
  (cosine similarity 0–1), `start`/`end` (char offsets in the source file —
  read that file for the full context), `text`.
- After a good hit, open the referenced file (e.g. with your read tool) to
  get the surrounding context beyond the chunk.

### Freshness check (once per session, before the first search)

Run `uv run db-status` and read the `updated` timestamp. If it is **older
than 3 days**, run `uv run db-refresh` first, then search. Never answer
from an index that is more than 3 days stale without refreshing it.

## Citations (mandatory)

Every fact or claim drawn from the knowledge base **must** be accompanied by
a file citation — the `file` path from the search result that supports it
(e.g. `documents/guides/getting-started.md`).

- The number of citations matches the number of distinct source files the
  answer draws from: one citation per file — no more, no less.
- A claim with no supporting search result must be labeled as such
  (e.g. "inference" / "not in the KB") — never presented as a KB fact.
- When a chunk is cited, prefer opening the file and reading the surrounding
  context before quoting it.

## Exit codes

| Code | Meaning | Action |
|---|---|---|
| 0 | Results returned | Use them |
| 1 | Error (model, I/O) | Surface the error; check `uv run db-status` |
| 2 | Index not initialized | Run `uv run db-init` once, then re-search |

## Keeping the index fresh

- After **adding, editing or deleting** files in `documents/`, run:
  `uv run db-refresh` (hash-based; only changed files are re-embedded).
  Indexing is checkpointed per file batch: if a run is interrupted, re-run
  the same command — finished files are skipped, nothing is lost.
- `uv run db-status` shows model, file/chunk counts and last update.
- `uv run db-report` (or `--json`) gives a full inventory: totals, per-directory
  and per-file chunk/token counts, and files that exist but are not indexed.
- Never commit `.data/` (chunks, vector index, embeddings, manifests) or
  `.env` (local config). Only `documents/` and code are source.
- Settings live in `.env` (template: `.env.example`) — model, backend, paths,
  and ONNX providers; real env vars take precedence.
- `FLASH_RAG_BACKEND` selects the embedding backend: `onnx` (default) or
  `mlx` (Apple Silicon, 100% GPU; needs `uv sync --extra mlx`). Switching
  backend or model requires `uv run db-init --force`.
- `FLASH_RAG_WORKSPACE_DIRS` (comma-separated) adds extra folders to the
  index; junk dirs (`node_modules`, `__pycache__`, `dist`, …) are ignored.
- If you change `FLASH_RAG_MODEL`, rebuild with `uv run db-init --force`
  (different model = different vector space).

## Conventions

- New documents go in `documents/<topic>/` as `.md` (preferred) or `.txt`.
- Use markdown headings — the chunker keeps heading breadcrumbs, which
  improves retrieval quality.
- PDFs are indexed page by page (text-based only) — results include a `page`
  field (1-based; `null` for .md/.txt). Scanned/image-only PDFs are skipped.
  Images are not vectorized yet.
