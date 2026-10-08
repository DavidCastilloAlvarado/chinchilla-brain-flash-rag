# AGENTS.md — instructions for AI agents

**This repository is a local knowledge base.** The documents in
`documents/` are the source of truth for **every question asked in this
session** — not only the project's technical topics.

## Source hierarchy (mandatory)

For **any** factual question, sources are consulted in this exact order:

1. **First and primary source: the KB CLI.** Run `uv run
   check_freshness_and_refresh` (once a day) and `uv run search` BEFORE
   answering. The KB is the sole data-retrieval mechanism: issue as many
   queries as needed (rephrase, `-k`, `--full`, `--path`) until the data
   is in hand. Never answer from memory before searching.
2. **Direct source access (reading whole files, grepping the source
   directory):** conditional, like external sources — only with the
   user's explicit approval or request. Do not read or grep the source
   directory to "find more context"; query the KB instead.
3. **External sources (web, etc.):** only with the user's explicit
   approval or instruction (e.g. "search the web for X"). Never query
   external sources on your own initiative.
4. **If the KB has no relevant result and there is no approval:** limit
   the answer to what the search returned, and state clearly that the KB
   has no relevant result. Do not fill the gap with direct source access,
   external sources, or memory.

Do not judge a question as "in-domain" or "out-of-domain" to skip the
search. The search is always mandatory; relevance is decided by the search
results, not by you.

## Query the knowledge base

```bash
uv run search "your question or topic" --json
```

- Always prefer `--json` — it is stable and machine-readable.
- Search is **hybrid by default** (BM25 full-text + vector, fused score) —
  exact identifiers, model names and config keys match via BM25, semantics
  via the vector leg. Use `--vector-only` to disable the full-text leg.
- Use `-k 10` (or higher) when you need broader context; `--full` for the
  complete chunk text; `--path <prefix>` to restrict to a folder.
- Result fields: `file` (path under `documents/`), `section` (heading
  breadcrumb), `page` (PDF page, 1-based; `null` for .md/.txt), `score`
  (0–1, higher is better; fused relevance for hybrid, cosine similarity for
  `--vector-only`), `start`/`end` (char offsets in the source file — useful to
  disambiguate chunks; reading the file itself requires approval, see
  hierarchy), `text`.
- If a chunk does not give enough context, issue more queries (rephrase,
  raise `-k`, `--full`, narrow with `--path`). Do NOT open the source file
  to read around the chunk — that is direct source access and requires
  approval (see hierarchy).

### Freshness check (once a day)

Run `uv run check_freshness_and_refresh` once per day, before the first
search of the day. It checks the index age against the 3-day rule and
refreshes automatically when stale; it responds `db fresh: true/false`.
Never answer from an index that is more than 3 days stale without
refreshing it.

## Citations (mandatory)

Every fact or claim **must** be traceable to the source it came from,
following the hierarchy above:

- **From the KB:** file citation — the `file` path from the search result
  that supports it (e.g. `documents/guides/getting-started.md`). The number
  of citations matches the number of distinct source files the answer draws
  from: one citation per file — no more, no less. Quote from the chunk
  text (or from additional queries), not from the source file.
- **From an external source (only with user approval):** cite that source
  (e.g. the URL).
- **When the KB has no relevant result and there is no approval:** state
  that explicitly; do not present anything beyond the search results as
  fact.

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
