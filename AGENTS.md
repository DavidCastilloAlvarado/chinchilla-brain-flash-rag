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

## Exit codes

| Code | Meaning | Action |
|---|---|---|
| 0 | Results returned | Use them |
| 1 | Error (model, I/O) | Surface the error; check `uv run db-status` |
| 2 | Index not initialized | Run `uv run db-init` once, then re-search |

## Keeping the index fresh

- After **adding, editing or deleting** files in `documents/`, run:
  `uv run db-refresh` (hash-based; only changed files are re-embedded).
- `uv run db-status` shows model, file/chunk counts and last update.
- `uv run db-report` (or `--json`) gives a full inventory: totals, per-directory
  and per-file chunk/token counts, and files that exist but are not indexed.
- Never commit `.data/` (chunks, vector index, embeddings, manifests) or
  `.env` (local config). Only `documents/` and code are source.
- Settings live in `.env` (template: `.env.example`) — model, paths, and
  ONNX providers; real env vars take precedence.
- If you change `FLASH_RAG_MODEL`, rebuild with `uv run db-init --force`
  (different model = different vector space).

## Conventions

- New documents go in `documents/<topic>/` as `.md` (preferred) or `.txt`.
- Use markdown headings — the chunker keeps heading breadcrumbs, which
  improves retrieval quality.
- PDFs are indexed page by page (text-based only) — results include a `page`
  field (1-based; `null` for .md/.txt). Scanned/image-only PDFs are skipped.
  Images are not vectorized yet.
