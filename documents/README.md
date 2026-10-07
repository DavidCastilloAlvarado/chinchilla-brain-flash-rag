# documents/

This folder is the **single source of truth** for this knowledge base.
Everything here is committed; the vector index is not.

## What gets indexed

| Type | Status |
|---|---|
| `.md`, `.markdown`, `.txt` (any depth, any number of folders) | ✅ vectorized |
| `.pdf`, images, other binaries | ⏸️ allowed to live here, **not** vectorized yet |

## Conventions

- Organize by topic: one folder per domain, subfolders as needed.
- Use markdown headings (`#`, `##`, `##`) — the chunker keeps the heading
  breadcrumb, so well-structured docs retrieve much better.
- After adding or editing files, run `uv run db-refresh` to sync the index.

## What never gets committed

Chunks, the LanceDB index, embeddings and manifests live in `.data/`
(git-ignored). Any clone can rebuild them with `uv run db-init`.
