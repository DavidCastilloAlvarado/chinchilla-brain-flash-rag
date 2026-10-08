# GitHub Copilot instructions

This repository is a **local knowledge base**. Before answering questions
about this project's domain, retrieve from it:

```bash
uv run search "topic or question" --json
```

- Exit code 2 means the index is not initialized: run `uv run db-init`
  first, then re-search.
- `search` auto-refreshes the index when it is older than 7 days (checked
  at most once a day; state in `.data/freshness_state.json`) — no manual
  refresh needed for that.
- After adding/editing files in `documents/`, run `uv run db-refresh`.
- Never commit `.data/` (index, chunks, embeddings) — only `documents/`
  and code.

Full agent instructions: see [AGENTS.md](../../AGENTS.md).
