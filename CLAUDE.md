# Claude Code instructions

This repository is a **local knowledge base**. Query it before answering
domain questions:

```bash
uv run search "topic or question" --json
```

- Exit code 2 → index not initialized → run `uv run db-init`, then re-search.
- After changing files in `documents/` → run `uv run db-refresh`.
- Never commit `.data/`. Only `documents/` and code are source.

Full instructions: see [AGENTS.md](AGENTS.md).
