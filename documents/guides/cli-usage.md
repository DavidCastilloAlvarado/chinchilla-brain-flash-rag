# CLI reference

All commands run from the repository root (or any subdirectory) with `uv run`.

## `search`

```
uv run search "query" [options]
```

| Option | Default | Description |
|---|---|---|
| `-k, --top-k N` | 5 | Number of results (1–50) |
| `--json` | off | Emit a single JSON object (for agents/harnesses) |
| `--path PREFIX` | – | Only return chunks whose file path starts with PREFIX |
| `--full` | off | Print the full chunk text instead of a ~400-char snippet |

### JSON output shape

```json
{
  "query": "chunking strategy",
  "model": "nomic-embed-text-v1.5",
  "count": 2,
  "results": [
    {
      "rank": 1,
      "score": 0.7312,
      "file": "reference/architecture.md",
      "section": "Architecture > Chunking",
      "start": 1204,
      "end": 2871,
      "text": "..."
    }
  ]
}
```

`score` is cosine similarity in [0, 1]. `start`/`end` are character offsets in
the source file — open the file and jump straight to the passage.

### Exit codes

| Code | Meaning |
|---|---|
| 0 | Results returned (possibly empty) |
| 1 | Error (bad model, I/O, …) |
| 2 | Index not initialized → run `uv run db-init` |

## `db-init`

```
uv run db-init [--force]
```

Maps `documents/`, chunks and embeds every supported file, and creates the
LanceDB index in `.data/`. Refuses to run over an existing index unless
`--force` is given (use `db-refresh` for updates).

## `db-refresh`

```
uv run db-refresh
```

Hashes every file (SHA-256) and diffs against the manifest:

- **new** file → chunked + embedded + added
- **changed** file (hash differs) → old chunks deleted, new ones added
- **deleted** file → its chunks are removed
- **unchanged** → untouched (no re-embedding)

## `db-status`

```
uv run db-status
```

Shows model, vector dimension, indexed file count, chunk count, and
created/updated timestamps.

## `db-report`

```
uv run db-report [--json]
```

Full inventory of the knowledge base:

- totals: indexed files, chunks, total tokens, index size on disk
- per-directory breakdown (files / chunks / tokens)
- per-file breakdown, sorted by token count
- files that exist in `documents/` but are **not** indexed (unsupported
  suffixes like `.pdf`/images, empty or oversized files)
