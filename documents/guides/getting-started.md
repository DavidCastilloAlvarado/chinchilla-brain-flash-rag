# Getting started

This repository is a **local knowledge base**. The original documents live in
`documents/` (committed); the vector index lives in `.data/` (never committed)
and is rebuilt locally with `uv`.

## Prerequisites

- [uv](https://docs.astral.sh/uv/) (`curl -LsSf https://astral.sh/uv/install.sh | sh`)
- Python 3.11+ (uv manages it automatically)
- ~1 GB of disk for the embedding model (downloaded once, cached in `~/.cache/fastembed`)

## Setup

```bash
uv sync                 # create .venv and install dependencies
uv run db-init          # scan documents/, chunk + embed everything (one-time)
```

The first `db-init` downloads the embedding model
(`nomic-ai/nomic-embed-text-v1.5`, 137M params, Apache-2.0) from Hugging Face.

## Search

```bash
uv run search "how do I refresh the index"
uv run search "embedding model" --json      # machine-readable, for agents
uv run search "chunking" -k 10 --full       # more results, full text
uv run search "cli" --path guides/          # restrict to a path prefix
```

## Keep the index fresh

```bash
uv run db-refresh       # add new/changed files, drop deleted ones (hash-based)
uv run db-status        # model, file/chunk counts, last update
```

## Changing the embedding model

```bash
FLASH_RAG_MODEL="BAAI/bge-small-en-v1.5" uv run db-init --force
```

> ⚠️ Changing the model changes the vector space — always rebuild with
> `db-init --force` after switching models.
