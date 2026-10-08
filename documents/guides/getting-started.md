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

## Using the GPU on Apple Silicon (M1/M2/M3/M4)

By default embeddings run on the CPU via ONNX Runtime. On a Mac, the best
GPU path is the **MLX backend** — it runs the whole model natively on the
GPU/ANE with no conversion step:

```bash
uv sync --extra mlx                      # installs mlx + mlx-embeddings
export FLASH_RAG_BACKEND=mlx             # or put it in .env
uv run db-init --force                   # rebuild once (vectors differ slightly)
```

MLX loads the same Hugging Face weights directly (no ONNX export), so the
same model ids work (`nomic-ai/nomic-embed-text-v1.5`, …). Note that
`mlx-embeddings` doesn't ship a NomicBert implementation, so flash-rag
bundles a small port of it (`src/flash_rag/nomic_bert.py`) — nomic models
just work.

> ⚠️ Why not CoreML? The ONNX Runtime CoreML provider only converts part of
> the graph (e.g. ~40% of nodes for nomic-embed); the rest falls back to the
> CPU, and the many GPU↔CPU hand-offs can make it *slower* than CPU-only.
> MLX has no such split. `FLASH_RAG_PROVIDERS=CoreMLExecutionProvider` still
> works if you want to try it.

Switching backends changes the vectors slightly (different kernels) — rebuild
with `db-init --force` when you switch. Switching models always requires a
rebuild as well.
