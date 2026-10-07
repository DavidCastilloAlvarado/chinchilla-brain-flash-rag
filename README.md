# chinchilla-brain-flash-rag

A **local knowledge base** that lives inside this repository. The original
documents sit in [`documents/`](documents/) (committed); chunks, the vector
index and embeddings sit in `.data/` (never committed) and are rebuilt
locally. Any harness — GitHub Copilot, DSH, Codex, Claude, or a human —
queries it through one small CLI:

```bash
uv run search "how do I refresh the index"
```

100% local: no cloud APIs, no API keys, no server. The embedding model
(`nomic-ai/nomic-embed-text-v1.5`, 137M params, Apache-2.0) is downloaded once from
Hugging Face and cached in `~/.cache/fastembed`.

## Quick start

```bash
uv sync           # install dependencies into .venv
uv run db-init    # scan documents/, chunk + embed everything (one-time)
uv run search "chunking strategy"
```

## Commands

| Command | What it does |
|---|---|
| `uv run search "query"` | Semantic search over the index. Flags: `-k/--top-k`, `--json`, `--path PREFIX`, `--full` |
| `uv run db-init` | First-time build: map, chunk and embed all supported files. `--force` rebuilds from scratch |
| `uv run db-refresh` | Incremental sync: hash-diff files, embed only what's new/changed, drop deleted |
| `uv run db-status` | Model, file/chunk counts, timestamps |

If `search` runs before `db-init`, it prints a hint to run `uv run db-init`
and exits with code **2** (so agents can detect and react).

## What gets indexed

`.md`, `.markdown`, `.txt` at any depth under `documents/`. PDFs and images
may live in `documents/` but are **not** vectorized yet (parsers on the
roadmap). Files > 2 MB are skipped with a warning.

## How it works

```
documents/  ──►  scan (SHA-256)  ──►  chunk (markdown-aware, ~512 tokens,
                                        ~12% overlap, heading breadcrumbs)
          ──►  embed (fastembed / ONNX Runtime, local HF model)
          ──►  LanceDB (embedded vector DB, .data/lancedb)
```

- **Vector engine — LanceDB.** Embedded, zero-server, Rust core, columnar
  Lance format, HNSW/IVF-PQ indexes, metadata filtering. It is the standard
  "local Pinecone" alternative (Pinecone itself is cloud-only; its local mode
  is a dev clone of the managed API).
- **Embeddings — fastembed.** ONNX Runtime with quantized weights: no torch,
  no GPU, the most efficient local option. Model is swappable via
  `FLASH_RAG_MODEL` (e.g. `BAAI/bge-small-en-v1.5` for speed).
- **Chunking.** Markdown is split on the heading hierarchy (code fences
  respected); every chunk keeps a `section` breadcrumb (`H1 > H2 > H3`).
  Chunks are ≤ 512 tokens with a ~64-token overlap. 256–512 tokens with
  10–15% overlap is the current RAG best practice.
- **Incremental refresh.** `.data/manifest.json` maps each file to its
  SHA-256; `db-refresh` re-embeds only new/changed files and deletes chunks
  of removed files.

## Configuration (env vars)

| Variable | Default | Meaning |
|---|---|---|
| `FLASH_RAG_MODEL` | `nomic-ai/nomic-embed-text-v1.5` | fastembed model id |
| `FLASH_RAG_DOCS_DIR` | `<root>/documents` | documents to index |
| `FLASH_RAG_DATA_DIR` | `<root>/.data` | where the index lives |

## What is committed vs. not

| Committed | Not committed (git-ignored) |
|---|---|
| `documents/**` (original files) | `.data/` (LanceDB index, chunks, embeddings, manifests) |
| code, `pyproject.toml`, `uv.lock` | `.venv/`, `__pycache__/` |

## For AI agents

See **[AGENTS.md](AGENTS.md)** — it tells any harness (Copilot, DSH, Codex,
Claude…) how to use this repo as a knowledge source. Also mirrored in
`.github/copilot-instructions.md` and `CLAUDE.md`.

## Research notes (what we looked at)

- **Embedding models (2025/26 local RAG rankings):** BGE-M3 (hybrid
  dense+sparse, multilingual), Qwen3-Embedding family (top MTEB, 32k
  context), Nomic Embed v1.5/v2, EmbeddingGemma-300M, bge-small-en-v1.5.
  We picked **nomic-embed-text-v1.5**: best quality-per-second on CPU for
  English docs, Apache-2.0, and available in fastembed's ONNX form.
- **Vector engines:** LanceDB vs Chroma vs sqlite-vec vs Qdrant Edge.
  Chroma's HNSW is memory-resident (6–8 GB at 1M vectors); sqlite-vec has no
  HNSW/IVF; **LanceDB** is the embedded option with real ANN indexes and
  the best insert throughput in the comparisons.
- **Similar projects:** [`ariel-frischer/kb`](https://github.com/ariel-frischer/kb)
  (CLI RAG, sqlite-vec, hybrid BM25+vector, incremental hashing — closest
  sibling), [`lyonzin/knowledge-rag`](https://github.com/lyonzin/knowledge-rag)
  (MCP-first local RAG, FastEmbed+Chroma), [`kevwan/rag-agent`](https://github.com/kevwan/rag-agent)
  (uv-managed markdown RAG with Milvus).
