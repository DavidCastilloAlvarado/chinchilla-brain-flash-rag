# Architecture

## Pipeline

```
documents/            .data/ (git-ignored)
├── guides/*.md   ──►  ├── lancedb/chunks.lance   (vectors + text + metadata)
├── notes/*.txt    │   ├── manifest.json          (file hash → index state)
└── ...            │   └── meta.json              (model, dim, timestamps)
        │
        ▼
  scan (hash) → chunk (markdown-aware) → embed (fastembed/ONNX) → LanceDB
```

## Component choices (researched 2025/26)

| Layer | Choice | Why |
|---|---|---|
| Vector engine | **LanceDB** | Embedded (zero-server), Rust core, columnar Lance format, HNSW/IVF-PQ, metadata filters, versioned writes. The standard "local Pinecone" alternative — Pinecone itself is cloud-only. |
| Embeddings | **fastembed** (ONNX Runtime) | Quantized ONNX weights, CPU-only, no torch dependency → smallest footprint and fastest cold start of the local options. |
| Model | **nomic-ai/nomic-embed-text-v1.5** | 137M params, 768 dims, 8k context, Apache-2.0. The quality/speed sweet spot for local RAG. |
| Chunking | custom, structure-aware | Markdown split on heading hierarchy with breadcrumbs; ~512-token chunks with ~12% overlap (current best practice: 256–512 tokens, 10–15% overlap). |
| Package manager | **uv** | One command (`uv run …`) resolves env + deps; no venv juggling for agents. |

## Chunking

- Markdown: sections are split at headings (code fences respected); each chunk
  carries a `section` breadcrumb like `Architecture > Chunking`.
- Long sections/paragraphs are packed greedily to ≤ 512 tokens; the tail of a
  chunk (~64 tokens) is repeated at the start of the next one so context
  survives boundaries.
- Plain text: paragraph-based, same packing.
- Embedding input is prefixed with `file_path > breadcrumb` so the vector
  carries where the text lives; the stored text stays clean.

## Incremental refresh

`db-refresh` compares SHA-256 hashes against `.data/manifest.json`, so a
typical run re-embeds only what changed. Deleting a file removes its chunks.

## What is committed

Only `documents/` (the original files) and the code. `.data/` — chunks,
vectors, embeddings, manifests — is git-ignored and rebuilt with
`uv run db-init`.

## Roadmap

- PDF + image ingestion (parsers to be added; files already allowed in `documents/`)
- Optional hybrid search (BM25/FTS5 + vector, RRF fusion)
- Optional reranker (cross-encoder) for top-k precision
