"""Query the local vector index."""

from __future__ import annotations

from .config import Config
from .embedder import make_embedder
from .index import NotInitialized
from .store import Store


def run_search(
    cfg: Config,
    query: str,
    top_k: int = 5,
    path_filter: str | None = None,
    vector_only: bool = False,
) -> dict:
    """Embed *query* and return the top-k chunks as a JSON-ready dict.

    Hybrid search (BM25 full-text + vector, fused score) by default;
    *vector_only* disables the full-text leg.
    """
    store = Store(cfg.data_dir, cfg.model)
    if not store.exists():
        raise NotInitialized()
    store.ensure_fts_index()  # self-heal indexes built before FTS existed
    embedder = make_embedder(cfg)
    vector = embedder.embed([query])[0]
    hits = store.search(
        vector, top_k, path_filter, text_query=None if vector_only else query
    )

    results = []
    for i, row in enumerate(hits, 1):
        if "_relevance_score" in row:  # hybrid: fused 0-1 relevance
            score = float(row["_relevance_score"])
        else:  # vector-only: cosine distance
            distance = row.get("_distance")
            score = 1.0 - float(distance) if distance is not None else 0.0
        page = row.get("page", -1)
        results.append(
            {
                "rank": i,
                "score": round(max(0.0, min(1.0, score)), 4),
                "file": row["file_path"],
                "section": row.get("section", ""),
                "page": page if page is not None and page >= 0 else None,
                "start": row.get("start"),
                "end": row.get("end"),
                "text": row["text"],
            }
        )
    return {
        "query": query,
        "model": cfg.model,
        "mode": "vector" if vector_only else "hybrid",
        "count": len(results),
        "results": results,
    }
