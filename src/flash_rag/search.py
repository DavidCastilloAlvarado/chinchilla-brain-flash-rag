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
) -> dict:
    """Embed *query* and return the top-k chunks as a JSON-ready dict."""
    store = Store(cfg.data_dir, cfg.model)
    if not store.exists():
        raise NotInitialized()
    embedder = make_embedder(cfg)
    vector = embedder.embed([query])[0]
    hits = store.search(vector, top_k, path_filter)

    results = []
    for i, row in enumerate(hits, 1):
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
        "count": len(results),
        "results": results,
    }
