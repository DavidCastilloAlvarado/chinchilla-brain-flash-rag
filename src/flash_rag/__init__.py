"""flash-rag: local semantic search over documents/.

A repo-local knowledge base: the original documents live in ``documents/``
(committed), while chunks, the vector index and embeddings live in ``.data/``
(never committed) and are rebuilt with ``uv run db-init`` / ``uv run db-refresh``.
"""

__version__ = "0.1.0"
