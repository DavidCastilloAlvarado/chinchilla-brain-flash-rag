"""Local embedding via fastembed (ONNX Runtime — no torch, no GPU required).

fastembed is the most efficient local option: quantized ONNX weights,
CPU-only inference, and it downloads models straight from Hugging Face on
first use (cached under the configured data directory).

The model name is a fastembed model id, e.g.::

    nomic-ai/nomic-embed-text-v1.5   (default — 137M, 768 dims, Apache-2.0)
    Qwen/Qwen3-Embedding-0.6B        (top MTEB, 32k context, heavier)
    BAAI/bge-small-en-v1.5           (33M, 384 dims — fastest)
    sentence-transformers/all-MiniLM-L6-v2  (22M, 384 dims — classic)
"""

from __future__ import annotations

from pathlib import Path

from fastembed import TextEmbedding


class Embedder:
    """Thin wrapper around fastembed's TextEmbedding."""

    def __init__(
        self, model_name: str, providers: tuple[str, ...] = (), *, cache_dir: Path
    ):
        self.model_name = model_name
        if providers:
            self._model = TextEmbedding(
                model_name, cache_dir=str(cache_dir), providers=list(providers)
            )
        else:
            self._model = TextEmbedding(model_name, cache_dir=str(cache_dir))

    @property
    def dim(self) -> int:
        try:
            return int(self._model.dim)
        except AttributeError:  # pragma: no cover - older fastembed
            probe = self._model.embed(["dim probe"])
            return len(next(iter(probe)))

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return [vec.tolist() for vec in self._model.embed(list(texts))]
