"""MLX embedding backend (Apple Silicon) — runs 100% on GPU/ANE.

Unlike the ONNX/CoreML path (where the CoreML execution provider only
converts part of the graph and the rest falls back to the CPU, splitting the
inference into many partitions), MLX executes the whole model natively on the
GPU with no conversion step.

Uses the ``mlx-embeddings`` package (Blaizzy/mlx-embeddings), which loads
Hugging Face weights directly (no ONNX export) and applies the model's own
pooling config (mean pooling + L2 normalization for BERT-family models).

Install (Apple Silicon only):  uv sync --extra mlx
Enable:                        FLASH_RAG_BACKEND=mlx
"""

from __future__ import annotations

import os
from pathlib import Path

# Chunks are ~512 tokens + a short path/section prefix; 1024 gives headroom
# while staying far below nomic's 8192 context.
MLX_MAX_LENGTH = 1024


class MlxEmbedder:
    """Same interface as ``Embedder`` (ONNX), but runs on Apple's MLX."""

    def __init__(self, model_name: str, *, cache_dir: Path):
        try:
            import mlx.core as mx
            from mlx_embeddings.utils import get_model_path, load_config, load
        except ImportError as exc:
            raise RuntimeError(
                "The MLX backend requires Apple Silicon and the 'mlx' extra. "
                "Install with: uv sync --extra mlx   (and set FLASH_RAG_BACKEND=mlx)"
            ) from exc

        # Route HF downloads under our data dir when the user hasn't set a
        # cache location themselves.
        os.environ.setdefault("HF_HUB_CACHE", str(cache_dir / "hf-hub"))

        # mlx-embeddings has no nomic_bert module; register our vendored port
        # so the loader can resolve model_type "nomic_bert".
        model_path = get_model_path(model_name)
        model_type = str(load_config(model_path).get("model_type", "")).replace("-", "_")
        if model_type == "nomic_bert":
            from .nomic_bert import register_nomic_bert

            register_nomic_bert()

        self.model_name = model_name
        self._mx = mx
        self._model, self._tokenizer = load(model_name)
        self._dim: int | None = None

    @property
    def dim(self) -> int:
        if self._dim is None:
            self._dim = len(self.embed(["dim probe"])[0])
        return self._dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        enc = self._tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=MLX_MAX_LENGTH,
            return_tensors="mlx",
        )
        out = self._model(enc["input_ids"], attention_mask=enc["attention_mask"])
        self._mx.eval(out.text_embeds)
        return [vec.tolist() for vec in out.text_embeds]
