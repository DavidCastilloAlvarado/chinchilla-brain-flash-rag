"""Runtime configuration: project root, paths, model, chunking parameters.

Everything can be overridden with environment variables:

- ``FLASH_RAG_MODEL``     embedding model (fastembed name), default nomic-ai/nomic-embed-text-v1.5
- ``FLASH_RAG_DOCS_DIR``  documents directory (default ``<root>/documents``)
- ``FLASH_RAG_DATA_DIR``  data directory (default ``<root>/.data``)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_NAME = "flash-rag"
DEFAULT_MODEL = "nomic-ai/nomic-embed-text-v1.5"
TABLE_NAME = "chunks"

# Chunking: 256-512 tokens with ~10-15% overlap is the 2025/26 RAG sweet spot.
CHUNK_MAX_TOKENS = 512
CHUNK_OVERLAP_TOKENS = 64

# Files above this size are skipped (with a warning) to keep indexing fast.
MAX_FILE_BYTES = 2 * 1024 * 1024  # 2 MB

# Only these suffixes are vectorized today. PDFs/images may sit in documents/
# but are ignored until a parser is added.
SUPPORTED_SUFFIXES = {".md", ".markdown", ".txt"}

# Embedding batch size (fastembed batches internally; this drives progress).
EMBED_BATCH_SIZE = 64


def find_project_root(start: Path | None = None) -> Path:
    """Walk up from *start* (default: CWD) to the flash-rag project root.

    A directory counts as the root when it holds a ``pyproject.toml`` that
    declares the flash-rag project, or a ``.flash-rag`` marker file.
    Falls back to *start* when nothing matches (still works from a subfolder
    of the repo, which is where agents usually run commands from).
    """
    cur = (start or Path.cwd()).resolve()
    for candidate in [cur, *cur.parents]:
        pyproject = candidate / "pyproject.toml"
        if pyproject.is_file():
            try:
                if PROJECT_NAME in pyproject.read_text(encoding="utf-8"):
                    return candidate
            except OSError:
                pass
        if (candidate / ".flash-rag").is_file():
            return candidate
    return cur


@dataclass(frozen=True)
class Config:
    root: Path
    docs_dir: Path
    data_dir: Path
    model: str
    providers: tuple[str, ...] = ()  # ONNX Runtime providers, e.g. ("CoreMLExecutionProvider",)

    @property
    def lancedb_dir(self) -> Path:
        return self.data_dir / "lancedb"

    @property
    def manifest_path(self) -> Path:
        return self.data_dir / "manifest.json"

    @property
    def meta_path(self) -> Path:
        return self.data_dir / "meta.json"


def load_config(start: Path | None = None) -> Config:
    root = find_project_root(start)

    raw_docs = os.environ.get("FLASH_RAG_DOCS_DIR", "").strip()
    docs_dir = Path(raw_docs) if raw_docs else root / "documents"
    if not docs_dir.is_absolute():
        docs_dir = root / docs_dir

    raw_data = os.environ.get("FLASH_RAG_DATA_DIR", "").strip()
    data_dir = Path(raw_data) if raw_data else root / ".data"
    if not data_dir.is_absolute():
        data_dir = root / data_dir

    model = os.environ.get("FLASH_RAG_MODEL", DEFAULT_MODEL)
    raw_providers = os.environ.get("FLASH_RAG_PROVIDERS", "").strip()
    providers = tuple(p.strip() for p in raw_providers.split(",") if p.strip())
    return Config(root=root, docs_dir=docs_dir, data_dir=data_dir, model=model, providers=providers)
