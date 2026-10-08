"""Runtime configuration: project root, paths, model, chunking parameters.

Everything can be overridden with environment variables, either exported in
the shell or stored in a ``.env`` file at the project root (see
``.env.example``). Real environment variables always take precedence over
the ``.env`` file.

- ``FLASH_RAG_MODEL``     embedding model (fastembed name), default nomic-ai/nomic-embed-text-v1.5
- ``FLASH_RAG_DOCS_DIR``  documents directory (default ``<root>/documents``)
- ``FLASH_RAG_DATA_DIR``  data directory (default ``<root>/.data``)
- ``FLASH_RAG_PROVIDERS`` comma-separated ONNX Runtime providers (GPU on Apple Silicon)
- ``FLASH_RAG_AUTO_REFRESH_MAX_AGE_DAYS``       max index age before ``search`` auto-refreshes (default 7)
- ``FLASH_RAG_AUTO_REFRESH_CHECK_INTERVAL_HOURS`` min hours between pre-search freshness checks (default 24)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_NAME = "flash-rag"
DEFAULT_MODEL = "nomic-ai/nomic-embed-text-v1.5"
TABLE_NAME = "chunks"

# Embedding backends: "onnx" (default, cross-platform via fastembed) or
# "mlx" (Apple Silicon — 100% GPU/ANE, no CoreML graph splitting).
BACKENDS = ("onnx", "mlx")
DEFAULT_BACKEND = "onnx"

# Chunking: 256-512 tokens with ~10-15% overlap is the 2025/26 RAG sweet spot.
CHUNK_MAX_TOKENS = 512
CHUNK_OVERLAP_TOKENS = 64

# Files above this size are skipped (with a warning) to keep indexing fast.
MAX_FILE_BYTES = 2 * 1024 * 1024  # 2 MB

# Suffixes that are vectorized. PDFs must contain extractable text (no OCR);
# image-only PDFs are skipped with a warning.
SUPPORTED_SUFFIXES = {".md", ".markdown", ".txt", ".pdf"}

# Embedding batch size (fastembed batches internally; this drives progress).
EMBED_BATCH_SIZE = 64

# Auto-refresh before search: when the index's ``updated`` timestamp is older
# than this, ``search`` runs an incremental refresh first. The freshness check
# itself runs at most once per CHECK_INTERVAL_HOURS (state in
# ``<data_dir>/freshness_state.json``).
DEFAULT_AUTO_REFRESH_MAX_AGE_DAYS = 7.0
DEFAULT_AUTO_REFRESH_CHECK_INTERVAL_HOURS = 24.0


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
    workspace_dirs: tuple[Path, ...] = ()  # extra dirs to index (FLASH_RAG_WORKSPACE_DIRS)
    backend: str = DEFAULT_BACKEND  # "onnx" or "mlx"
    auto_refresh_max_age_days: float = DEFAULT_AUTO_REFRESH_MAX_AGE_DAYS  # search auto-refreshes when the index is older than this
    auto_refresh_check_interval_hours: float = (
        DEFAULT_AUTO_REFRESH_CHECK_INTERVAL_HOURS  # pre-search freshness check runs at most once per interval
    )

    @property
    def lancedb_dir(self) -> Path:
        return self.data_dir / "lancedb"

    @property
    def model_cache_dir(self) -> Path:
        return self.data_dir / "models"

    @property
    def manifest_path(self) -> Path:
        return self.data_dir / "manifest.json"

    @property
    def meta_path(self) -> Path:
        return self.data_dir / "meta.json"

    @property
    def freshness_state_path(self) -> Path:
        return self.data_dir / "freshness_state.json"


def _load_env_file(root: Path) -> None:
    """Load ``<root>/.env`` if present. Real env vars always win (override=False)."""
    env_file = root / ".env"
    if env_file.is_file():
        load_dotenv(env_file, override=False)


def _env_float(name: str, default: float) -> float:
    """Read a float env var, falling back to *default* when unset/empty."""
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        raise ValueError(f"Invalid {name}={raw!r} — expected a number (e.g. 7 or 24)") from None


def load_config(start: Path | None = None) -> Config:
    root = find_project_root(start)
    _load_env_file(root)

    raw_docs = os.environ.get("FLASH_RAG_DOCS_DIR", "").strip()
    docs_dir = Path(raw_docs) if raw_docs else root / "documents"
    if not docs_dir.is_absolute():
        docs_dir = root / docs_dir

    raw_data = os.environ.get("FLASH_RAG_DATA_DIR", "").strip()
    data_dir = Path(raw_data) if raw_data else root / ".data"
    if not data_dir.is_absolute():
        data_dir = root / data_dir

    model = os.environ.get("FLASH_RAG_MODEL", DEFAULT_MODEL)
    backend = os.environ.get("FLASH_RAG_BACKEND", DEFAULT_BACKEND).strip().lower()
    if backend not in BACKENDS:
        raise ValueError(
            f"Invalid FLASH_RAG_BACKEND {backend!r} — expected one of: {', '.join(BACKENDS)}"
        )
    raw_providers = os.environ.get("FLASH_RAG_PROVIDERS", "").strip()
    providers = tuple(p.strip() for p in raw_providers.split(",") if p.strip())

    raw_ws = os.environ.get("FLASH_RAG_WORKSPACE_DIRS", "").strip()
    workspace_dirs: tuple[Path, ...] = ()
    if raw_ws:
        workspace_dirs = tuple(
            Path(p) if Path(p).is_absolute() else root / Path(p)
            for p in (part.strip() for part in raw_ws.split(","))
            if p.strip()
        )

    auto_refresh_max_age_days = _env_float(
        "FLASH_RAG_AUTO_REFRESH_MAX_AGE_DAYS", DEFAULT_AUTO_REFRESH_MAX_AGE_DAYS
    )
    auto_refresh_check_interval_hours = _env_float(
        "FLASH_RAG_AUTO_REFRESH_CHECK_INTERVAL_HOURS",
        DEFAULT_AUTO_REFRESH_CHECK_INTERVAL_HOURS,
    )

    return Config(
        root=root,
        docs_dir=docs_dir,
        data_dir=data_dir,
        model=model,
        providers=providers,
        workspace_dirs=workspace_dirs,
        backend=backend,
        auto_refresh_max_age_days=auto_refresh_max_age_days,
        auto_refresh_check_interval_hours=auto_refresh_check_interval_hours,
    )
