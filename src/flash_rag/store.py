"""LanceDB-backed chunk store.

LanceDB is an embedded, zero-server vector database (Rust core, columnar
Lance format, HNSW/IVF-PQ indexes, metadata filtering, versioned writes).
It is the standard "local Pinecone" alternative: the index is a plain
directory on disk (``.data/lancedb``), no daemon, no network.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import lancedb
import pyarrow as pa
from lancedb.index import FTS

from .config import TABLE_NAME

def make_schema(dim: int) -> pa.Schema:
    """LanceDB requires a fixed-size list for the vector column."""
    return pa.schema(
        [
            pa.field("id", pa.string()),
            pa.field("file_path", pa.string()),
            pa.field("section", pa.string()),
            pa.field("page", pa.int32()),  # 1-based PDF page, -1 for non-PDF files
            pa.field("text", pa.string()),
            pa.field("start", pa.int64()),
            pa.field("end", pa.int64()),
            pa.field("tokens", pa.int32()),
            pa.field("file_hash", pa.string()),
            pa.field("mtime", pa.float64()),
            pa.field("vector", pa.list_(pa.float32(), dim)),
        ]
    )


def _escape(value: str) -> str:
    return value.replace("'", "''")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Store:
    def __init__(self, data_dir: Path, model: str):
        self.data_dir = data_dir
        self.model = model
        data_dir.mkdir(parents=True, exist_ok=True)
        self._db = lancedb.connect(str(data_dir / "lancedb"))

    # -- lifecycle ----------------------------------------------------------

    def exists(self) -> bool:
        return TABLE_NAME in self._db.list_tables().tables

    def drop(self) -> None:
        if self.exists():
            self._db.drop_table(TABLE_NAME)

    def _table(self):
        return self._db.open_table(TABLE_NAME)

    def table(self):
        """Public accessor for the LanceDB table (used by reports)."""
        return self._table()

    def create(self, rows: list[dict], dim: int) -> None:
        if self.exists():
            self.drop()
        self._db.create_table(TABLE_NAME, data=rows, schema=make_schema(dim))

    def add(self, rows: list[dict], dim: int | None = None) -> None:
        if not rows:
            return
        if self.exists():
            self._table().add(rows)
        else:
            if dim is None:
                dim = len(rows[0]["vector"])
            self._db.create_table(TABLE_NAME, data=rows, schema=make_schema(dim))

    def delete_file(self, rel_path: str) -> None:
        if self.exists():
            self._table().delete(f"file_path = '{_escape(rel_path)}'")

    def ensure_fts_index(self) -> None:
        """Create the BM25 full-text index on ``text`` if missing (idempotent).

        Uses the classic English stop-word list + stemming (FTS defaults).
        The index is maintained automatically by LanceDB on subsequent adds.
        """
        if not self.exists():
            return
        table = self._table()
        for idx in table.list_indices():
            if idx.index_type == "FTS" and "text" in idx.columns:
                return
        table.create_index("text", config=FTS(remove_stop_words=True))

    def count(self) -> int:
        if not self.exists():
            return 0
        return self._table().count_rows()

    # -- search -------------------------------------------------------------

    def search(
        self,
        vector: list[float],
        k: int,
        path_filter: str | None = None,
        text_query: str | None = None,
    ) -> list[dict]:
        """Hybrid search (BM25 + vector, fused) when *text_query* is given,
        pure cosine vector search otherwise."""
        table = self._table()
        if text_query:
            q = (
                table.search(None, query_type="hybrid")
                .vector(vector)
                .text(text_query)
                .metric("cosine")
            )
        else:
            q = table.search(vector).metric("cosine")
        if path_filter:
            q = q.where(f"file_path LIKE '{_escape(path_filter)}%'")
        return q.limit(k).to_list()

    # -- metadata -----------------------------------------------------------

    def write_meta(self, dim: int, created: bool) -> None:
        meta = {
            "model": self.model,
            "dim": dim,
            "created": _now(),
            "updated": _now(),
            "version": 1,
        }
        if not created and self.meta_path.is_file():
            try:
                old = json.loads(self.meta_path.read_text(encoding="utf-8"))
                meta["created"] = old.get("created", meta["created"])
            except (OSError, json.JSONDecodeError):
                pass
        self.meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    def read_meta(self) -> dict | None:
        if not self.meta_path.is_file():
            return None
        try:
            return json.loads(self.meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    @property
    def meta_path(self) -> Path:
        return self.data_dir / "meta.json"
