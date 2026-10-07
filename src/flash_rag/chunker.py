"""Structure-aware chunking for markdown and plain text.

Strategy (current RAG best practice):

- Markdown: split on the heading hierarchy, keep a breadcrumb
  (``H1 > H2 > H3``) per section, then pack sections into chunks of at most
  ``CHUNK_MAX_TOKENS`` tokens with a small token overlap so context survives
  chunk boundaries.
- Plain text: split on paragraphs, pack the same way.
- Units longer than the chunk budget are hard-split at token boundaries.
- Code fences are respected: ``#`` lines inside ``` fences are not headings.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import tiktoken

from .config import CHUNK_MAX_TOKENS, CHUNK_OVERLAP_TOKENS

_ENCODER: tiktoken.Encoding | None = None
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_FENCE_RE = re.compile(r"^\s*(```|~~~)")


def _enc() -> tiktoken.Encoding:
    global _ENCODER
    if _ENCODER is None:
        _ENCODER = tiktoken.get_encoding("cl100k_base")
    return _ENCODER


def count_tokens(text: str) -> int:
    return len(_enc().encode(text))


@dataclass(frozen=True)
class Chunk:
    index: int
    text: str
    section: str  # heading breadcrumb, "" for plain text
    start: int  # best-effort char offset in the source file
    end: int
    tokens: int


# ---------------------------------------------------------------------------
# low-level helpers
# ---------------------------------------------------------------------------


def _split_long(text: str) -> list[str]:
    """Hard-split a unit longer than the chunk budget at token boundaries."""
    enc = _enc()
    toks = enc.encode(text)
    if len(toks) <= CHUNK_MAX_TOKENS:
        return [text]
    parts: list[str] = []
    i, n = 0, len(toks)
    while i < n:
        j = min(i + CHUNK_MAX_TOKENS, n)
        parts.append(enc.decode(toks[i:j]))
        if j >= n:
            break
        i = max(j - CHUNK_OVERLAP_TOKENS, i + 1)
    return parts


def _tail(text: str, n_tokens: int) -> str:
    """Last *n_tokens* tokens of *text* (used as overlap prefix)."""
    if n_tokens <= 0:
        return ""
    enc = _enc()
    toks = enc.encode(text)
    if len(toks) <= n_tokens:
        return text.strip()
    return enc.decode(toks[-n_tokens:]).lstrip()


def _pack(units: list[str]) -> list[tuple[str, int, int]]:
    """Greedily pack units into chunks <= CHUNK_MAX_TOKENS.

    Returns ``(chunk_text, first_unit_index, last_unit_index)`` tuples.
    """
    if not units:
        return []
    enc = _enc()
    chunks: list[tuple[str, int, int]] = []
    cur: list[str] = []
    first_i = 0
    cur_tokens = 0
    for i, u in enumerate(units):
        u_tokens = len(enc.encode(u))
        if cur and cur_tokens + u_tokens > CHUNK_MAX_TOKENS:
            chunks.append(("\n\n".join(cur), first_i, i - 1))
            tail = _tail("\n\n".join(cur), CHUNK_OVERLAP_TOKENS)
            cur = [tail] if tail else []
            first_i = i
            cur_tokens = len(enc.encode(tail)) if tail else 0
        else:
            if not cur:
                first_i = i
        cur.append(u)
        cur_tokens += u_tokens
    if cur:
        chunks.append(("\n\n".join(cur), first_i, len(units) - 1))
    return chunks


def _paragraph_units(text: str) -> list[tuple[str, int]]:
    """Split *text* into paragraph units with their char offsets."""
    units: list[tuple[str, int]] = []
    cursor = 0
    for para in text.split("\n\n"):
        stripped = para.strip()
        if not stripped:
            continue
        idx = text.find(para, cursor)
        if idx < 0:
            idx = cursor
        cursor = idx + len(para)
        offset = idx + (len(para) - len(para.lstrip()))
        for part in _split_long(stripped):
            units.append((part, offset))
    return units


def _to_chunks(units: list[tuple[str, int]], section: str) -> list[Chunk]:
    packed = _pack([u for u, _ in units])
    chunks: list[Chunk] = []
    for text, i0, i1 in packed:
        start = units[i0][1]
        end = units[i1][1] + len(units[i1][0])
        chunks.append(
            Chunk(
                index=len(chunks),
                text=text,
                section=section,
                start=start,
                end=end,
                tokens=count_tokens(text),
            )
        )
    return chunks


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------


def _split_sections(lines: list[str]) -> list[tuple[str, list[str], int]]:
    """Split markdown lines into ``(breadcrumb, lines, base_offset)`` sections.

    *base_offset* is the char offset of the section's first line in the
    original text (best effort), so chunk offsets can be made file-relative.
    """
    stack: dict[int, str] = {}
    sections: list[tuple[str, list[str], int]] = []
    buf: list[str] = []
    buf_start = 0
    in_fence = False
    offset = 0

    def close() -> None:
        nonlocal buf
        if buf:
            breadcrumb = " > ".join(stack[k] for k in sorted(stack))
            sections.append((breadcrumb, list(buf), buf_start))
            buf = []

    for line in lines:
        if not buf:
            buf_start = offset + (len(line) - len(line.lstrip()))
        if _FENCE_RE.match(line):
            in_fence = not in_fence
        m = None if in_fence else _HEADING_RE.match(line)
        if m:
            close()
            buf_start = offset + (len(line) - len(line.lstrip()))
            level = len(m.group(1))
            stack[level] = m.group(2).strip()
            for lv in [k for k in stack if k > level]:
                del stack[lv]
            buf.append(line)
        else:
            buf.append(line)
        offset += len(line) + 1
    close()
    return sections


def chunk_markdown(text: str) -> list[Chunk]:
    """Chunk a markdown document, preserving heading breadcrumbs."""
    sections = _split_sections(text.splitlines())
    chunks: list[Chunk] = []
    for breadcrumb, lines, base in sections:
        body = "\n".join(lines).strip()
        if not body:
            continue
        units = _paragraph_units(body)
        if not units:
            continue
        for c in _to_chunks(units, breadcrumb):
            chunks.append(
                Chunk(
                    index=len(chunks),
                    text=c.text,
                    section=c.section,
                    start=c.start + base,
                    end=c.end + base,
                    tokens=c.tokens,
                )
            )
    return chunks


def chunk_text(text: str, section: str = "") -> list[Chunk]:
    """Chunk plain text (paragraph-based)."""
    units = _paragraph_units(text)
    return _to_chunks(units, section)


def chunk_file(text: str, rel_path: str) -> list[Chunk]:
    """Dispatch on file extension."""
    suffix = Path(rel_path).suffix.lower()
    if suffix in {".md", ".markdown"}:
        return chunk_markdown(text)
    return chunk_text(text)
