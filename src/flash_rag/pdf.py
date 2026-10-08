"""Page-by-page text extraction from PDFs (text-based PDFs only, no OCR).

Each page becomes its own unit of retrieval, so search results can tell you
exactly which page contains the answer. Image-only / scanned PDFs yield no
text and are skipped with a reason (OCR is intentionally out of scope).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from pypdf import PdfReader


@dataclass
class PdfExtraction:
    pages: dict[int, str] = field(default_factory=dict)  # 1-based page -> text
    skipped: list[str] = field(default_factory=list)


def extract_pdf_pages(path: Path) -> PdfExtraction:
    """Extract text per page. Returns empty ``pages`` + a reason if nothing usable."""
    result = PdfExtraction()
    try:
        reader = PdfReader(str(path))
    except Exception as exc:  # noqa: BLE001 - any parse failure is a skip
        result.skipped.append(f"unreadable PDF ({exc.__class__.__name__})")
        return result

    if reader.is_encrypted:
        try:
            reader.decrypt("")  # try the empty password
        except Exception:  # noqa: BLE001
            result.skipped.append("encrypted PDF (password required)")
            return result

    for num, page in enumerate(reader.pages, start=1):
        try:
            text = (page.extract_text() or "").strip()
        except Exception:  # noqa: BLE001 - a bad page should not kill the file
            continue
        if text:
            result.pages[num] = text

    if not result.pages:
        result.skipped.append("no extractable text (scanned/image-only PDF?)")
    return result
