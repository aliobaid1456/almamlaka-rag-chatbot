"""Structure-aware chunking.

Strategy: one chunk per numbered section, never crossing a page boundary (so
every chunk maps to exactly one verifiable PDF page). A section longer than
``max_tokens`` is split on sentence/row boundaries with ``overlap_tokens`` of
overlap. Every chunk carries its document title and section heading as a
context header for embedding and lexical search, while ``text`` keeps only the
verbatim page content used for excerpts and grounding checks.
"""
from __future__ import annotations

import hashlib
import re

from pydantic import BaseModel

from .pdf_parser import ParsedDocument


class Chunk(BaseModel):
    chunk_id: str
    document_id: str
    document_name: str
    document_title: str
    page_number: int
    section_title: str
    text: str
    content_hash: str

    @property
    def index_text(self) -> str:
        """Text used for embedding / BM25: verbatim content plus a context header."""
        return f"{self.document_title} | {self.section_title}\n{self.text}"


_TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def count_tokens(text: str) -> int:
    """Approximate token count (word pieces + punctuation). Good enough for sizing."""
    return len(_TOKEN_RE.findall(text))


def _split_units(lines: list[str]) -> list[str]:
    units: list[str] = []
    for line in lines:
        units.extend(s.strip() for s in re.split(r"(?<=[.!?؟])\s+", line) if s.strip())
    return units


def _window(units: list[str], max_tokens: int, overlap_tokens: int) -> list[list[str]]:
    windows: list[list[str]] = []
    current: list[str] = []
    for unit in units:
        if current and count_tokens(" ".join(current + [unit])) > max_tokens:
            windows.append(current)
            carry: list[str] = []
            for prev in reversed(current):
                if count_tokens(" ".join([prev] + carry)) > overlap_tokens:
                    break
                carry.insert(0, prev)
            current = carry
        current.append(unit)
    if current:
        windows.append(current)
    return windows


def chunk_document(doc: ParsedDocument, max_tokens: int = 500, overlap_tokens: int = 80) -> list[Chunk]:
    full_title = " — ".join(t for t in (doc.title, doc.subtitle) if t)
    chunks: list[Chunk] = []
    section = "Preamble"
    for page in doc.pages:
        buffer: list[str] = []

        def flush() -> None:
            if not buffer:
                return
            if count_tokens(" ".join(buffer)) <= max_tokens:
                windows = [list(buffer)]  # keep original lines intact
            else:
                windows = _window(_split_units(buffer), max_tokens, overlap_tokens)
            for part, window in enumerate(windows):
                text = "\n".join(window)
                n = len(chunks)
                chunks.append(
                    Chunk(
                        chunk_id=f"{doc.document_id}-p{page.page_number}-c{n:02d}",
                        document_id=doc.document_id,
                        document_name=doc.document_name,
                        document_title=full_title,
                        page_number=page.page_number,
                        section_title=section if part == 0 else f"{section} (cont.)",
                        text=text,
                        content_hash=hashlib.sha256(text.encode("utf-8")).hexdigest(),
                    )
                )
            buffer.clear()

        for block in page.blocks:
            if block.kind in ("title", "subtitle"):
                continue
            if block.kind == "heading":
                flush()
                section = block.text
                continue
            buffer.append(block.text)
        flush()  # sections never cross a page boundary
    return chunks


def chunk_documents(docs: list[ParsedDocument], max_tokens: int = 500, overlap_tokens: int = 80) -> list[Chunk]:
    out: list[Chunk] = []
    for d in docs:
        out.extend(chunk_document(d, max_tokens, overlap_tokens))
    return out
