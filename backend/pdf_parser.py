"""Page-by-page PDF extraction with PyMuPDF.

Produces an ordered list of typed blocks per page (title, subtitle, heading,
text, table_row) so chunking can respect document structure. Tables are
detected with PyMuPDF's table finder and emitted row by row, each row carrying
its column headers, so a row like "Q2 2027" stays self-describing after
chunking. Page numbers are physical 1-based PDF page indices; the PDF page
label (if any) is stored separately.
"""
from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import pymupdf


@dataclass
class Block:
    kind: str  # title | subtitle | heading | text | table_row
    text: str
    y0: float


@dataclass
class ParsedPage:
    page_number: int  # physical, 1-based
    page_label: str
    blocks: list[Block]
    has_images: bool
    table_count: int

    @property
    def text(self) -> str:
        return "\n".join(b.text for b in self.blocks)


@dataclass
class ParsedDocument:
    document_id: str
    document_name: str
    sha256: str
    title: str
    subtitle: str
    pages: list[ParsedPage] = field(default_factory=list)

    @property
    def page_count(self) -> int:
        return len(self.pages)


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("­", "")  # soft hyphen
    text = re.sub(r"[ \t ]+", " ", text)
    return text.strip()


def document_id_from_name(name: str) -> str:
    stem = Path(name).stem.lower()
    return re.sub(r"[^a-z0-9]+", "_", stem).strip("_")


def _in_any_bbox(rect: tuple, bboxes: list[pymupdf.Rect]) -> bool:
    r = pymupdf.Rect(rect)
    return any(b.intersects(r) and (b & r).get_area() > 0.5 * max(r.get_area(), 1e-6) for b in bboxes)


def _line_text(line: dict) -> str:
    return normalize_text("".join(span["text"] for span in line["spans"]))


def parse_pdf(path: str | Path) -> ParsedDocument:
    path = Path(path)
    data = path.read_bytes()
    sha = hashlib.sha256(data).hexdigest()
    doc = pymupdf.open(stream=data, filetype="pdf")

    # Body font size = the size carrying the most characters (character-weighted,
    # so many short table cells do not outvote running paragraphs).
    char_counts: dict[float, int] = {}
    for page in doc:
        for blk in page.get_text("dict")["blocks"]:
            for line in blk.get("lines", []):
                for span in line["spans"]:
                    n = len(span["text"].strip())
                    if n:
                        key = round(span["size"], 1)
                        char_counts[key] = char_counts.get(key, 0) + n
    body_size = max(char_counts, key=char_counts.get) if char_counts else 10.0
    max_size = max(char_counts) if char_counts else body_size

    parsed = ParsedDocument(
        document_id=document_id_from_name(path.name),
        document_name=path.name,
        sha256=sha,
        title="",
        subtitle="",
    )

    for page_index, page in enumerate(doc):
        blocks: list[Block] = []
        tables = page.find_tables().tables
        table_bboxes = [pymupdf.Rect(t.bbox) for t in tables]

        for t in tables:
            rows = [[normalize_text(c or "") for c in row] for row in t.extract()]
            rows = [r for r in rows if any(r)]
            if not rows:
                continue
            header, body = rows[0], rows[1:]
            for row in body:
                pairs = [f"{h}: {v}" for h, v in zip(header, row) if v]
                blocks.append(Block("table_row", "; ".join(pairs), t.bbox[1] + 0.01 * len(blocks)))

        for blk in page.get_text("dict")["blocks"]:
            if blk.get("type") != 0:
                continue
            if _in_any_bbox(blk["bbox"], table_bboxes):
                continue
            lines = [ln for ln in blk.get("lines", []) if _line_text(ln)]
            if not lines:
                continue
            first = lines[0]["spans"][0]
            size = round(first["size"], 1)
            bold = "bold" in first["font"].lower() or bool(first["flags"] & 16)
            text = normalize_text(" ".join(_line_text(ln) for ln in lines))
            if size >= max_size and page_index == 0 and not parsed.title:
                kind = "title"
                parsed.title = text
            elif bold and size > body_size and page_index == 0 and not parsed.subtitle and not any(
                b.kind == "heading" for b in blocks
            ) and size < 1.5 * body_size:
                kind = "subtitle"
                parsed.subtitle = text
            elif bold and size >= 1.4 * body_size:
                kind = "heading"
            else:
                kind = "text"
            blocks.append(Block(kind, text, blk["bbox"][1]))

        blocks.sort(key=lambda b: b.y0)
        parsed.pages.append(
            ParsedPage(
                page_number=page_index + 1,
                page_label=page.get_label() or "",
                blocks=blocks,
                has_images=bool(page.get_images()),
                table_count=len(tables),
            )
        )
    return parsed


def parse_directory(pdf_dir: str | Path) -> list[ParsedDocument]:
    paths = sorted(Path(pdf_dir).glob("*.pdf"))
    return [parse_pdf(p) for p in paths]
