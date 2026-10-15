"""PDF extraction, chunk metadata and index lifecycle (categories 15, 16)."""
import json

import pymupdf
import pytest

from backend.chunking import chunk_documents, count_tokens
from backend.config import Settings
from backend.ingestion import IngestionError, build_index, build_or_load_index
from backend.pdf_parser import parse_directory
from conftest import PDF_DIR


def test_three_documents_two_pages_each():
    docs = parse_directory(PDF_DIR)
    assert sorted(d.document_name for d in docs) == [
        "Almamlaka_Budget_Timeline.pdf",
        "Almamlaka_Project_Overview.pdf",
        "Almamlaka_Team_Governance.pdf",
    ]
    assert all(d.page_count == 2 for d in docs)
    assert all(p.blocks for d in docs for p in d.pages), "every page must yield text (no OCR needed)"


def test_tables_are_extracted_row_by_row_with_headers():
    budget = next(d for d in parse_directory(PDF_DIR) if "Budget" in d.document_name)
    rows = [b.text for b in budget.pages[0].blocks if b.kind == "table_row"]
    assert "Quarter: Q2 2027; Development: $500,000; Marketing: $150,000; Operations: $70,000; Total: $720,000" in rows
    timeline = [b.text for b in budget.pages[1].blocks if b.kind == "table_row"]
    assert "Milestone: Public launch; Target Date: March 15, 2027" in timeline


def test_chunks_have_required_metadata_and_respect_size(store):
    assert len(store.chunks) == 16
    for c in store.chunks:
        assert c.document_id and c.document_name.endswith(".pdf")
        assert c.page_number in (1, 2)
        assert c.section_title and c.chunk_id and len(c.content_hash) == 64
        assert count_tokens(c.text) <= 500
    assert len({c.chunk_id for c in store.chunks}) == 16


def test_every_chunk_text_is_on_its_cited_page():
    """Page metadata is verified against the PDF itself, not assumed."""
    chunks = chunk_documents(parse_directory(PDF_DIR))
    for c in chunks:
        page_text = " ".join(pymupdf.open(PDF_DIR / c.document_name)[c.page_number - 1].get_text().split())
        for line in c.text.splitlines():
            if ";" in line:  # table rows are reformatted; check each cell value instead
                for cell in line.split("; "):
                    assert cell.split(": ", 1)[1] in page_text, (c.chunk_id, cell)
            else:
                assert line in page_text, (c.chunk_id, line[:60])


def test_long_sections_are_split_with_overlap():
    docs = parse_directory(PDF_DIR)
    small = chunk_documents(docs, max_tokens=40, overlap_tokens=10)
    assert len(small) > 16
    assert any(c.section_title.endswith("(cont.)") for c in small)


def test_ingestion_is_deterministic_and_reuses_index(tmp_path, tmp_pdf_dir):
    s = Settings(embedding_backend="none", pdf_dir=tmp_pdf_dir, index_dir=tmp_path / "idx")
    _, m1 = build_index(s, None)
    first = (s.index_dir / "chunks.json").read_text()
    store2, m2 = build_or_load_index(s, None)
    assert m2["build_seconds"] == m1["build_seconds"]  # loaded, not rebuilt
    build_index(s, None)
    assert (s.index_dir / "chunks.json").read_text() == first  # identical rebuild


def test_index_rebuilds_when_a_pdf_changes(tmp_path, tmp_pdf_dir):
    s = Settings(embedding_backend="none", pdf_dir=tmp_pdf_dir, index_dir=tmp_path / "idx")
    build_index(s, None)
    (tmp_pdf_dir / "Almamlaka_Team_Governance.pdf").unlink()
    store, manifest = build_or_load_index(s, None)
    assert len(manifest["pdfs"]) == 2
    assert all("governance" not in c.document_id for c in store.chunks)


def test_missing_documents_raise(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(IngestionError):
        build_or_load_index(Settings(embedding_backend="none", pdf_dir=empty, index_dir=tmp_path / "i"), None)
    with pytest.raises(IngestionError):
        build_or_load_index(Settings(embedding_backend="none", pdf_dir=tmp_path / "nope", index_dir=tmp_path / "i"), None)


def test_corrupt_index_is_rebuilt(tmp_path, tmp_pdf_dir):
    s = Settings(embedding_backend="none", pdf_dir=tmp_pdf_dir, index_dir=tmp_path / "idx")
    build_index(s, None)
    (s.index_dir / "chunks.json").write_text("{not json")
    store, _ = build_or_load_index(s, None)
    assert len(store.chunks) == 16
    json.loads((s.index_dir / "chunks.json").read_text())
