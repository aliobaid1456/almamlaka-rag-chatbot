"""Ingestion: PDFs -> structured pages -> chunks -> embeddings -> persisted index.

Deterministic and idempotent: the manifest fingerprints the PDFs, chunking
parameters and embedding model; ``build_or_load_index`` reuses the saved index
when the fingerprint matches and rebuilds it otherwise.
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path

from .chunking import chunk_documents
from .config import Settings
from .embeddings import Embedder
from .pdf_parser import parse_directory
from .vector_store import VectorStore

log = logging.getLogger(__name__)

INGESTION_VERSION = "1"


class IngestionError(RuntimeError):
    pass


def _pdf_fingerprints(pdf_dir: Path) -> dict[str, str]:
    paths = sorted(pdf_dir.glob("*.pdf"))
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def expected_manifest(settings: Settings, embedder: Embedder | None) -> dict:
    if not settings.pdf_dir.is_dir():
        raise IngestionError(f"PDF directory not found: {settings.pdf_dir.name}")
    pdfs = _pdf_fingerprints(settings.pdf_dir)
    if not pdfs:
        raise IngestionError("No PDF documents found; the knowledge base is empty.")
    if len(pdfs) != 3:
        log.warning("Expected 3 PDFs, found %d", len(pdfs))
    return {
        "ingestion_version": INGESTION_VERSION,
        "pdfs": pdfs,
        "chunk_max_tokens": settings.chunk_max_tokens,
        "chunk_overlap_tokens": settings.chunk_overlap_tokens,
        "embedding_model": embedder.name if embedder else None,
    }


def build_index(settings: Settings, embedder: Embedder | None) -> tuple[VectorStore, dict]:
    manifest = expected_manifest(settings, embedder)
    started = time.time()
    docs = parse_directory(settings.pdf_dir)
    chunks = chunk_documents(docs, settings.chunk_max_tokens, settings.chunk_overlap_tokens)
    empty_pages = [f"{d.document_name} p{p.page_number}" for d in docs for p in d.pages if not p.blocks]
    if empty_pages:
        log.warning("Pages with no extractable text (would need OCR): %s", empty_pages)
    vectors = embedder.embed_documents([c.index_text for c in chunks]) if embedder else None
    store = VectorStore(chunks, vectors)
    manifest = {
        **manifest,
        "chunk_count": len(chunks),
        "documents": [
            {"name": d.document_name, "title": d.subtitle or d.title, "pages": d.page_count, "sha256": d.sha256}
            for d in docs
        ],
        "empty_pages": empty_pages,
        "build_seconds": round(time.time() - started, 2),
    }
    store.save(settings.index_dir, manifest)
    return store, manifest


def _fingerprint(m: dict) -> str:
    keys = ("ingestion_version", "pdfs", "chunk_max_tokens", "chunk_overlap_tokens", "embedding_model")
    return json.dumps({k: m.get(k) for k in keys}, sort_keys=True)


def build_or_load_index(settings: Settings, embedder: Embedder | None) -> tuple[VectorStore, dict]:
    expected = expected_manifest(settings, embedder)
    if (settings.index_dir / "manifest.json").exists():
        try:
            store, manifest = VectorStore.load(settings.index_dir)
            if _fingerprint(manifest) == _fingerprint(expected) and (store.has_vectors or embedder is None):
                return store, manifest
            log.info("Index is stale; rebuilding.")
        except Exception as exc:
            log.warning("Saved index unreadable (%s); rebuilding.", type(exc).__name__)
    return build_index(settings, embedder)
