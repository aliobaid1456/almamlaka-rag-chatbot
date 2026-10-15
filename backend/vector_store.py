"""Persistent FAISS vector index plus chunk metadata.

The corpus is 16 chunks, so an exact inner-product index (IndexFlatIP over
L2-normalized vectors = cosine similarity) is the right tool: no approximation
error and no service to run. The index directory holds:

  vectors.faiss   the FAISS index
  chunks.json     chunk text + metadata, row-aligned with the index
  manifest.json   PDF hashes, chunking config and embedding model name

``manifest.json`` makes ingestion deterministic and safe to rerun: the index is
rebuilt only when a PDF, the chunking settings or the embedding model change.
"""
from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np

from .chunking import Chunk


class VectorStore:
    def __init__(self, chunks: list[Chunk], vectors: np.ndarray | None):
        self.chunks = chunks
        self.index: faiss.Index | None = None
        if vectors is not None:
            vectors = np.ascontiguousarray(vectors, dtype="float32")
            if len(vectors) != len(chunks):
                raise ValueError("vectors and chunks are not aligned")
            self.index = faiss.IndexFlatIP(vectors.shape[1])
            self.index.add(vectors)

    @property
    def has_vectors(self) -> bool:
        return self.index is not None and self.index.ntotal == len(self.chunks)

    def search(self, query_vector: np.ndarray, k: int) -> list[tuple[int, float]]:
        if not self.has_vectors:
            return []
        q = np.ascontiguousarray(query_vector.reshape(1, -1), dtype="float32")
        scores, ids = self.index.search(q, min(k, len(self.chunks)))
        return [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if i >= 0]

    # ---- persistence -------------------------------------------------------
    def save(self, index_dir: Path, manifest: dict) -> None:
        index_dir.mkdir(parents=True, exist_ok=True)
        (index_dir / "chunks.json").write_text(
            json.dumps([c.model_dump() for c in self.chunks], ensure_ascii=False, indent=1), encoding="utf-8"
        )
        if self.index is not None:
            faiss.write_index(self.index, str(index_dir / "vectors.faiss"))
        elif (index_dir / "vectors.faiss").exists():
            (index_dir / "vectors.faiss").unlink()
        (index_dir / "manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True), encoding="utf-8")

    @classmethod
    def load(cls, index_dir: Path) -> tuple["VectorStore", dict]:
        manifest = json.loads((index_dir / "manifest.json").read_text(encoding="utf-8"))
        chunks = [Chunk(**c) for c in json.loads((index_dir / "chunks.json").read_text(encoding="utf-8"))]
        store = cls(chunks, None)
        vec_path = index_dir / "vectors.faiss"
        if vec_path.exists():
            store.index = faiss.read_index(str(vec_path))
            if store.index.ntotal != len(chunks):
                raise ValueError("Index is corrupt: vector count does not match chunk count")
        return store, manifest
