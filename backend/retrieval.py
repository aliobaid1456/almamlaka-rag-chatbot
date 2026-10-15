"""Hybrid retrieval: dense (multilingual embeddings) + lexical (BM25), fused
with Reciprocal Rank Fusion, then a cross-document coverage step.

* Dense search runs on the user's original wording *and* the English search
  queries produced by the query rewriter, so Arabic questions retrieve English
  passages even if one of the two signals is weak.
* BM25 catches exact names, amounts and dates ("$2,600,000", "Omar Fayez")
  that embeddings tend to blur. Numbers are indexed without separators, and
  Arabic text is normalized (alef forms, ta marbuta, diacritics).
* Cross-document coverage: if another document has a chunk ranked just
  outside the top-k, it is added. Conflicts live *between* documents, so the
  generator must see both sides to be able to flag them.

Scores are only a ranking signal. Whether the evidence actually answers the
question is decided later by the grounded generator and the validators.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from rank_bm25 import BM25Okapi

from .chunking import Chunk
from .embeddings import Embedder
from .text_utils import tokenize
from .vector_store import VectorStore

RRF_K = 60


@dataclass
class RetrievedChunk:
    chunk: Chunk
    fused_score: float
    dense_score: float | None
    bm25_score: float
    rank: int


class HybridRetriever:
    def __init__(self, store: VectorStore, embedder: Embedder | None, top_k: int = 6, coverage_window: int = 4):
        self.store = store
        self.embedder = embedder if store.has_vectors else None
        self.top_k = top_k
        self.coverage_window = coverage_window
        self._bm25 = BM25Okapi([tokenize(c.index_text) for c in store.chunks])

    @property
    def mode(self) -> str:
        return "hybrid (dense + BM25)" if self.embedder else "lexical (BM25 only)"

    def max_lexical_score(self, queries: list[str]) -> float:
        """Best BM25 score of any chunk for any query: a cheap 'does this mention
        something the documents talk about?' signal."""
        best = 0.0
        for q in queries:
            if q and q.strip():
                best = max(best, float(np.max(self._bm25.get_scores(tokenize(q)))))
        return best

    def retrieve(self, queries: list[str], top_k: int | None = None) -> list[RetrievedChunk]:
        k = top_k or self.top_k
        queries = [q for q in dict.fromkeys(q.strip() for q in queries) if q]
        n = len(self.store.chunks)
        if not queries or n == 0:
            return []

        fused = np.zeros(n)
        best_dense = np.full(n, -1.0)
        best_bm25 = np.zeros(n)

        for q in queries:
            if self.embedder is not None:
                qv = self.embedder.embed_query(q)
                for rank, (idx, score) in enumerate(self.store.search(qv, n)):
                    fused[idx] += 1.0 / (RRF_K + rank + 1)
                    best_dense[idx] = max(best_dense[idx], score)
            scores = self._bm25.get_scores(tokenize(q))
            order = [i for i in np.argsort(-scores, kind="stable") if scores[i] > 0]
            for rank, idx in enumerate(order):
                fused[idx] += 1.0 / (RRF_K + rank + 1)
                best_bm25[idx] = max(best_bm25[idx], scores[idx])

        ranked = [int(i) for i in np.argsort(-fused, kind="stable") if fused[i] > 0]
        selected = ranked[:k]

        # Cross-document coverage (needed for conflict detection).
        covered = {self.store.chunks[i].document_id for i in selected}
        for i in ranked[k : k + self.coverage_window]:
            doc = self.store.chunks[i].document_id
            if doc not in covered:
                selected.append(i)
                covered.add(doc)

        return [
            RetrievedChunk(
                chunk=self.store.chunks[i],
                fused_score=float(fused[i]),
                dense_score=float(best_dense[i]) if self.embedder else None,
                bm25_score=float(best_bm25[i]),
                rank=r + 1,
            )
            for r, i in enumerate(selected)
        ]
