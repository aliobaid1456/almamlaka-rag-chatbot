"""Embedding backends.

Default: ``sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`` run
through FastEmbed (ONNX Runtime, quantized, ~0.22 GB, Apache-2.0). It was
trained by multilingual knowledge distillation on parallel sentences, so an
Arabic question and its English source passage land close together — the
property this corpus needs, because all three PDFs are English while users
may ask in Arabic. No PyTorch dependency, so it fits small free hosting tiers.
"""
from __future__ import annotations

import logging
from typing import Protocol

import numpy as np

log = logging.getLogger(__name__)


class Embedder(Protocol):
    name: str

    def embed_documents(self, texts: list[str]) -> np.ndarray: ...

    def embed_query(self, text: str) -> np.ndarray: ...


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype="float32")
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.clip(norms, 1e-12, None)


class FastEmbedEmbedder:
    def __init__(self, model_name: str):
        from fastembed import TextEmbedding  # imported lazily: optional at test time

        self.name = model_name
        self._model = TextEmbedding(model_name=model_name)
        # E5 models expect role prefixes; paraphrase models do not.
        self._e5 = "e5" in model_name.lower()

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        texts = [f"passage: {t}" for t in texts] if self._e5 else texts
        return _l2_normalize(np.array(list(self._model.embed(texts))))

    def embed_query(self, text: str) -> np.ndarray:
        text = f"query: {text}" if self._e5 else text
        return _l2_normalize(np.array(list(self._model.embed([text]))))[0]


def load_embedder(backend: str, model_name: str) -> Embedder | None:
    """Return an embedder, or None (lexical-only retrieval) if disabled or unavailable."""
    if backend == "none":
        return None
    if backend != "fastembed":
        raise ValueError(f"Unknown EMBEDDING_BACKEND: {backend!r}")
    try:
        return FastEmbedEmbedder(model_name)
    except Exception as exc:  # model download blocked, missing package, etc.
        log.warning("Embedding model unavailable (%s); falling back to lexical-only retrieval.", type(exc).__name__)
        return None
