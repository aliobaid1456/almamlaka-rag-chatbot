"""Retrieval on real PDF content (categories 1-4, 10)."""
import numpy as np
import pytest

from backend.retrieval import HybridRetriever
from backend.text_utils import detect_language, tokenize
from backend.vector_store import VectorStore


def ids(results):
    return [r.chunk.chunk_id for r in results]


@pytest.fixture()
def retriever(store):
    return HybridRetriever(store, None, top_k=6)


@pytest.mark.parametrize(
    "query, expected",
    [
        ("Who is the project lead?", "almamlaka_team_governance-p1-c00"),
        ("What is the Q2 2027 marketing budget?", "almamlaka_budget_timeline-p1-c00"),
        ("How many languages will the mobile app support?", "almamlaka_project_overview-p2-c03"),
        ("Who approves budget changes above $50,000?", "almamlaka_team_governance-p1-c01"),
        ("Who should I contact for technical escalations?", "almamlaka_team_governance-p2-c05"),
        ("What is the primary goal of the initiative?", "almamlaka_project_overview-p1-c01"),
        ("Is user data shared with third parties?", "almamlaka_team_governance-p2-c03"),
        ("When is the post-launch review?", "almamlaka_budget_timeline-p2-c02"),
    ],
)
def test_expected_chunk_in_top_k(retriever, query, expected):
    assert expected in ids(retriever.retrieve([query]))


def test_launch_date_retrieves_all_three_documents(retriever):
    got = ids(retriever.retrieve(["When is the public launch date?"]))
    assert "almamlaka_team_governance-p1-c02" in got  # revised: April 1, 2027
    assert "almamlaka_budget_timeline-p2-c02" in got  # timeline: March 15, 2027
    assert "almamlaka_project_overview-p1-c02" in got  # key details: March 15, 2027


def test_budget_retrieves_both_sides_of_conflict(retriever):
    got = ids(retriever.retrieve(["What is the total project budget?"]))
    assert {"almamlaka_project_overview-p1-c02", "almamlaka_budget_timeline-p1-c01"} <= set(got)


def test_exact_amount_lookup(retriever):
    assert ids(retriever.retrieve(["$2,600,000"]))[0] == "almamlaka_budget_timeline-p1-c01"
    assert "almamlaka_budget_timeline-p1-c00" in ids(retriever.retrieve(["720,000"]))


def test_cross_language_via_english_search_queries(retriever):
    """An Arabic question alone has no lexical overlap with English PDFs; the
    rewriter's English queries make it retrievable (dense embeddings add a
    second cross-lingual path in production)."""
    arabic = "من هو قائد المشروع؟"
    assert detect_language(arabic) == "ar"
    got = ids(retriever.retrieve([arabic, "project lead name"]))
    assert "almamlaka_team_governance-p1-c00" in got


def test_tokenizer_normalizes_numbers_and_arabic():
    assert "2400000" in tokenize("Total Approved Budget: $2,400,000")
    assert "2400000" in tokenize("٢٬٤٠٠٬٠٠٠")
    assert tokenize("الميزانيّة") == tokenize("الميزانية")
    assert "ميزانيه" in tokenize("الميزانية")


class FakeEmbedder:
    """Deterministic bag-of-words hashing embedder for testing the dense path."""

    name = "fake"

    def _vec(self, text):
        v = np.zeros(256, dtype="float32")
        for t in tokenize(text):
            v[hash(t) % 256] += 1
        n = np.linalg.norm(v)
        return v / n if n else v

    def embed_documents(self, texts):
        return np.stack([self._vec(t) for t in texts])

    def embed_query(self, text):
        return self._vec(text)


def test_hybrid_mode_fuses_dense_and_lexical(store):
    emb = FakeEmbedder()
    dense_store = VectorStore(store.chunks, emb.embed_documents([c.index_text for c in store.chunks]))
    r = HybridRetriever(dense_store, emb, top_k=6)
    assert r.mode.startswith("hybrid")
    res = r.retrieve(["Who is the lead engineer?"])
    assert res[0].dense_score is not None
    assert "almamlaka_team_governance-p1-c00" in ids(res)
