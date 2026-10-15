"""Shared fixtures. The LLM is replaced by a scripted fake so the tests are
deterministic and run offline; they test the application's own logic
(retrieval, validation, citations, conflicts, security, sessions). End-to-end
answer quality with a real model is measured by scripts/run_evaluation.py."""
from __future__ import annotations

import json
import re
import shutil
import sys
from pathlib import Path
from typing import Callable

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.config import Settings  # noqa: E402
from backend.ingestion import build_or_load_index  # noqa: E402
from backend.rag import RAGPipeline  # noqa: E402
from backend.retrieval import HybridRetriever  # noqa: E402

PDF_DIR = ROOT / "data" / "pdfs"
EVIDENCE_RE = re.compile(r'<evidence id="(E\d+)" document="([^"]+)" page="(\d+)" section="([^"]*)"[^>]*>\n(.*?)\n</evidence>', re.S)


def parse_evidence(prompt: str) -> dict[str, dict]:
    return {
        m.group(1): {"document": m.group(2), "page": int(m.group(3)), "section": m.group(4), "text": m.group(5)}
        for m in EVIDENCE_RE.finditer(prompt)
    }


def eid_with(prompt: str, needle: str, document: str | None = None) -> str:
    for eid, ev in parse_evidence(prompt).items():
        if needle in ev["text"] and (document is None or ev["document"] == document):
            return eid
    raise AssertionError(f"no evidence containing {needle!r} was retrieved")


class FakeLLM:
    """Routes calls to a rewrite handler or an answer handler by system prompt."""

    def __init__(self, answer: Callable[[str], dict | str] | None = None, rewrite: Callable[[str], dict] | None = None):
        self.answer_fn = answer
        self.rewrite_fn = rewrite
        self.calls: list[tuple[str, str]] = []

    def complete(self, system: str, user: str, max_tokens: int = 1200) -> str:
        self.calls.append((system, user))
        if "query-analysis component" in system:
            if self.rewrite_fn:
                return json.dumps(self.rewrite_fn(user), ensure_ascii=False)
            q = re.search(r"<current_message>\n(.*)\n</current_message>", user, re.S).group(1)
            return json.dumps({"intent": "document_question", "standalone_question": q, "search_queries": [q]})
        if self.answer_fn is None:
            return json.dumps({"answerable": False, "answer": ""})
        out = self.answer_fn(user)
        return out if isinstance(out, str) else json.dumps(out, ensure_ascii=False)


@pytest.fixture(scope="session")
def settings(tmp_path_factory) -> Settings:
    s = Settings()
    s.embedding_backend = "none"
    s.pdf_dir = PDF_DIR
    s.index_dir = tmp_path_factory.mktemp("index")
    return s


@pytest.fixture(scope="session")
def store(settings):
    store, _ = build_or_load_index(settings, None)
    return store


@pytest.fixture()
def make_pipeline(settings, store):
    def _make(llm) -> RAGPipeline:
        return RAGPipeline(settings, HybridRetriever(store, None, settings.top_k), llm, {"chunk_count": len(store.chunks)})

    return _make


@pytest.fixture()
def tmp_pdf_dir(tmp_path):
    d = tmp_path / "pdfs"
    d.mkdir()
    for p in PDF_DIR.glob("*.pdf"):
        shutil.copy(p, d / p.name)
    return d
