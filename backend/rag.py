"""End-to-end pipeline: validate -> classify/rewrite -> retrieve -> generate ->
validate -> render. Stateless: conversation history is passed in by the
caller (one history per Streamlit session), so sessions cannot see each other.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from .citations import RenderedCitation, assign_evidence_ids, render_answer
from .config import ABSTAIN_AR, ABSTAIN_EN, Settings
from .conflict_detection import conflict_evidence_ids
from .embeddings import load_embedder
from .generation import generate_answer
from .ingestion import build_or_load_index
from .llm import LLMClient, LLMError, make_llm
from .query_rewriting import Turn, rewrite_query
from .retrieval import HybridRetriever
from .security import InputRejected, detect_injection, redact, sanitize_user_input
from .text_utils import detect_language

log = logging.getLogger(__name__)

OUT_OF_SCOPE_OVERRIDE_BM25 = 4.0

MESSAGES = {
    "abstain": {"en": ABSTAIN_EN, "ar": ABSTAIN_AR},
    "out_of_scope": {
        "en": "I can only answer questions about the three supplied documents on the Almamlaka TV Digital "
        "Expansion Initiative (project overview, budget & timeline, and team & governance). "
        "That question is outside their scope.",
        "ar": "يمكنني الإجابة فقط عن الأسئلة المتعلقة بالوثائق الثلاث المقدّمة حول مبادرة التوسع الرقمي لقناة المملكة "
        "(نظرة عامة على المشروع، والميزانية والجدول الزمني، والفريق والحوكمة). هذا السؤال خارج نطاقها.",
    },
    "refused": {
        "en": "I can't help with that. I only answer questions about the three supplied Almamlaka TV documents, "
        "and my instructions and configuration can't be changed or disclosed.",
        "ar": "لا يمكنني المساعدة في ذلك. أجيب فقط عن الأسئلة المتعلقة بوثائق قناة المملكة الثلاث المقدّمة، "
        "ولا يمكن تغيير تعليماتي أو إعداداتي أو الكشف عنها.",
    },
    "greeting": {
        "en": "Hello! I'm the Almamlaka TV Document Assistant. Ask me about the Digital Expansion Initiative — "
        "its goals, budget, timeline, team or policies — and I'll answer from the three source documents with citations.",
        "ar": "مرحباً! أنا مساعد وثائق قناة المملكة. اسألني عن مبادرة التوسع الرقمي — أهدافها أو ميزانيتها أو جدولها الزمني "
        "أو فريقها أو سياساتها — وسأجيبك من الوثائق الثلاث مع ذكر المصادر.",
    },
    "too_long": {
        "en": "Your message is too long. Please keep questions under {n} characters.",
        "ar": "رسالتك طويلة جداً. يُرجى ألا يتجاوز السؤال {n} حرفاً.",
    },
    "empty": {"en": "Please type a question.", "ar": "يُرجى كتابة سؤال."},
    "error": {
        "en": "Sorry, the assistant is temporarily unavailable: {reason}",
        "ar": "عذراً، المساعد غير متاح مؤقتاً. يُرجى المحاولة لاحقاً. ({reason})",
    },
}


@dataclass
class RenderedConflict:
    topic: str
    sides: list[tuple[str, list[int]]]  # (claim, citation numbers); same value merged
    note: str = ""


def group_sides(conflict, number: dict[str, int]) -> list[tuple[str, list[int]]]:
    """Merge sides that state the same value (e.g. $2,400,000 in two documents)."""
    from .conflict_detection import value_signature

    groups: dict[str, tuple[str, list[int]]] = {}
    for side in conflict.sides:
        key = value_signature(side.claim)
        claim, nums = groups.setdefault(key, (side.claim, []))
        n = number[side.evidence_id]
        if n not in nums:
            nums.append(n)
    return list(groups.values())


@dataclass
class ChatResponse:
    kind: str  # answer | abstain | out_of_scope | greeting | refused | error
    language: str
    text: str
    citations: list[RenderedCitation] = field(default_factory=list)
    conflicts: list[RenderedConflict] = field(default_factory=list)
    debug: dict = field(default_factory=dict)


class RAGPipeline:
    def __init__(self, settings: Settings, retriever: HybridRetriever, llm: LLMClient | None, manifest: dict,
                 llm_error: str = ""):
        self.settings = settings
        self.retriever = retriever
        self.llm = llm
        self.manifest = manifest
        self.llm_error = llm_error

    @classmethod
    def from_settings(cls, settings: Settings) -> "RAGPipeline":
        embedder = load_embedder(settings.embedding_backend, settings.embedding_model)
        store, manifest = build_or_load_index(settings, embedder)
        retriever = HybridRetriever(store, embedder, settings.top_k)
        try:
            llm, err = make_llm(settings), ""
        except LLMError as exc:
            llm, err = None, str(exc)
            log.error("LLM not configured: %s", err)
        return cls(settings, retriever, llm, manifest, err)

    def _msg(self, kind: str, lang: str, **kw) -> ChatResponse:
        return ChatResponse(kind=kind if kind in ("abstain", "out_of_scope", "greeting", "refused", "error") else "refused",
                            language=lang, text=MESSAGES[kind][lang].format(**kw))

    def answer(self, message: str, history: list[Turn] | None = None) -> ChatResponse:
        history = history or []
        t0 = time.time()
        lang = detect_language(message or "")
        try:
            question = sanitize_user_input(message, self.settings.max_input_chars)
        except InputRejected as exc:
            return self._msg(exc.reason if exc.reason in MESSAGES else "empty", lang, n=self.settings.max_input_chars)
        lang = detect_language(question)

        if detect_injection(question):
            log.info("Blocked by injection heuristic: %s", redact(question[:200]))
            resp = self._msg("refused", lang)
            resp.debug = {"blocked_by": "input_heuristic"}
            return resp

        if self.llm is None:
            return self._msg("error", lang, reason=self.llm_error or "model not configured")

        try:
            rewrite = rewrite_query(question, history, self.llm, self.settings.history_turns)
            debug = {"intent": rewrite.intent, "standalone_question": rewrite.standalone_question,
                     "search_queries": rewrite.search_queries, "retrieval_mode": self.retriever.mode}
            if rewrite.intent == "out_of_scope" and self.retriever.max_lexical_score(
                [question, rewrite.standalone_question, *rewrite.search_queries]
            ) >= OUT_OF_SCOPE_OVERRIDE_BM25:
                # It names people/terms from the documents (e.g. "Sara Al-Rashid's email"),
                # so answer "not in the documents" via the normal grounded path.
                rewrite.intent = "document_question"
                debug["intent"] = "document_question (classifier said out_of_scope; overridden by lexical match)"
            if rewrite.intent in ("greeting", "out_of_scope"):
                resp = self._msg(rewrite.intent, lang)
                resp.debug = debug
                return resp
            if rewrite.intent == "prompt_injection":
                resp = self._msg("refused", lang)
                resp.debug = {**debug, "blocked_by": "query_classifier"}
                return resp

            queries = [rewrite.standalone_question, *rewrite.search_queries, question]
            retrieved = self.retriever.retrieve(queries)
            debug["retrieved"] = [(r.chunk.chunk_id, round(r.fused_score, 4)) for r in retrieved]
            if not retrieved:
                resp = self._msg("abstain", lang)
                resp.debug = debug
                return resp

            evidence = assign_evidence_ids([r.chunk for r in retrieved])
            gen = generate_answer(rewrite.standalone_question, evidence, lang, self.llm)
        except LLMError as exc:
            log.error("LLM failure: %s", exc)
            return self._msg("error", lang, reason=str(exc))

        debug.update({"attempts": gen.attempts, "validation_errors": gen.errors,
                      "rejected_conflicts": gen.rejected_conflicts, "seconds": round(time.time() - t0, 2)})
        log.info("q_len=%d lang=%s intent=%s retrieved=%s attempts=%d errors=%d",
                 len(question), lang, rewrite.intent, [c for c, _ in debug["retrieved"]], gen.attempts, len(gen.errors))

        if gen.leaked:
            resp = self._msg("refused", lang)
            resp.debug = {**debug, "blocked_by": "output_canary"}
            return resp
        if gen.output is None or not gen.output.answerable:
            resp = self._msg("abstain", lang)
            resp.debug = debug
            return resp

        out = gen.output
        text, citations = render_answer(out.answer, evidence, conflict_evidence_ids(out.conflicts))
        number = {c.eid: c.number for c in citations}
        conflicts = [RenderedConflict(c.topic, group_sides(c, number), c.note) for c in out.conflicts]
        return ChatResponse("answer", lang, text, citations, conflicts, debug)
