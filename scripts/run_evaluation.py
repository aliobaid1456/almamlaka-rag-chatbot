"""Evaluate the chatbot against evaluation/test_set.json.

Modes
  python scripts/run_evaluation.py                  full end-to-end (needs LLM_* env vars)
  python scripts/run_evaluation.py --retrieval-only retrieval recall only, no model calls

Writes evaluation/results.json and evaluation/report.md. Every number in the
report is computed from the run; nothing is filled in by hand.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pymupdf  # noqa: E402

from backend.config import Settings  # noqa: E402
from backend.query_rewriting import Turn  # noqa: E402
from backend.rag import RAGPipeline  # noqa: E402
from backend.text_utils import arabic_ratio, normalize_digits  # noqa: E402

EVAL_DIR = ROOT / "evaluation"


def norm(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[\u2010-\u2015\u2212]", "-", text)  # non-breaking / typographic hyphens
    text = re.sub(r"[\u00a0\u2000-\u200b\u202f\u205f]", " ", text)  # unusual spaces
    text = normalize_digits(text.lower())
    text = re.sub(r"(?<=\d),(?=\d{3})", "", text)
    return text


def facts_ok(text: str, groups: list[list[str]]) -> bool:
    t = norm(text)
    return all(any(norm(alt) in t for alt in group) for group in groups)


def slot_recall(slots: list[str], retrieved: list[str]) -> tuple[int, int]:
    hit = sum(1 for s in slots if any(c in retrieved for c in s.split("|")))
    return hit, len(slots)


def citation_valid(c, pdf_dir: Path) -> bool:
    """Re-open the PDF and check that every line of the cited excerpt is on the cited page."""
    try:
        page = pymupdf.open(pdf_dir / c.document_name)[c.page_number - 1]
    except Exception:
        return False
    page_text = " ".join(page.get_text().split())
    for line in c.excerpt.splitlines():
        parts = [p.split(": ", 1)[-1] for p in line.split("; ")] if ";" in line else [line]
        if not all(" ".join(p.split()) in page_text for p in parts):
            return False
    return True


def run_retrieval_only(pipeline: RAGPipeline, cases: list[dict]) -> dict:
    rows = []
    for case in cases:
        if not case.get("chunks"):
            continue
        q = case["turns"][-1] if len(case["turns"]) == 1 else " ".join(case["turns"])
        got = [r.chunk.chunk_id for r in pipeline.retriever.retrieve([q])]
        hit, total = slot_recall(case["chunks"], got)
        rows.append({"id": case["id"], "lang": case["lang"], "hit": hit, "total": total, "retrieved": got})
    summary = {}
    for lang in ("en", "ar"):
        sub = [r for r in rows if r["lang"] == lang]
        if sub:
            summary[lang] = round(sum(r["hit"] for r in sub) / sum(r["total"] for r in sub), 3)
    return {"mode": "retrieval-only (raw user question, no query rewriting)", "retrieval": pipeline.retriever.mode,
            "recall_by_language": summary, "cases": rows}


def run_full(pipeline: RAGPipeline, cases: list[dict], delay: float = 0.0) -> dict:
    rows = []
    for i, case in enumerate(cases):
        if delay and i:
            time.sleep(delay)
        history: list[Turn] = []
        t0 = time.time()
        resp = None
        for turn in case["turns"]:
            resp = pipeline.answer(turn, history)
            history += [Turn("user", turn), Turn("assistant", resp.text)]
        allowed = case["expect"].split("|")
        retrieved = [cid for cid, _ in resp.debug.get("retrieved", [])]
        row = {
            "id": case["id"],
            "category": case["category"],
            "question": case["turns"][-1],
            "turns": case["turns"],
            "expected": case["expect"],
            "kind": resp.kind,
            "behavior_ok": resp.kind in allowed,
            "language_ok": resp.language == case["lang"]
            and (resp.kind != "answer" or (arabic_ratio(resp.text) >= 0.4) == (case["lang"] == "ar")),
            "answer": resp.text,
            "citations": [f"[{c.number}] {c.document_name} p.{c.page_number} {c.section_title}" for c in resp.citations],
            "citations_valid": all(citation_valid(c, pipeline.settings.pdf_dir) for c in resp.citations),
            "conflicts": [{"topic": c.topic, "sides": c.sides} for c in resp.conflicts],
            "retrieved": retrieved,
            "rewrite": resp.debug.get("standalone_question"),
            "attempts": resp.debug.get("attempts"),
            "validation_errors": resp.debug.get("validation_errors", []),
            "seconds": round(time.time() - t0, 1),
        }
        if case.get("chunks"):
            row["recall_hit"], row["recall_total"] = slot_recall(case["chunks"], retrieved)
        if resp.kind == "answer":
            groups = case.get("facts") or case.get("facts_if_answer") or []
            row["facts_ok"] = facts_ok(resp.text + " " + " ".join(s for c in resp.conflicts for s, _ in c.sides), groups)
            row["has_citations"] = bool(resp.citations)
        if case.get("conflict") or resp.kind == "answer":
            row["conflict_expected"] = bool(case.get("conflict"))
            row["conflict_ok"] = case.get("conflict_any", False) or bool(resp.conflicts) == bool(case.get("conflict"))
        row["passed"] = (
            row["behavior_ok"]
            and row["language_ok"]
            and row["citations_valid"]
            and row.get("facts_ok", True)
            and row.get("has_citations", True)
            and (not case.get("conflict") or row.get("conflict_ok", False))
        )
        rows.append(row)
        print(f"{'PASS' if row['passed'] else 'FAIL'} {case['id']:12} kind={resp.kind:12} {row['seconds']:>5}s  {case['turns'][-1][:60]}")

    def rate(key, sub=None):
        sub = [r for r in (sub or rows) if key in r]
        return (round(sum(1 for r in sub if r[key]) / len(sub), 3), len(sub)) if sub else (None, 0)

    answers = [r for r in rows if r["kind"] == "answer"]
    rec = [r for r in rows if "recall_total" in r]
    abstain_cases = [r for r in rows if set(r["expected"].split("|")) <= {"abstain", "out_of_scope", "refused"}]
    conflict_cases = [r for r in rows if r.get("conflict_expected")]
    either = {c["id"] for c in cases if c.get("conflict_any")}
    non_conflict_answers = [r for r in answers if not r.get("conflict_expected") and r["id"] not in either]
    summary = {
        "cases": len(rows),
        "passed": sum(r["passed"] for r in rows),
        "retrieval_recall": round(sum(r["recall_hit"] for r in rec) / sum(r["recall_total"] for r in rec), 3) if rec else None,
        "retrieval_recall_en": _recall([r for r in rec if not r["id"].startswith(("ar-", "fu-03", "fu-04"))]),
        "retrieval_recall_ar": _recall([r for r in rec if r["id"].startswith(("ar-", "fu-03", "fu-04"))]),
        "behavior_accuracy": rate("behavior_ok"),
        "fact_accuracy_on_answers": rate("facts_ok"),
        "citation_validity_on_answers": rate("citations_valid", answers),
        "answers_with_citations": rate("has_citations"),
        "abstention_correctness": rate("behavior_ok", abstain_cases),
        "conflicts_detected": rate("conflict_ok", conflict_cases),
        "false_conflicts_on_other_answers": (sum(1 for r in non_conflict_answers if r["conflicts"]), len(non_conflict_answers)),
        "language_match": rate("language_ok"),
        "security_cases_passed": rate("passed", [r for r in rows if r["id"].startswith("inj-")]),
        "mean_seconds_per_case": round(sum(r["seconds"] for r in rows) / len(rows), 1),
    }
    return {"mode": "end-to-end", "summary": summary, "cases": rows}


def _recall(sub):
    return round(sum(r["recall_hit"] for r in sub) / sum(r["recall_total"] for r in sub), 3) if sub else None


def write_report(result: dict, settings: Settings, pipeline: RAGPipeline) -> None:
    meta = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "llm_provider": settings.llm_provider,
        "llm_model": settings.llm_model,
        "retrieval": pipeline.retriever.mode,
        "embedding_model": pipeline.manifest.get("embedding_model"),
        "top_k": settings.top_k,
    }
    result = {"meta": meta, **result}
    EVAL_DIR.mkdir(exist_ok=True)
    name = "results.json" if result["mode"] == "end-to-end" else "retrieval_results.json"
    (EVAL_DIR / name).write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    if result["mode"] == "end-to-end":
        (EVAL_DIR / "report.md").write_text(render_report(result), encoding="utf-8")


def render_report(result: dict) -> str:
    meta = result["meta"]
    s = result["summary"]
    lines = [
        "# Evaluation report",
        "",
        f"Run: {meta['timestamp_utc']} · model `{meta['llm_model']}` ({meta['llm_provider']}) · retrieval {meta['retrieval']}"
        f" · embeddings `{meta['embedding_model']}` · top_k {meta['top_k']}",
        "",
        "| Metric | Value |",
        "|---|---|",
    ]
    for k, v in s.items():
        if isinstance(v, tuple) or isinstance(v, list):
            v = f"{v[0]} (n={v[1]})" if k != "false_conflicts_on_other_answers" else f"{v[0]} of {v[1]} answers"
        lines.append(f"| {k.replace('_', ' ')} | {v} |")
    lines += ["", "| ID | Category | Question | Expected | Got | Pass |", "|---|---|---|---|---|---|"]
    for r in result["cases"]:
        q = " → ".join(r["turns"]).replace("|", "/")
        lines.append(f"| {r['id']} | {r['category']} | {q} | {r['expected']} | {r['kind']} | {'✅' if r['passed'] else '❌'} |")
    lines += ["", "## Answers", ""]
    for r in result["cases"]:
        lines += [f"### {r['id']} — {' → '.join(r['turns'])}", "", r["answer"], ""]
        if r["conflicts"]:
            for c in r["conflicts"]:
                lines.append(f"- Conflict: **{c['topic']}** — " + "; ".join(f"{s} [{n}]" for s, n in c["sides"]))
        for c in r["citations"]:
            lines.append(f"- {c}")
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--retrieval-only", action="store_true")
    ap.add_argument("--only", help="comma-separated case IDs")
    ap.add_argument("--delay", type=float, default=0.0, help="seconds to wait between cases (free-tier rate limits)")
    args = ap.parse_args()
    settings = Settings.from_env()
    pipeline = RAGPipeline.from_settings(settings)
    cases = json.loads((EVAL_DIR / "test_set.json").read_text(encoding="utf-8"))
    if args.only:
        wanted = set(args.only.split(","))
        cases = [c for c in cases if c["id"] in wanted]
    if args.retrieval_only:
        result = run_retrieval_only(pipeline, cases)
        print(json.dumps(result["recall_by_language"]))
    else:
        if pipeline.llm is None:
            sys.exit(f"LLM not configured: {pipeline.llm_error}")
        result = run_full(pipeline, cases, args.delay)
        print(json.dumps(result["summary"], indent=1))
    write_report(result, settings, pipeline)


if __name__ == "__main__":
    main()
