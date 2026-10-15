"""Evidence identifiers, citation validation and grounding checks.

The model never writes document names or page numbers. It may only reference
evidence IDs (E1, E2, ...) that the application assigned to the passages it
retrieved for *this* question. Code then:

  * rejects unknown IDs,
  * requires every line that states a figure to carry a citation,
  * checks every number in the answer against the cited passages (or the
    user's own question, or an arithmetic step the code re-computes),
  * renders citations from trusted chunk metadata.
"""
from __future__ import annotations

import ast
import operator
import re
from dataclasses import dataclass

from pydantic import BaseModel, Field

from .chunking import Chunk
from .text_utils import arabic_ratio, extract_numbers, normalize_digits

MARKER_RE = re.compile(r"\[\s*(E\d+(?:\s*[,،]\s*E\d+)*)\s*\]")


class ConflictSide(BaseModel):
    claim: str
    evidence_id: str


class Conflict(BaseModel):
    topic: str
    sides: list[ConflictSide] = Field(default_factory=list)
    note: str = ""


class Calculation(BaseModel):
    expression: str
    result: str


class GenerationOutput(BaseModel):
    answerable: bool
    answer: str = ""
    evidence_ids: list[str] = Field(default_factory=list)
    conflicts: list[Conflict] = Field(default_factory=list)
    calculations: list[Calculation] = Field(default_factory=list)


@dataclass
class Evidence:
    eid: str
    chunk: Chunk


def assign_evidence_ids(chunks: list[Chunk]) -> dict[str, Evidence]:
    return {f"E{i + 1}": Evidence(f"E{i + 1}", c) for i, c in enumerate(chunks)}


_ALT_MARKER_RE = re.compile(r"[\(（【]\s*(E\d+(?:\s*[,،]\s*E\d+)*)\s*[\)）】]|\]\s*(E\d+)\s*\[")


def normalize_markers(text: str) -> str:
    """Accept (E1), （E1）, 【E1】 and RTL-mirrored ]E1[ as [E1]; models sometimes vary
    the bracket shape, especially in Arabic answers."""
    return _ALT_MARKER_RE.sub(lambda m: f"[{m.group(1) or m.group(2)}]", text)


def markers_in(text: str) -> list[str]:
    ids: list[str] = []
    for group in MARKER_RE.findall(text):
        for eid in re.split(r"\s*[,،]\s*", group):
            if eid not in ids:
                ids.append(eid)
    return ids


def strip_markers(text: str) -> str:
    return MARKER_RE.sub("", text)


# ---- safe arithmetic ---------------------------------------------------------
_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv}


def _eval(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_eval(node.operand)
    raise ValueError("unsupported expression")


def safe_arithmetic(expression: str) -> float:
    expr = normalize_digits(expression).replace("$", "").replace(",", "").replace("×", "*").replace("÷", "/")
    expr = re.sub(r"(\d)\s*%", r"\1", expr)
    if not re.fullmatch(r"[\d\.\s\+\-\*/\(\)]+", expr) or len(expr) > 200:
        raise ValueError("unsupported expression")
    return _eval(ast.parse(expr, mode="eval"))


def _num(value: str) -> float | None:
    try:
        return float(normalize_digits(value).replace("$", "").replace(",", "").replace("%", "").strip())
    except ValueError:
        return None


def _canonical(x: float) -> str:
    return str(int(x)) if abs(x - round(x)) < 1e-9 else f"{x:.6f}".rstrip("0").rstrip(".")


def verify_calculations(calcs: list[Calculation], evidence_numbers: set[str]) -> tuple[set[str], list[str]]:
    """Recompute each declared calculation. Operands must come from evidence."""
    allowed: set[str] = set()
    errors: list[str] = []
    for c in calcs:
        try:
            value = safe_arithmetic(c.expression)
        except (ValueError, SyntaxError, ZeroDivisionError):
            errors.append(f"Calculation '{c.expression}' is not a valid arithmetic expression.")
            continue
        claimed = _num(c.result)
        if claimed is None or abs(claimed - value) > max(0.005 * abs(value), 0.01):
            errors.append(f"Calculation '{c.expression}' does not equal {c.result} (it equals {_canonical(value)}).")
            continue
        operands = extract_numbers(c.expression)
        missing = operands - evidence_numbers
        if missing:
            errors.append(f"Calculation operands {sorted(missing)} do not appear in the cited evidence.")
            continue
        allowed |= extract_numbers(_canonical(value)) | extract_numbers(c.result)
        if abs(value) >= 1000:  # allow "$2.6 million" style restatement of the result
            allowed |= extract_numbers(_canonical(value / 1_000_000)) | extract_numbers(_canonical(value / 1000))
    return allowed, errors


def _scaled_forms(numbers: set[str]) -> set[str]:
    """'2400000' also licenses '2.4' (million) and '2400' (thousand)."""
    out = set(numbers)
    for n in numbers:
        v = _num(n)
        if v is not None and v >= 1000:
            out |= extract_numbers(_canonical(v / 1_000_000)) | extract_numbers(_canonical(v / 1000))
    return out


def evidence_numbers(evidence: dict[str, Evidence], ids: list[str]) -> set[str]:
    nums: set[str] = set()
    for eid in ids:
        if eid in evidence:
            ch = evidence[eid].chunk
            nums |= extract_numbers(ch.text) | extract_numbers(ch.section_title)
    return nums


def validate_answer(
    out: GenerationOutput, evidence: dict[str, Evidence], question: str, language: str
) -> list[str]:
    """Return a list of human-readable validation errors (empty = valid)."""
    errors: list[str] = []
    if not out.answerable:
        return errors
    answer = out.answer.strip()
    if not answer:
        return ["answerable is true but the answer is empty."]

    cited = markers_in(answer)
    unknown = [e for e in cited + out.evidence_ids if e not in evidence]
    if unknown:
        errors.append(f"Unknown evidence IDs {sorted(set(unknown))}; only {list(evidence)} exist.")
    if not cited:
        errors.append("The answer has no [E#] citation markers.")

    valid_cited = [e for e in cited if e in evidence]
    ev_nums = evidence_numbers(evidence, valid_cited)
    calc_allowed, calc_errors = verify_calculations(out.calculations, evidence_numbers(evidence, list(evidence)))
    errors.extend(calc_errors)
    allowed = _scaled_forms(ev_nums) | calc_allowed | extract_numbers(question)

    for line in answer.splitlines():
        body = strip_markers(line)
        nums = extract_numbers(body)
        if nums and not MARKER_RE.search(line):
            errors.append(f"This line states figures without a citation marker: '{body.strip()[:80]}'")
        ungrounded = nums - allowed
        if ungrounded:
            errors.append(f"Figures {sorted(ungrounded)} do not appear in the cited evidence.")

    prose = strip_markers(answer)
    ratio = arabic_ratio(prose)
    if language == "ar" and ratio < 0.4:
        errors.append("The question is in Arabic, so the answer must be written in Arabic.")
    if language == "en" and ratio > 0.2:
        errors.append("The question is in English, so the answer must be written in English.")
    return errors


@dataclass
class RenderedCitation:
    number: int
    eid: str
    document_name: str
    document_title: str
    page_number: int
    section_title: str
    excerpt: str


def render_answer(answer: str, evidence: dict[str, Evidence], extra_ids: list[str] = ()) -> tuple[str, list[RenderedCitation]]:
    """Replace [E#] markers with numbered references built from trusted metadata."""
    order: list[str] = []
    for eid in markers_in(answer) + list(extra_ids):
        if eid in evidence and eid not in order:
            order.append(eid)
    number = {eid: i + 1 for i, eid in enumerate(order)}

    def repl(m: re.Match) -> str:
        ids = [e for e in re.split(r"\s*[,،]\s*", m.group(1)) if e in number]
        return "".join(f" [{number[e]}]" for e in ids)

    text = MARKER_RE.sub(repl, answer)
    text = re.sub(r"[ \t]+\[", " [", text)
    citations = [
        RenderedCitation(
            number=number[eid],
            eid=eid,
            document_name=evidence[eid].chunk.document_name,
            document_title=evidence[eid].chunk.document_title,
            page_number=evidence[eid].chunk.page_number,
            section_title=evidence[eid].chunk.section_title,
            excerpt=evidence[eid].chunk.text,
        )
        for eid in order
    ]
    return text, citations
