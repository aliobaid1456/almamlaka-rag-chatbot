"""Cross-document conflict handling.

Pipeline:
1. Retrieval guarantees cross-document coverage (see retrieval.py), so the
   generator sees both sides of a disagreement.
2. The generator reports conflicts in a dedicated structured field: a topic and
   two or more sides, each a claim tied to one evidence ID.
3. This module validates every reported conflict in code before it is shown:
   - each side cites a real evidence ID from this retrieval,
   - the sides cite different passages,
   - every figure in a side's claim appears in that side's passage,
   - the sides actually state different values (not a restatement).
   Conflicts failing these checks are discarded and logged — a conflict is only
   shown when the evidence supports it.
4. The UI renders validated conflicts in a separate, clearly marked panel with
   citations for every side; the app never picks a winner.
"""
from __future__ import annotations

import logging

from .citations import Conflict, Evidence
from .text_utils import extract_numbers, normalize_digits

log = logging.getLogger(__name__)


def value_signature(claim: str) -> str:
    nums = sorted(extract_numbers(claim))
    if nums:
        return "|".join(nums)
    return " ".join(normalize_digits(claim).lower().split())


def validate_conflict(conflict: Conflict, evidence: dict[str, Evidence]) -> list[str]:
    errors: list[str] = []
    if len(conflict.sides) < 2:
        return [f"Conflict '{conflict.topic}' needs at least two sides."]
    ids = [s.evidence_id for s in conflict.sides]
    unknown = [e for e in ids if e not in evidence]
    if unknown:
        return [f"Conflict '{conflict.topic}' cites unknown evidence IDs {unknown}."]
    if len(set(ids)) < 2:
        errors.append(f"Conflict '{conflict.topic}': all sides cite the same passage.")
    for side in conflict.sides:
        chunk_nums = extract_numbers(evidence[side.evidence_id].chunk.text)
        missing = extract_numbers(side.claim) - chunk_nums
        if missing:
            errors.append(
                f"Conflict '{conflict.topic}': figures {sorted(missing)} are not in {side.evidence_id}."
            )
    if len({value_signature(s.claim) for s in conflict.sides}) < 2:
        errors.append(f"Conflict '{conflict.topic}': the sides state the same value, so it is not a conflict.")
    return errors


def filter_conflicts(conflicts: list[Conflict], evidence: dict[str, Evidence]) -> tuple[list[Conflict], list[str]]:
    kept: list[Conflict] = []
    rejected: list[str] = []
    for c in conflicts:
        errs = validate_conflict(c, evidence)
        if errs:
            rejected.extend(errs)
            log.info("Discarded unsupported conflict: %s", errs)
        else:
            kept.append(c)
    return kept, rejected


def conflict_evidence_ids(conflicts: list[Conflict]) -> list[str]:
    ids: list[str] = []
    for c in conflicts:
        for s in c.sides:
            if s.evidence_id not in ids:
                ids.append(s.evidence_id)
    return ids

