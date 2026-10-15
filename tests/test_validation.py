"""Citation validation, numeric grounding, language checks and conflicts
(categories 4, 6, 11, 14, 15)."""
import pytest

from backend.citations import (
    Calculation,
    Conflict,
    ConflictSide,
    GenerationOutput,
    assign_evidence_ids,
    normalize_markers,
    render_answer,
    safe_arithmetic,
    validate_answer,
)
from backend.conflict_detection import filter_conflicts, validate_conflict


@pytest.fixture()
def evidence(store):
    by_id = {c.chunk_id: c for c in store.chunks}
    order = [
        "almamlaka_project_overview-p1-c02",  # E1 Key Details: $2,400,000, March 15, 2027
        "almamlaka_budget_timeline-p1-c01",  # E2 Budget Notes: $2,600,000
        "almamlaka_budget_timeline-p1-c00",  # E3 quarterly table
        "almamlaka_team_governance-p1-c02",  # E4 Revised launch: April 1, 2027
    ]
    return assign_evidence_ids([by_id[i] for i in order])


def out(answer, **kw):
    return GenerationOutput(answerable=True, answer=answer, **kw)


def test_valid_grounded_answer(evidence):
    o = out("The originally approved total budget is $2,400,000 [E1].")
    assert validate_answer(o, evidence, "What is the budget?", "en") == []


def test_unknown_evidence_id_rejected(evidence):
    errs = validate_answer(out("The budget is $2,400,000 [E9]."), evidence, "budget?", "en")
    assert any("Unknown evidence IDs" in e for e in errs)


def test_missing_citation_rejected(evidence):
    errs = validate_answer(out("The project lead is Sara Al-Rashid."), evidence, "lead?", "en")
    assert any("no [E#] citation" in e for e in errs)


def test_fabricated_figure_rejected(evidence):
    errs = validate_answer(out("The total budget is $2,500,000 [E1]."), evidence, "budget?", "en")
    assert any("2500000" in e for e in errs)


def test_figure_must_be_in_the_cited_passage_not_just_any_passage(evidence):
    # $2,600,000 is in E2, but the sentence cites only E1.
    errs = validate_answer(out("The revised budget is $2,600,000 [E1]."), evidence, "budget?", "en")
    assert any("2600000" in e for e in errs)


def test_fabricated_page_number_rejected(evidence):
    errs = validate_answer(out("See page 17 of the overview: budget $2,400,000 [E1]."), evidence, "budget?", "en")
    assert any("17" in e for e in errs)


def test_uncited_line_with_figure_rejected(evidence):
    answer = "Budget: $2,400,000 [E1]\n- Revised: $2,600,000"
    errs = validate_answer(out(answer), evidence, "budget?", "en")
    assert any("without a citation marker" in e for e in errs)


def test_million_restatement_allowed(evidence):
    o = out("The original budget was $2.4 million [E1], later revised to $2.6 million [E2].")
    assert validate_answer(o, evidence, "budget?", "en") == []


def test_false_premise_numbers_from_question_allowed(evidence):
    q = "Why was the budget cut to $1,000,000?"
    o = out("The documents do not mention a cut to $1,000,000. The approved budget is $2,400,000 [E1].")
    assert validate_answer(o, evidence, q, "en") == []


def test_verified_calculation_allowed(evidence):
    o = out(
        "The revision added $200,000 [E1][E2].",
        calculations=[Calculation(expression="2,600,000 - 2,400,000", result="200,000")],
    )
    assert validate_answer(o, evidence, "difference?", "en") == []


def test_wrong_calculation_rejected(evidence):
    o = out(
        "The revision added $300,000 [E1][E2].",
        calculations=[Calculation(expression="2,600,000 - 2,400,000", result="300,000")],
    )
    errs = validate_answer(o, evidence, "difference?", "en")
    assert any("does not equal" in e for e in errs)


def test_safe_arithmetic_refuses_code():
    assert safe_arithmetic("$120,000 + $150,000") == 270000
    for bad in ("__import__('os').system('ls')", "2**1000", "open('x')"):
        with pytest.raises(ValueError):
            safe_arithmetic(bad)


def test_language_must_match_question(evidence):
    en_answer = out("The budget is $2,400,000 [E1].")
    assert any("Arabic" in e for e in validate_answer(en_answer, evidence, "ما هي الميزانية؟", "ar"))
    ar_answer = out("الميزانية المعتمدة الإجمالية هي $2,400,000 [E1].")
    assert validate_answer(ar_answer, evidence, "ما هي الميزانية؟", "ar") == []
    assert any("English" in e for e in validate_answer(ar_answer, evidence, "What is the budget?", "en"))


def test_arabic_indic_digits_are_grounded(evidence):
    o = out("الميزانية المعتمدة هي ٢٬٤٠٠٬٠٠٠ دولار [E1].")
    assert validate_answer(o, evidence, "ما هي الميزانية؟", "ar") == []


def test_render_uses_trusted_metadata(evidence):
    text, cites = render_answer("Budget $2,400,000 [E1]; revised $2,600,000 [E2].", evidence)
    assert text == "Budget $2,400,000 [1]; revised $2,600,000 [2]."
    assert (cites[0].document_name, cites[0].page_number, cites[0].section_title) == (
        "Almamlaka_Project_Overview.pdf", 1, "3. Key Details")
    assert (cites[1].document_name, cites[1].page_number) == ("Almamlaka_Budget_Timeline.pdf", 1)


# ---- conflicts ----------------------------------------------------------------
def budget_conflict(a="E1", b="E2", va="$2,400,000", vb="$2,600,000"):
    return Conflict(topic="Total budget", sides=[ConflictSide(claim=f"Total budget {va}", evidence_id=a),
                                                 ConflictSide(claim=f"Total budget {vb}", evidence_id=b)])


def test_genuine_budget_conflict_accepted(evidence):
    assert validate_conflict(budget_conflict(), evidence) == []


def test_genuine_launch_conflict_accepted(evidence):
    c = Conflict(topic="Launch date", sides=[ConflictSide(claim="Planned launch March 15, 2027", evidence_id="E1"),
                                             ConflictSide(claim="Launch moved to April 1, 2027", evidence_id="E4")])
    assert validate_conflict(c, evidence) == []


def test_conflict_rejected_when_value_not_in_cited_passage(evidence):
    assert validate_conflict(budget_conflict(vb="$3,000,000"), evidence)


def test_conflict_rejected_when_sides_cite_same_passage(evidence):
    assert validate_conflict(budget_conflict(b="E1", vb="$2,400,000"), evidence)


def test_conflict_rejected_when_values_identical(evidence):
    c = Conflict(topic="Budget", sides=[ConflictSide(claim="Total $2,400,000", evidence_id="E1"),
                                        ConflictSide(claim="Table total $2,400,000", evidence_id="E3")])
    assert any("same value" in e for e in validate_conflict(c, evidence))


def test_conflict_rejected_with_unknown_id_or_single_side(evidence):
    assert validate_conflict(budget_conflict(b="E7"), evidence)
    assert validate_conflict(Conflict(topic="x", sides=[ConflictSide(claim="a", evidence_id="E1")]), evidence)


def test_filter_keeps_only_supported_conflicts(evidence):
    kept, rejected = filter_conflicts([budget_conflict(), budget_conflict(vb="$9,999,999")], evidence)
    assert len(kept) == 1 and rejected


def test_normalize_markers():
    assert normalize_markers("x (E1) y （E2） z 【E3】 w ]E4[") == "x [E1] y [E2] z [E3] w [E4]"
    assert normalize_markers("(E1, E2)") == "[E1, E2]"
    assert normalize_markers("(see page 2)") == "(see page 2)"
