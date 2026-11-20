from km.validate import Validator

DOC = "Credit Card Dispute Policy | page 2, paragraph 3\nStandard disputes must be resolved within 30 calendar days of receipt."
SQL_ROWS = {"columns": ["complaints", "resolved_after_30_days"], "rows": [[173, 63]]}
EVIDENCE = [
    {"id": "D1", "kind": "doc", "source": {"title": "Credit Card Dispute Policy"}, "text": DOC},
    {"id": "S1", "kind": "sql", "source": {"sql": "..."}, **SQL_ROWS,
     "text": "SQL: SELECT ... WHERE days_to_resolve > 30 ... LIMIT 50\ncomplaints | resolved_after_30_days\n173 | 63"},
]
Q_OK = [{"id": "D1", "quote": "Standard disputes must be resolved within 30 calendar days of receipt"}]

v = Validator(llm=None, threshold=0.8)


def test_grounded_answer_passes_with_high_confidence():
    r = v.validate("Standard disputes must be resolved within 30 calendar days [D1]. "
                   "The database shows 173 complaints, 63 resolved after 30 days [S1].", Q_OK, EVIDENCE)
    assert r.passed, r.reasons
    assert r.groundedness == 1.0
    assert r.confidence_label == "High"
    assert "every claim is stated directly" in r.confidence_explanation


def test_fabricated_number_fails():
    r = v.validate("Standard disputes must be resolved within 45 calendar days [D1].", Q_OK, EVIDENCE)
    assert not r.passed and "45" in r.unsupported_numbers


def test_number_from_sql_text_not_rows_fails():
    # "50" only appears in the query's LIMIT, not in any result row
    r = v.validate("50 complaints were resolved after 30 days [S1].", [], EVIDENCE)
    assert not r.passed and "50" in r.unsupported_numbers


def test_scaled_numbers_match_raw_values():
    ev = [{"id": "S1", "kind": "sql", "source": {}, "columns": ["value"], "rows": [[391035000000]],
           "text": "value\n391035000000"}]
    assert v.validate("Net sales were $391.0 billion [S1].", [], ev).passed
    assert v.validate("Net sales were $391,035 million [S1].", [], ev).passed
    assert not v.validate("Net sales were $392.0 billion [S1].", [], ev).passed


def test_invalid_citation_fails():
    r = v.validate("Standard disputes must be resolved within 30 calendar days [D7].", Q_OK, EVIDENCE)
    assert not r.passed and r.invalid_citations == ["D7"]


def test_fake_quote_or_missing_quote_fails():
    fake = [{"id": "D1", "quote": "disputes are resolved within 30 days by the fraud team"}]
    r = v.validate("Standard disputes must be resolved within 30 calendar days [D1].", fake, EVIDENCE)
    assert not r.passed and r.invalid_quotes
    r = v.validate("Standard disputes must be resolved within 30 calendar days [D1].", [], EVIDENCE)
    assert not r.passed and r.missing_quotes == ["D1"]


def test_weak_claim_gives_low_confidence_and_names_it():
    r = v.validate("Disputes are handled by the fraud team in Singapore [D1].", Q_OK, EVIDENCE)
    assert not r.passed
    assert r.confidence_label == "Low"
    assert "fraud team in Singapore" in r.confidence_explanation


# --- gaps found in the audit (real 10-K shapes) ---
TENK = [{"id": "D1", "kind": "doc", "source": {"fiscal_year": 2024, "heading": "Item 7. MD&A"},
         "text": "Apple Inc. 10-K FY2024 (fiscal year 2024) | Item 7. MD&A, paragraph 4\n"
                 "Services net sales increased 13% during 2024 compared to 2023."}]
Q_TENK = [{"id": "D1", "quote": "Services net sales increased 13% during 2024"}]


def test_section_and_form_labels_are_not_figures():
    # "Item 7" must not make a wrong "7%" pass, and naming "the 10-K" must not fail the number check
    assert not v.validate("Services net sales increased 7% during 2024 [D1].", Q_TENK, TENK).passed
    r = v.validate("According to Apple's 10-K (Item 7), services net sales increased 13% during 2024 [D1].",
                   Q_TENK, TENK)
    assert r.passed, r.reasons


def test_quote_from_the_label_line_is_rejected():
    r = v.validate("Services grew [D1].", [{"id": "D1", "quote": "Apple Inc. 10-K FY2024 (fiscal year 2024)"}], TENK)
    assert not r.passed and r.invalid_quotes


class _Judge:
    name = "fake"

    def __init__(self, claims=None, error=None):
        self.claims, self.error = claims, error

    def structured(self, system, user, schema):
        if self.error:
            raise self.error
        return schema(claims=self.claims)


def test_a_contradicted_claim_blocks_the_answer():
    from km.validate import ClaimCheck
    ok = [ClaimCheck(claim=f"c{i}", citations=["D1"], verdict="supported", confidence=0.95, reason="ok") for i in range(4)]
    bad = ClaimCheck(claim="Services fell", citations=["D1"], verdict="contradicted", confidence=0.95, reason="says rose")
    r = Validator(_Judge(ok + [bad]), 0.8).validate("Services net sales increased during 2024 [D1].", Q_TENK, TENK)
    assert not r.passed and r.confidence_label == "Low"
    assert any("contradict" in reason for reason in r.reasons)


def test_failed_fact_check_call_fails_closed():
    r = Validator(_Judge(error=TimeoutError("upstream timeout")), 0.8).validate(
        "Services net sales increased 13% during 2024 [D1].", Q_TENK, TENK)
    assert not r.passed
    assert any("fact-check model call failed" in reason for reason in r.reasons)


def test_fiscal_year_labels_like_fy2025_are_not_figures():
    rows = {"columns": ["ticker", "period_fiscal_year", "value"], "rows": [["AAPL", 2025, 416161000000]]}
    ev = [{"id": "S1", "kind": "sql", "source": {"sql": "..."}, **rows, "text": "SQL: ...\nticker | period_fiscal_year | value"}]
    r = v.validate("FY2025 total net sales were $416,161,000,000 [S1].", [], ev)
    assert r.passed, r.reasons


def test_citation_forms_models_actually_write():
    from km.text import extract_citations, strip_citations
    text = 'A [D1, D2]. B [D8: "Total revenue | 64,601"]. C [S1].'
    assert extract_citations(text) == ["D1", "D2", "D8", "S1"]
    assert strip_citations(text) == "A . B . C ."


def test_numbers_may_come_from_the_cited_paragraph_not_only_the_quote():
    para = [{"id": "D1", "kind": "doc", "source": {"fiscal_year": 2026},
             "text": "Meta 10-Q | Part I, Item 2, paragraph 9\nCost of revenue rose $4.16 billion. R&D rose $4.91 billion."},
            {"id": "D2", "kind": "doc", "source": {}, "text": "x | y\nHeadcount was 76,834."}]
    q = [{"id": "D1", "quote": "Cost of revenue rose"}]
    assert v.validate("Cost of revenue rose $4.16 billion and R&D $4.91 billion [D1].", q, para).passed
    # a number from a passage the answer does not cite is still rejected
    assert "76834" in "".join(v.validate("Cost of revenue rose with 76,834 staff [D1].", q, para).unsupported_numbers)


def test_years_inside_sql_column_names_count():
    rows = {"columns": ["q3_fy2025_value", "q3_fy2026_value"], "rows": [[30707000000, 51613000000]]}
    ev = [{"id": "S1", "kind": "sql", "source": {"sql": "..."}, **rows, "text": "SQL: ...\nq3_fy2025_value | q3_fy2026_value"}]
    assert v.validate("Costs were $51,613,000,000 in Q3 2026 versus $30,707,000,000 in Q3 2025 [S1].", [], ev).passed


def test_fact_checker_can_say_the_answer_misses_the_question():
    from km.validate import ClaimCheck, JudgeOutput

    class Judge:
        name = "fake"

        def structured(self, system, user, schema):
            assert "QUESTION:\nWhat is the pet insurance policy?" in user
            return JudgeOutput(claims=[ClaimCheck(claim="Services rose", citations=["D1"], verdict="supported",
                                                  confidence=0.95, reason="stated")], answers_question=False)

    r = Validator(Judge(), 0.8).validate("Services net sales increased 13% during 2024 [D1].", Q_TENK, TENK,
                                         question="What is the pet insurance policy?")
    assert not r.passed and r.reasons[0] == "the evidence found does not answer the question"


def test_billions_match_a_filing_table_stated_in_millions():
    table = [{"id": "D1", "kind": "doc", "source": {}, "text": "Meta 10-Q | Part I, Item 2, paragraph 5\n"
              "Provision for income taxes | 3,141 | 18,954 | (83)%\nTotal net sales | 391,035"}]
    q = [{"id": "D1", "quote": "Provision for income taxes | 3,141 | 18,954"}]
    def unsupported(answer):   # the deterministic number check only (the offline judge can't read units)
        return v.validate(answer, q, table).unsupported_numbers

    assert unsupported("The provision for income taxes was $3.141 billion, down from $18.954 billion [D1].") == []
    assert unsupported("Total net sales were $391.0 billion [D1].") == []
    assert unsupported("The provision for income taxes was $3.5 billion [D1].") == ["3.5 billion"]


def test_a_mislabelled_quote_is_pointed_at_the_paragraph_that_contains_it():
    from km.validate import repair_citations
    ev = [{"id": f"D{i}", "kind": "doc", "source": {}, "text": f"Meta 10-Q | paragraph {i}\nFiller sentence number {i} about other matters."}
          for i in range(1, 4)]
    ev[1]["text"] = "Meta 10-Q | paragraph 2\nThe decrease was mostly due to the effects of the One Big Beautiful Bill Act."
    quotes = [{"id": "D25", "quote": "mostly due to the effects of the One Big Beautiful Bill Act"}]
    answer, fixed = repair_citations("The tax rate fell mostly due to the effects of the One Big Beautiful Bill Act [D25].", quotes, ev)
    assert "[D2]" in answer and "D25" not in answer and fixed[0]["id"] == "D2"
    assert v.validate(answer, fixed, ev).passed
    # words that appear nowhere stay wrong: nothing is "repaired" into a pass
    answer, fixed = repair_citations("Revenue tripled [D25].", [{"id": "D25", "quote": "revenue tripled in the quarter"}], ev)
    assert "[D25]" in answer and not v.validate(answer, fixed, ev).passed
