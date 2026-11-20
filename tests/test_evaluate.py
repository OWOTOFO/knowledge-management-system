import json
from pathlib import Path

import pytest
import yaml

from km.cli import main
from km.evaluate import answer_issues

SEC_CASES = {
    case["id"]: case
    for case in yaml.safe_load((Path(__file__).parents[1] / "eval" / "sec_questions.yaml").read_text())
}


@pytest.mark.parametrize("amount", ["$391.035 billion", "$391,035 million", "USD 391,035,000,000"])
def test_accepts_equivalent_exact_dollar_amounts(amount):
    result = {
        "status": "answered", "answer": f"FY2024: {amount} [S1].", "citations": [{"kind": "sql"}],
        "sql_rows": [["AAPL", 2024, "USD", 391035000000]],
    }
    assert answer_issues(SEC_CASES["aapl-fy24-net-sales"], result) == []


@pytest.mark.parametrize("amount", ["$391.0 billion", "-$391.035 billion", "$391,035,000,000 billion"])
def test_rejects_wrong_precision_sign_or_scale(amount):
    result = {
        "status": "answered", "answer": f"FY2024: {amount} [S1].", "citations": [{"kind": "sql"}],
        "sql_rows": [["AAPL", 2024, "USD", 391035000000]],
    }
    issues = answer_issues(SEC_CASES["aapl-fy24-net-sales"], result)
    assert len(issues) == 1 and "exact USD amount" in issues[0]


def test_requires_sql_even_when_exact_revenue_was_quoted_from_document():
    result = {
        "status": "answered", "answer": "Meta's Q3 2026 revenue was $64.601 billion [D1].",
        "citations": [{"kind": "doc", "quotes": ["Total revenue was $64.601 billion."]}],
    }
    assert answer_issues(SEC_CASES["meta-q3-fy26-revenue"], result) == [
        "Missing cited evidence (sql).", "SQL results lack a row containing ['META', 2026, 'Q3', 'USD', 64601000000].",
    ]


def test_rejects_swapped_company_amounts_in_result_rows_and_answer():
    result = {
        "status": "answered", "answer": "Apple: $62.36 billion [S1].\nMeta: $93.736 billion [S1].",
        "citations": [{"kind": "sql"}], "sql_rows": [["AAPL", 2024, "USD", 62360000000],
                                                     ["META", 2024, "USD", 93736000000]],
    }
    issues = answer_issues(SEC_CASES["aapl-meta-fy24-net-income"], result)
    assert sum("exact USD amount" in issue for issue in issues) == 2
    assert sum("SQL results lack a row" in issue for issue in issues) == 2


def test_legal_citation_must_point_to_part_ii():
    result = {
        "status": "answered",
        "answer": "The cases allege Meta collected personal information from third-party sites [D1].",
        "citations": [{
            "kind": "doc", "url": "https://www.sec.gov/archives/meta-20260930.htm", "quotes": ["business tools"],
            "location": "Part I, Item 1. Financial Statements, paragraph 7",
        }],
    }
    case = SEC_CASES["meta-q3-fy26-meta-pixel"]
    assert len(answer_issues(case, result)) == 1
    result["citations"][0]["location"] = "Part II, Item 1. Legal Proceedings, paragraph 12"
    assert answer_issues(case, result) == []


def test_eval_saves_a_failure_report_and_exits_nonzero(tmp_path, pipeline, monkeypatch):
    monkeypatch.setattr("km.graph.KMPipeline", lambda: pipeline)
    fixture = tmp_path / "questions.yaml"
    fixture.write_text(yaml.safe_dump([{
        "id": "intentional-mismatch", "question": "How many complaints were logged per category?",
        "expected_route": "structured", "must_contain": ["should never appear in the answer"],
    }]))
    report = tmp_path / "report.json"
    with pytest.raises(SystemExit) as exc:
        main(["eval", "--file", str(fixture), "--report", str(report)])
    assert exc.value.code == 1
    data = json.loads(report.read_text())
    assert data["summary"]["passed"] == 0
    assert data["results"][0]["id"] == "intentional-mismatch"
    assert data["results"][0]["checks"] == {"route": True, "status": True, "content": False}
