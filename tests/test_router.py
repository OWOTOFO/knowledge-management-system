import pytest

from km.router import Router, heuristic_route


@pytest.mark.parametrize("question,expected", [
    ("How many complaints were logged per category?", "structured"),
    ("What was the total loan volume by region?", "structured"),
    ("What should I do if I lose my laptop while working remotely?", "unstructured"),
    ("What is the procedure for handling a credit card dispute?", "unstructured"),
    ("Are credit card complaint resolution times within the dispute policy SLA?", "hybrid"),
    ("Compare our mortgage interest rates with the early repayment terms in the mortgage guide.", "hybrid"),
])
def test_heuristic_routes(question, expected):
    assert heuristic_route(question).route == expected


def test_low_confidence_escalates_to_hybrid():
    router = Router(llm=None, schema_text="t(a int)", catalog=[{"title": "T", "source": "t.pdf"}], min_confidence=0.6)
    d = router.route("Tell me about Lakeview")   # no cues at all
    assert d.route == "hybrid"
    assert d.doc_question and d.sql_question


CATALOG = [{"title": "Apple Inc. 10-K FY2024", "source": "a.htm", "ticker": "AAPL", "company": "Apple Inc."},
           {"title": "Meta Platforms, Inc. 10-K FY2024", "source": "m.htm", "ticker": "META", "company": "Meta Platforms, Inc."}]


def test_companies_and_missing_stores():
    router = Router(llm=None, schema_text="t(a int)", catalog=CATALOG, min_confidence=0.6)
    assert router.route("Compare Apple and Meta revenue").companies == ["AAPL", "META"]
    assert router.route("What risks does AAPL describe?").companies == ["AAPL"]
    assert router.route("What is the remote work policy?").companies == []
    assert Router(None, "", CATALOG, 0.6).route("How many complaints?").route == "unstructured"   # no database
    assert Router(None, "t(a int)", [], 0.6).route("What is the policy?").route == "structured"   # no documents
