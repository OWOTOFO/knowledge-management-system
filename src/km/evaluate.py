"""Deterministic acceptance checks for question/answer fixtures."""

from __future__ import annotations

import re
from decimal import Decimal

from km.text import SCALE, normalize_quote, strip_citations

_SCALES = {name: Decimal(int(scale)) for name, scale in SCALE.items()}
_SCALES.update({"m": _SCALES["million"], "b": _SCALES["billion"], "mn": _SCALES["million"]})
_AMOUNT = re.compile(
    r"(?<![\w.,$+−-])(?P<sign>[+−-]?)\s*(?:(?:US\$|\$|USD\b)\s*)?"
    r"(?P<currency_sign>[+−-]?)\s*(?P<number>(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)"
    r"(?!\d|[.,]\d)(?:\s*(?P<scale>" + "|".join(_SCALES) + r")\b)?(?!\s*(?:%|percent\b))",
    re.IGNORECASE,
)


def _usd_amounts(text: str) -> set[Decimal]:
    amounts = set()
    for match in _AMOUNT.finditer(strip_citations(text)):
        amount = Decimal(match["number"].replace(",", ""))
        scale = _SCALES.get((match["scale"] or "").lower(), 1)
        sign = -1 if (match["sign"] or match["currency_sign"]) in ("-", "−") else 1
        amounts.add(sign * amount * scale)
    return amounts


def _row_value(value):
    return value.casefold() if isinstance(value, str) else value


def _matches_citation(target: dict, citation: dict) -> bool:
    if target["kind"] != citation.get("kind"):
        return False
    if target["kind"] == "sql":
        return True
    return (
        bool(citation.get("quotes"))
        and (not target.get("filing") or (citation.get("url") or "").endswith("/" + target["filing"]))
        and normalize_quote(target.get("section", "")) in normalize_quote(citation.get("location") or "")
    )


def answer_issues(case: dict, result: dict) -> list[str]:
    """Check answer content and cited evidence independently of the pipeline's own confidence."""
    if case.get("expect_abstain"):
        return []
    if result["status"] != "answered":
        return ["No answer was returned."]

    answer = result["answer"]
    citations = result["citations"]
    issues = [
        f"Answer is missing {phrase!r}."
        for phrase in case.get("must_contain", [])
        if phrase.casefold() not in answer.casefold()
    ]
    heading = case.get("must_cite_heading", "").casefold()
    if heading and not any(heading in (c["location"] or "").casefold() for c in citations):
        issues.append(f"No citation to heading {heading!r}.")
    for pattern in case.get("answer_patterns", []):
        if not re.search(pattern, answer, re.IGNORECASE):
            issues.append(f"Answer does not match {pattern!r}.")

    lines = [re.sub(r"^\s*[-*]\s+", "", line) for line in answer.splitlines()]
    for target in case.get("answer_amounts_usd", []):
        value = target["usd"]
        context = target.get("on_line")
        if not any((not context or re.search(context, line, re.IGNORECASE)) and value in _usd_amounts(line)
                   for line in lines):
            suffix = f" on a line matching {context!r}" if context else ""
            issues.append(f"Answer is missing the exact USD amount {value:,}{suffix}.")

    for target in case.get("required_citations", []):
        if not any(_matches_citation(target, c) for c in citations):
            source = ", ".join(target[key] for key in ("kind", "filing", "section") if target.get(key))
            issues.append(f"Missing cited evidence ({source}).")
    sql_rows = [{_row_value(value) for value in row} for row in result.get("sql_rows", [])]
    for row in case.get("required_sql_rows", []):
        required = {_row_value(value) for value in row}
        if not any(required <= actual for actual in sql_rows):
            issues.append(f"SQL results lack a row containing {row!r}.")
    return issues
