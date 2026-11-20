"""Intent routing: decide whether a question needs documents, SQL, or both, and which companies.

With an LLM the router gets the *catalog* of both stores (table schema + document titles) and
returns a typed decision.  Without one (mock mode) a transparent cue-based classifier is used.
Low-confidence decisions fall back to HYBRID: over-retrieving is cheap, missing evidence is not.
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field

from km.llm.providers import LLM

Route = Literal["unstructured", "structured", "hybrid"]


class RouteDecision(BaseModel):
    route: Route
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str
    doc_question: str | None = Field(
        default=None, description="Rewritten sub-question for document search (unstructured/hybrid)")
    sql_question: str | None = Field(
        default=None, description="Rewritten sub-question for the SQL database (structured/hybrid)")
    companies: list[str] = Field(
        default_factory=list,
        description="Tickers (from the document catalog) of the companies the question is about; empty if none")


ROUTER_SYSTEM = """You route questions in an internal knowledge assistant.
Two sources exist:
1. STRUCTURED: a SQL database (schema below) - use for counts, totals, averages, rankings,
   trends, financial figures (revenue, net income, EPS, cash...), lists of records.
2. UNSTRUCTURED: documents (catalog below) - use for rules, policies, procedures, risks,
   strategy, explanations ("why", "how"), definitions and narrative.
Choose:
- "structured" if only the database is needed,
- "unstructured" if only documents are needed,
- "hybrid" if the answer needs both (e.g. a figure AND the explanation for it, or actual data
  compared with a policy threshold), or if you are unsure.
For hybrid, write a focused doc_question and sql_question. List the tickers of the companies the
question is about in `companies` (only tickers from the catalog). Report an honest confidence in [0,1]."""

STRUCTURED_CUES = [
    r"\bhow many\b", r"\bcount\b", r"\bnumber of\b", r"\btotal\b", r"\baverage\b", r"\bavg\b",
    r"\bsum\b", r"\btrend\b", r"\bby (region|branch|month|product|category|type|year|segment)\b",
    r"\bper (region|branch|month|year)\b", r"\btop \d+\b", r"\bhighest\b", r"\blowest\b", r"\brank",
    r"\bvolume\b", r"\binterest rates?\b", r"\bannual fees?\b", r"\bminimum balance\b", r"\bcomplaints?\b",
    r"\bresolution times?\b", r"\blist (all|the)\b", r"\bstatistic",
    r"\brevenues?\b", r"\bnet (sales|income|loss)\b", r"\bearnings per share\b", r"\beps\b",
    r"\bgross margin\b", r"\boperating (income|expenses?|margin)\b", r"\bcash and cash equivalents\b",
    r"\btotal assets\b", r"\bliabilities\b", r"\bresearch and development\b", r"\br&d\b",
    r"\bcapital expenditures?\b", r"\bdividends?\b", r"\bshare repurchases?\b",
]
UNSTRUCTURED_CUES = [
    r"\bpolic(y|ies)\b", r"\bprocedure\b", r"\bprocess\b", r"\bsteps?\b", r"\brules?\b",
    r"\bguidelines?\b", r"\bstandard\b", r"\bguide\b", r"\bsla\b", r"\bhow (do|should|can) i\b",
    r"\bwhat should\b", r"\bam i allowed\b", r"\ballowed\b", r"\brequired?\b", r"\brequirements?\b",
    r"\beligib", r"\bdeadline\b", r"\bdo i need\b", r"\bapprov", r"\breimburs", r"\bdocuments?\b",
    r"\bterms\b", r"\bcharge\b", r"\bmust\b", r"\bwhy\b", r"\bexplain", r"\breasons?\b", r"\bdrivers?\b",
    r"\brisks?\b", r"\bstrategy\b", r"\bdescribe", r"\bcompetition\b", r"\blegal proceedings\b",
]
HYBRID_CUES = [r"\bcompare\b", r"\bvs\.?\b", r"\bversus\b", r"\bwithin\b", r"\bcomply|complian",
               r"\bmeet(s|ing)?\b", r"\bagainst\b", r"\bbreach", r"\bexceed"]


def _hits(patterns: list[str], q: str) -> list[str]:
    return [p for p in patterns if re.search(p, q)]


def heuristic_route(question: str) -> RouteDecision:
    q = question.lower()
    s, u, h = _hits(STRUCTURED_CUES, q), _hits(UNSTRUCTURED_CUES, q), _hits(HYBRID_CUES, q)
    if s and u:
        conf = 0.85 if h else 0.7
        return RouteDecision(route="hybrid", confidence=conf,
                             rationale=f"data cues {len(s)} + document cues {len(u)}" + (" + comparison cue" if h else ""),
                             doc_question=question, sql_question=question)
    if s:
        return RouteDecision(route="structured", confidence=min(0.6 + 0.1 * len(s), 0.95),
                             rationale=f"{len(s)} data/aggregation cue(s)", sql_question=question)
    if u:
        return RouteDecision(route="unstructured", confidence=min(0.6 + 0.1 * len(u), 0.95),
                             rationale=f"{len(u)} policy/procedure cue(s)", doc_question=question)
    return RouteDecision(route="hybrid", confidence=0.3, rationale="no clear cues",
                         doc_question=question, sql_question=question)


class Router:
    def __init__(self, llm: LLM | None, schema_text: str, catalog: list[dict], min_confidence: float):
        self.llm = llm
        self.schema_text = schema_text
        self.has_docs = bool(catalog)
        self.catalog_text = "\n".join(
            f"- {c['title']} ({c['source']})" + (f" ticker={c['ticker']}" if c.get("ticker") else "") for c in catalog)
        # ticker -> lowercase names that identify it in a question ("apple", "meta")
        self.companies: dict[str, set[str]] = {}
        for c in catalog:
            if c.get("ticker"):
                names = self.companies.setdefault(c["ticker"].upper(), {c["ticker"].lower()})
                if c.get("company"):
                    names.add(c["company"].split()[0].strip(",.").lower())
        self.min_confidence = min_confidence

    def _companies(self, question: str, proposed: list[str]) -> list[str]:
        words = set(re.findall(r"[a-z0-9&]+", question.lower()))
        found = {t for t, names in self.companies.items() if names & words}
        for p in proposed:                       # keep the LLM's picks only if they are known tickers/names
            p_low = p.lower()
            found |= {t for t, names in self.companies.items() if p_low == t.lower() or p_low in names}
        return sorted(found)

    def route(self, question: str) -> RouteDecision:
        if self.llm is None:
            d = heuristic_route(question)
        else:
            try:
                d = self.llm.structured(
                    ROUTER_SYSTEM + f"\n\nSQL schema:\n{self.schema_text or '(none)'}\n\nDocuments:\n{self.catalog_text or '(none)'}",
                    f"Question: {question}", RouteDecision)
            except Exception as e:  # never fail closed on routing - fall back to heuristics
                d = heuristic_route(question)
                d.rationale += f" (LLM router failed: {type(e).__name__})"

        if d.confidence < self.min_confidence and d.route != "hybrid":
            d.rationale += f"; confidence {d.confidence:.2f} < {self.min_confidence} so escalated to hybrid"
            d.route = "hybrid"
        if not self.schema_text and d.route != "unstructured":
            d.route, d.rationale = "unstructured", d.rationale + "; no database loaded"
        elif not self.has_docs and d.route != "structured":
            d.route, d.rationale = "structured", d.rationale + "; no documents indexed"
        if d.route in ("unstructured", "hybrid") and not d.doc_question:
            d.doc_question = question
        if d.route in ("structured", "hybrid") and not d.sql_question:
            d.sql_question = question
        d.companies = self._companies(question, d.companies)
        return d
