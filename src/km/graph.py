"""The end-to-end pipeline as an explicit LangGraph state machine.

    START -> route -> (retrieve_docs || retrieve_sql) -> evidence_gate -> generate -> validate -> finalize -> END
                         ^                                   |                          |
                         +------------- retry (max 1) <------+--------------------------+
                                                             |                          |
                                                             +--------> abstain <-------+
"""

from __future__ import annotations

import re
import time
import uuid
from typing import TypedDict
from urllib.parse import quote as urlquote

from langgraph.graph import END, START, StateGraph

from km.audit import write_audit
from km.config import Settings, get_settings
from km.generate import AnswerGenerator
from km.ingest.chunker import location
from km.llm.providers import get_embeddings, get_llm
from km.retrieval.docs import DocRetriever
from km.retrieval.sql import SQLRetriever, format_sql_evidence
from km.router import Router
from km.text import extract_citations
from km.validate import NOT_ANSWERED, ValidationResult, Validator, repair_citations

MAX_RETRIES = 1
FAILED = "Failed to find an answer."


class KMState(TypedDict, total=False):
    question: str
    decision: dict
    route: str
    doc_query: str
    sql_query: str
    k: int
    attempt: int
    doc_hits: list[dict]
    sql_result: dict
    evidence: list[dict]
    missing: list[str]
    feedback: str
    answer: str
    quotes: list[dict]
    answers_question: bool
    validation: dict
    status: str          # answered | abstained
    final_answer: str


def paragraph_link(url: str | None, quote: str) -> str | None:
    """SEC URL + text fragment (#:~:text=...) so the browser scrolls to and highlights the quote."""
    if not url:
        return None
    # shortcut: uses the first run of 4-8 plain words; curly quotes etc. may differ from the page text.
    run: list[str] = []
    for w in quote.split():
        if re.fullmatch(r"[A-Za-z0-9$%(),.;:]+", w):
            run.append(w)
            if len(run) == 8:
                break
        elif len(run) >= 4:
            break
        else:
            run = []
    if len(run) < 4:
        return url
    return f"{url}#:~:text=" + urlquote(" ".join(run).rstrip(".,;:"), safe="").replace("-", "%2D")


def _abstain_message(state: KMState) -> str:
    v = state.get("validation")
    if v and v.get("reasons"):
        why = "a draft answer could not be verified: " + "; ".join(v["reasons"])
    elif state.get("missing"):
        why = "; ".join(state["missing"])
    else:
        why = "no supporting evidence was found"
    searched = []
    companies = (state.get("decision") or {}).get("companies") or []
    if state["route"] in ("unstructured", "hybrid"):
        searched.append("the documents" + (f" of {', '.join(companies)}" if companies else ""))
    sr = state.get("sql_result") or {}
    if state["route"] in ("structured", "hybrid"):
        searched.append("the database" + (f" (query: `{sr['sql']}`)" if sr.get("sql") else ""))
    lines = [f"**{FAILED}**", f"Why: {why}.", f"Searched: {' and '.join(searched) or 'nothing'}."]
    docs = [e for e in state.get("evidence") or [] if e["kind"] == "doc"][:3]
    if docs:
        lines.append("Closest passages found (not enough on their own):")
        lines += [f"- [{e['id']}] {e['source']['title']}, {e['source']['location']}: "
                  f"{e['text'].split(chr(10), 1)[-1][:160]}..." for e in docs]
    return "\n\n".join(lines)


def _dispatch(state: KMState) -> list[str]:
    return {"unstructured": ["retrieve_docs"], "structured": ["retrieve_sql"],
            "hybrid": ["retrieve_docs", "retrieve_sql"]}[state["route"]]


def build_graph(p: "KMPipeline"):
    def route(state: KMState) -> KMState:
        d = p.router.route(state["question"])
        return {"decision": d.model_dump(), "route": d.route, "doc_query": d.doc_question or state["question"],
                "sql_query": d.sql_question or state["question"], "k": p.s.top_k, "attempt": 0,
                "doc_hits": [], "sql_result": {}}

    def retrieve_docs(state: KMState) -> KMState:
        tickers = state["decision"].get("companies") or None
        return {"doc_hits": p.docs.search(state["doc_query"], state["k"], tickers=tickers)}

    def retrieve_sql(state: KMState) -> KMState:
        return {"sql_result": p.sql.run(state["sql_query"])}

    def evidence_gate(state: KMState) -> KMState:
        evidence, missing = [], []
        for i, h in enumerate(state.get("doc_hits") or [], start=1):
            m = h["metadata"]
            evidence.append({"id": f"D{i}", "kind": "doc", "text": h["text"],
                             "source": {"file": m["source"], "title": m.get("title", m["source"]),
                                        "location": location(m), "heading": m.get("heading"), "page": m.get("page"),
                                        "para": m.get("para"), "url": m.get("url"), "ticker": m.get("ticker"),
                                        "fiscal_year": m.get("fiscal_year"), "chunk_id": h["chunk_id"],
                                        "dense_similarity": h["dense_similarity"],
                                        "term_coverage": h["term_coverage"]}})
        sr = state.get("sql_result") or {}
        if sr.get("ok") and sr.get("rows"):
            evidence.append({"id": "S1", "kind": "sql", "text": format_sql_evidence(sr),
                             "columns": sr["columns"], "rows": sr["rows"],
                             "source": {"title": "Database", "location": "query result", "sql": sr["sql"],
                                        "row_count": len(sr["rows"])}})
        if state["route"] in ("unstructured", "hybrid") and not state.get("doc_hits"):
            companies = state["decision"].get("companies")
            missing.append("no document paragraph" + (f" of {', '.join(companies)}" if companies else "")
                           + " matched the question closely enough")
        if state["route"] in ("structured", "hybrid") and not (sr.get("ok") and sr.get("rows")):
            missing.append("the database query returned no rows" if sr.get("ok")
                           else f"the database could not answer it ({sr.get('error') or 'no query'})")
        return {"evidence": evidence, "missing": missing}

    def after_gate(state: KMState) -> str:
        if state["evidence"]:
            return "generate"
        return "retry" if state["attempt"] < MAX_RETRIES else "abstain"

    def generate(state: KMState) -> KMState:
        g = p.generator.generate(state["question"], state["evidence"], state.get("feedback"))
        answer, quotes = repair_citations(g.answer, [q.model_dump() for q in g.quotes], state["evidence"])
        return {"answer": answer, "quotes": quotes, "answers_question": g.answers_question}

    def validate(state: KMState) -> KMState:
        v = p.validator.validate(state["answer"], state.get("quotes") or [], state["evidence"], state["question"])
        if not state.get("answers_question", True) and NOT_ANSWERED not in v.reasons:   # the writer said so too
            v.passed = False
            v.reasons.insert(0, NOT_ANSWERED)
        return {"validation": v.model_dump()}

    def after_validate(state: KMState) -> str:
        if state["validation"]["passed"]:
            return "finalize"
        return "retry" if state["attempt"] < MAX_RETRIES else "abstain"

    def retry(state: KMState) -> KMState:
        """Broaden: both stores, twice the depth, documents searched with the full question; the company
        filter and the router's focused SQL sub-question are kept (a full mixed question makes the SQL
        writer refuse the parts a database can't hold)."""
        feedback = ValidationResult(**state["validation"]).feedback() if state.get("validation") else None
        return {"attempt": state["attempt"] + 1, "route": "hybrid", "k": state["k"] * 2,
                "doc_query": state["question"],
                "sql_query": state["decision"].get("sql_question") or state["question"], "feedback": feedback}

    def finalize(state: KMState) -> KMState:
        return {"status": "answered", "final_answer": state["answer"]}

    def abstain(state: KMState) -> KMState:
        return {"status": "abstained", "final_answer": _abstain_message(state)}

    g = StateGraph(KMState)
    for name, fn in [("route", route), ("retrieve_docs", retrieve_docs), ("retrieve_sql", retrieve_sql),
                     ("evidence_gate", evidence_gate), ("generate", generate), ("validate", validate),
                     ("retry", retry), ("finalize", finalize), ("abstain", abstain)]:
        g.add_node(name, fn)
    g.add_edge(START, "route")
    g.add_conditional_edges("route", _dispatch, ["retrieve_docs", "retrieve_sql"])
    g.add_conditional_edges("retry", _dispatch, ["retrieve_docs", "retrieve_sql"])
    g.add_edge("retrieve_docs", "evidence_gate")   # separate edges: gate fires once per superstep
    g.add_edge("retrieve_sql", "evidence_gate")
    g.add_conditional_edges("evidence_gate", after_gate, ["generate", "retry", "abstain"])
    g.add_edge("generate", "validate")
    g.add_conditional_edges("validate", after_validate, ["finalize", "retry", "abstain"])
    g.add_edge("finalize", END)
    g.add_edge("abstain", END)
    return g.compile()


class KMPipeline:
    def __init__(self, settings: Settings | None = None):
        self.s = settings or get_settings()
        self.llm = get_llm(self.s)
        self.docs = DocRetriever(self.s, get_embeddings(self.s))
        self.sql = SQLRetriever(self.s, self.llm)
        self.router = Router(self.llm, self.sql.schema_text, self.docs.catalog, self.s.router_min_confidence)
        self.generator = AnswerGenerator(self.llm)
        self.validator = Validator(self.llm, self.s.groundedness_threshold)
        self.graph = build_graph(self)

    @property
    def provider(self) -> str:
        return self.llm.name if self.llm else "mock"

    def ask(self, question: str) -> dict:
        run_id, t0 = uuid.uuid4().hex[:12], time.perf_counter()
        try:
            st = self.graph.invoke({"question": question})
        except Exception as e:   # still leave an audit trail, then let the caller show the error
            write_audit(self.s.audit_log, {"run_id": run_id, "question": question, "status": "error",
                                           "error": f"{type(e).__name__}: {e}", "provider": self.provider})
            raise
        v = st.get("validation") or {}
        sr = st.get("sql_result") or {}
        answered = st["status"] == "answered"
        cited = set(extract_citations(st["answer"])) if answered else set()
        verified = v.get("verified_quotes") or []
        citations = []
        for e in st.get("evidence") or []:
            if e["id"] not in cited:
                continue
            quotes = [q["quote"] for q in verified if q["id"] == e["id"]]
            src = e["source"]
            citations.append({"id": e["id"], "kind": e["kind"], "title": src.get("title"),
                              "location": src.get("location"), "url": src.get("url"), "quotes": quotes,
                              "link": paragraph_link(src.get("url"), quotes[0]) if quotes else src.get("url"),
                              "sql": src.get("sql")})
        row_links = []
        if "accession" in sr.get("columns", []):
            urls, col = self.sql.filing_urls(), sr["columns"].index("accession")
            row_links = [urls.get(r[col]) for r in sr["rows"]]
        result = {
            "run_id": run_id,
            "question": question,
            "status": st["status"],
            "answer": st["final_answer"],
            "confidence": v.get("confidence_label") if answered else None,
            "confidence_explanation": v.get("confidence_explanation") if answered else None,
            "citations": citations,
            "route": st["decision"]["route"],
            "final_route": st["route"],
            "companies": st["decision"].get("companies", []),
            "route_confidence": st["decision"]["confidence"],
            "route_rationale": st["decision"]["rationale"],
            "retries": st["attempt"],
            "groundedness": v.get("groundedness"),
            "validation": v,
            "sql": sr.get("sql"),
            "sql_error": sr.get("error"),
            "sql_columns": sr.get("columns", []),
            "sql_rows": sr.get("rows", []),
            "sql_row_links": row_links,
            "evidence": [{"id": e["id"], "kind": e["kind"], "source": e["source"], "text": e["text"]}
                         for e in st.get("evidence") or []],
            "provider": self.provider,
            "latency_s": round(time.perf_counter() - t0, 3),
        }
        write_audit(self.s.audit_log, {k: val for k, val in result.items() if k not in ("evidence", "sql_rows")}
                    | {"evidence_refs": [{"id": e["id"], **e["source"]} for e in result["evidence"]]}
                    | ({} if answered else {"rejected_draft": st.get("answer")}))
        return result
