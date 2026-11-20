"""Structured retrieval: natural language -> SQL -> guarded, read-only execution.

Defence in depth, because LLM-written SQL is untrusted input:
  1. The LLM only sees the schema (never row data) when writing SQL, plus optional
     `km_notes` (usage hints) and the best-matching `km_glossary` terms (schema linking).
  2. `guard_sql` parses the statement with sqlglot: exactly one read-only SELECT, allow-listed
     tables/views only, no dangerous functions, row LIMIT always enforced.
  3. Execution uses a read-only SQLite connection with `query_only` and a time budget.
  4. On a DB error the LLM gets one repair attempt with the error message.
"""

from __future__ import annotations

import re
import sqlite3
import time
from contextlib import closing

import sqlglot
from pydantic import BaseModel, Field
from rank_bm25 import BM25Okapi
from sqlglot import exp

from km.config import Settings
from km.llm.providers import LLM
from km.text import content_tokens

_FORBIDDEN_NODES = tuple(
    getattr(exp, n)
    for n in ("Insert", "Update", "Delete", "Drop", "Create", "Alter", "AlterTable", "Command",
              "Pragma", "Attach", "Detach", "Merge", "TruncateTable", "Transaction", "Set")
    if hasattr(exp, n)
)
_FORBIDDEN_FUNCS = {"load_extension", "readfile", "writefile", "edit", "fts3_tokenizer"}
GLOSSARY_HINTS = 25
ALL_TAGS_MAX = 1000   # a glossary this small is listed in full, so the right tag is never missing


class SQLGuardError(ValueError):
    pass


def guard_sql(sql: str, allowed_tables: set[str], max_rows: int) -> str:
    """Validate and normalise a SQL statement; raise SQLGuardError if it is not a safe SELECT."""
    try:
        statements = [s for s in sqlglot.parse(sql, read="sqlite") if s is not None]
    except sqlglot.errors.ParseError as e:
        raise SQLGuardError(f"SQL does not parse: {e}") from e
    if len(statements) != 1:
        raise SQLGuardError("Exactly one SQL statement is allowed.")
    stmt = statements[0]
    if not isinstance(stmt, exp.Query):
        raise SQLGuardError(f"Only SELECT queries are allowed (got {stmt.key.upper()}).")
    bad = stmt.find(*_FORBIDDEN_NODES)
    if bad is not None:
        raise SQLGuardError(f"Forbidden operation: {bad.key.upper()}")
    for fn in stmt.find_all(exp.Anonymous, exp.Func):
        name = (fn.name if isinstance(fn, exp.Anonymous) else fn.sql_name()).lower()
        if name in _FORBIDDEN_FUNCS:
            raise SQLGuardError(f"Forbidden function: {name}")

    cte_names = {cte.alias_or_name.lower() for cte in stmt.find_all(exp.CTE)}
    for table in stmt.find_all(exp.Table):
        name = table.name.lower()
        if name and name not in cte_names and name not in allowed_tables:
            raise SQLGuardError(f"Table not allowed: {name}")
    # Always wrap, so a query's own larger LIMIT can't bypass the cap.
    return f"SELECT * FROM ({stmt.sql(dialect='sqlite')}) AS q LIMIT {max_rows}"


class SQLDraft(BaseModel):
    can_answer: bool = Field(description="False if the schema cannot answer the question")
    sql: str = Field(default="", description="A single SQLite SELECT statement")
    explanation: str = Field(default="", description="One sentence on what the query computes")


SQL_SYSTEM = """You translate questions into a single SQLite SELECT statement for an internal
reporting database. Rules:
- Use ONLY the tables, views and columns in the schema. Never modify data.
- Follow the database notes if given. Concept hints are keyword matches, not a complete list: pick the
  exact code/tag from the full list when given, and never refuse just because a hint is missing.
- Return aggregated results where possible; ROUND averages and ratios to 2 decimals.
- If the question asks for a derived figure (growth, difference, ratio), compute it in SQL. For "how much did
  X change/increase/decrease between two periods", return ONE row with both period values AND the change as
  its own column, e.g. MAX(CASE WHEN period_fiscal_year = 2026 THEN value END)
  - MAX(CASE WHEN period_fiscal_year = 2025 THEN value END) AS change_usd. The answer may only use numbers in the result.
- Give every computed column a clear alias (e.g. avg_days_to_resolve).
- Include the columns needed to interpret and trace the answer (grouping columns, units, ids).
- The question may also ask for things a database cannot hold (reasons, explanations): write SQL for the
  part it can answer. Set can_answer=false only if no part of the question can be answered from the schema."""


def _mock_sql(question: str) -> SQLDraft:
    """Keyword-template NL->SQL for the synthetic demo data, used only in offline mock mode."""
    q = question.lower()
    rules: list[tuple[list[tuple[str, ...]], str, str]] = [
        ([("complain",), ("resol", "sla", "days", "time"), ("card",)],
         "SELECT COUNT(*) AS complaints, ROUND(AVG(c.days_to_resolve), 2) AS avg_days_to_resolve, "
         "MAX(c.days_to_resolve) AS max_days_to_resolve, "
         "SUM(CASE WHEN c.days_to_resolve > 30 THEN 1 ELSE 0 END) AS resolved_after_30_days "
         "FROM complaints c JOIN products p ON p.id = c.product_id WHERE p.type = 'credit_card'",
         "Credit card complaint resolution statistics"),
        ([("complain",), ("categor", "type", "reason")],
         "SELECT category, COUNT(*) AS complaints FROM complaints GROUP BY category ORDER BY complaints DESC",
         "Complaint counts by category"),
        ([("complain",), ("resol", "days", "time")],
         "SELECT p.type AS product_type, COUNT(*) AS complaints, ROUND(AVG(c.days_to_resolve), 2) AS "
         "avg_days_to_resolve FROM complaints c JOIN products p ON p.id = c.product_id GROUP BY p.type",
         "Average resolution days by product type"),
        ([("complain",)],
         "SELECT p.name AS product, COUNT(*) AS complaints FROM complaints c JOIN products p ON "
         "p.id = c.product_id GROUP BY p.name ORDER BY complaints DESC",
         "Complaint counts by product"),
        ([("loan", "mortgage", "lending"), ("region",)],
         "SELECT b.region, SUM(v.num_loans) AS loans, ROUND(SUM(v.amount), 2) AS total_amount FROM "
         "monthly_loan_volume v JOIN branches b ON b.id = v.branch_id GROUP BY b.region ORDER BY total_amount DESC",
         "Loan origination volume by region"),
        ([("loan", "mortgage", "lending"), ("month", "trend")],
         "SELECT month, SUM(num_loans) AS loans, ROUND(SUM(amount), 2) AS total_amount FROM "
         "monthly_loan_volume GROUP BY month ORDER BY month",
         "Monthly loan origination trend"),
        ([("loan", "mortgage", "lending"), ("branch",)],
         "SELECT b.name AS branch, SUM(v.num_loans) AS loans, ROUND(SUM(v.amount), 2) AS total_amount "
         "FROM monthly_loan_volume v JOIN branches b ON b.id = v.branch_id GROUP BY b.name ORDER BY total_amount DESC",
         "Loan origination volume by branch"),
        ([("rate", "interest", "apr")],
         "SELECT name, type, interest_rate FROM products "
         + ("WHERE type = 'credit_card' " if "card" in q else "WHERE type = 'mortgage' " if "mortgage" in q else "")
         + "ORDER BY type, interest_rate",
         "Product interest rates"),
        ([("fee",)], "SELECT name, type, annual_fee FROM products ORDER BY annual_fee DESC", "Product annual fees"),
        ([("balance",)], "SELECT name, type, min_balance FROM products WHERE min_balance > 0", "Minimum balances"),
        ([("branch",)], "SELECT region, COUNT(*) AS branches FROM branches GROUP BY region", "Branches per region"),
    ]
    for groups, sql, why in rules:
        if all(any(kw in q for kw in group) for group in groups):
            return SQLDraft(can_answer=True, sql=sql, explanation=why)
    return SQLDraft(can_answer=False, explanation="No SQL template matches this question (offline demo mode).")


def _split_camel(term: str) -> str:
    return re.sub(r"(?<=[a-z])(?=[A-Z])", " ", term)


class SQLRetriever:
    def __init__(self, s: Settings, llm: LLM | None):
        self.s = s
        self.llm = llm
        self.available = s.db_path.exists()
        self.schema: dict[str, list[tuple[str, str]]] = {}
        self.notes: list[str] = []
        self.glossary: list[tuple[str, str]] = []
        if self.available:
            self._introspect()
        self.allowed = {t.lower() for t in (s.sql_allowed_tables or self.schema.keys())}
        self.schema_text = "\n".join(
            f"{t}({', '.join(f'{c} {typ}' for c, typ in cols)})"
            for t, cols in self.schema.items() if t.lower() in self.allowed)
        self._glossary_bm25 = (BM25Okapi([content_tokens(_split_camel(t) + " " + d) for t, d in self.glossary])
                               if self.glossary else None)

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(f"file:{self.s.db_path}?mode=ro", uri=True)
        con.execute("PRAGMA query_only = ON")
        deadline = time.monotonic() + self.s.sql_timeout_s
        con.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 10_000)
        return con

    def _introspect(self) -> None:
        with closing(self._connect()) as con:
            names = [r[0] for r in con.execute(
                "SELECT name FROM sqlite_master WHERE type IN ('table', 'view') AND name NOT LIKE 'sqlite_%'")]
            for t in names:
                if t == "km_notes":
                    self.notes = [r[0] for r in con.execute("SELECT note FROM km_notes")]
                elif t == "km_glossary":
                    self.glossary = list(con.execute("SELECT term, description FROM km_glossary"))
                elif not t.startswith("km_"):   # km_* tables are hints for the LLM, not data
                    self.schema[t] = [(r[1], r[2]) for r in con.execute(f"PRAGMA table_info('{t}')")]

    def glossary_hints(self, question: str) -> list[str]:
        if not self._glossary_bm25:
            return []
        scores = self._glossary_bm25.get_scores(content_tokens(question))
        best = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:GLOSSARY_HINTS]
        return [f"{self.glossary[i][0]}: {self.glossary[i][1]}" for i in best if scores[i] > 0]

    def filing_urls(self) -> dict[str, str]:
        """accession -> SEC URL, when the SEC tables are loaded (used to link SQL rows to filings)."""
        if "filings" not in self.schema:
            return {}
        with closing(self._connect()) as con:
            return dict(con.execute("SELECT accession, url FROM filings"))

    def _draft(self, question: str, error: str | None = None, previous: str | None = None) -> SQLDraft:
        if self.llm is None:
            return _mock_sql(question)
        user = f"Schema:\n{self.schema_text}"
        if self.notes:
            user += "\n\nDatabase notes:\n" + "\n".join(f"- {n}" for n in self.notes)
        hints = self.glossary_hints(question)
        if hints:
            user += "\n\nConcept hints (best matches for this question):\n" + "\n".join(f"- {h}" for h in hints)
        if len(self.glossary) <= ALL_TAGS_MAX:
            user += "\n\nAll concept tags in the data: " + ", ".join(sorted(t for t, _ in self.glossary))
        user += f"\n\nQuestion: {question}"
        if error:
            user += f"\n\nYour previous SQL failed.\nSQL: {previous}\nError: {error}\nReturn a corrected query."
        return self.llm.structured(SQL_SYSTEM, user, SQLDraft)

    def run(self, question: str) -> dict:
        """Return {ok, sql, columns, rows, error, explanation, attempts}."""
        result = {"ok": False, "sql": None, "columns": [], "rows": [], "error": None,
                  "explanation": "", "attempts": 0}
        if not self.available:
            result["error"] = "No database is loaded."
            return result
        error, previous = None, None
        for attempt in range(2):   # initial try + one self-repair
            result["attempts"] = attempt + 1
            try:
                draft = self._draft(question, error, previous)
            except Exception as e:  # provider/network errors shouldn't crash the pipeline
                result["error"] = f"SQL generation failed: {e}"
                return result
            result["explanation"] = draft.explanation
            if not draft.can_answer or not draft.sql.strip():
                result["error"] = draft.explanation or "Question cannot be answered from the database."
                return result
            previous = draft.sql
            try:
                safe_sql = guard_sql(draft.sql, self.allowed, self.s.sql_max_rows)
                result["sql"] = safe_sql
                with closing(self._connect()) as con:
                    cur = con.execute(safe_sql)
                    result["columns"] = [d[0] for d in cur.description]
                    result["rows"] = [list(r) for r in cur.fetchall()]
                result.update(ok=True, error=None)
                return result
            except (SQLGuardError, sqlite3.Error) as e:
                error = f"{type(e).__name__}: {e}"
                result["error"] = error
                if self.llm is None:
                    break
        return result


def format_sql_evidence(result: dict) -> str:
    lines = [f"SQL: {result['sql']}", " | ".join(result["columns"])]
    # Large integers get thousands separators: a model misreads 2709000000 as "27,090,000,000" otherwise.
    lines += [" | ".join("NULL" if v is None else f"{v:,}" if isinstance(v, int) and abs(v) >= 10_000 else str(v)
                         for v in row) for row in result["rows"]]
    lines.append(f"({len(result['rows'])} rows)")
    return "\n".join(lines)
