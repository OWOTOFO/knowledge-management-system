import sqlite3
from pathlib import Path

from km.config import Settings
from km.retrieval.sql import SQLRetriever
from km.sec import load_companyfacts

FACTS = Path(__file__).parent / "fixtures" / "CIK0000320193.json"
DOC24 = {"cik": 320193, "fiscal_year": 2024, "form": "10-K", "ticker": "AAPL", "company": "Apple Inc."}


def test_only_downloaded_filings_and_period_fiscal_year(tmp_path):
    db = tmp_path / "km.db"
    out = load_companyfacts(db, [FACTS], {"aapl-20240928.htm": DOC24})
    assert out["urls"]["aapl-20240928.htm"] == \
        "https://www.sec.gov/Archives/edgar/data/320193/000032019324000123/aapl-20240928.htm"
    con = sqlite3.connect(db)
    # FY2022 revenue as reported in the FY2024 10-K: XBRL says fy=2024, the period is fiscal 2022
    rows = con.execute("SELECT period_fiscal_year, value, accession FROM annual_facts "
                       "WHERE concept LIKE 'RevenueFrom%' ORDER BY period_fiscal_year").fetchall()
    assert rows == [(2022, 394328000000, "0000320193-24-000123"), (2024, 391035000000, "0000320193-24-000123")]
    # quarter (90 days) is in facts but not in annual_facts; the 10-Q and the undownloaded FY2022 10-K are not loaded
    assert con.execute("SELECT COUNT(*) FROM facts WHERE duration_days < 100").fetchone() == (1,)
    assert con.execute("SELECT COUNT(DISTINCT accession) FROM facts").fetchone() == (1,)
    assert con.execute("SELECT value FROM annual_facts WHERE concept = 'Assets'").fetchone() == (364980000000,)


def test_latest_filing_wins_and_unmatched_documents_warn(tmp_path):
    db = tmp_path / "km.db"
    docs = {"aapl-20240928.htm": DOC24, "aapl-20220924.htm": {**DOC24, "fiscal_year": 2022},
            "Apple 2019.htm": {**DOC24, "fiscal_year": 2019}}
    out = load_companyfacts(db, [FACTS], docs)
    con = sqlite3.connect(db)
    rows = con.execute("SELECT accession FROM annual_facts WHERE concept LIKE 'RevenueFrom%' "
                       "AND period_fiscal_year = 2022").fetchall()
    assert rows == [("0000320193-24-000123",)]          # reported twice, latest filing kept
    assert any("fiscal year 2019" in w for w in out["warnings"])
    assert load_companyfacts(db, [FACTS], {})["warnings"]   # no matching 10-K: skipped with a warning


def test_sql_retriever_hides_km_tables_and_links_concepts(tmp_path):
    db = tmp_path / "km.db"
    load_companyfacts(db, [FACTS], {"aapl-20240928.htm": DOC24})
    r = SQLRetriever(Settings(llm_provider="mock", db_path=db), llm=None)
    assert "annual_facts" in r.allowed and "facts" in r.allowed
    assert not any(t.startswith("km_") for t in r.allowed)
    assert any("raw dollars" in n for n in r.notes)
    assert r.glossary_hints("What was Apple's revenue?")[0].startswith("RevenueFromContractWithCustomer")
    assert r.filing_urls()["0000320193-24-000123"].endswith("aapl-20240928.htm")


def test_10q_is_matched_by_quarter_and_feeds_quarterly_facts(tmp_path):
    db = tmp_path / "km.db"
    q1 = {**DOC24, "form": "10-Q", "fiscal_period": "Q1"}
    out = load_companyfacts(db, [FACTS], {"aapl-20240928.htm": DOC24, "aapl-20231230.html": q1})
    # saved as .html by the browser; SEC's name is .htm
    assert out["urls"]["aapl-20231230.html"].endswith("/000032019324000006/aapl-20231230.htm")
    con = sqlite3.connect(db)
    assert con.execute("SELECT period_fiscal_year, fiscal_quarter, value FROM quarterly_facts").fetchall() == \
        [(2024, "Q1", 119575000000)]
    assert con.execute("SELECT COUNT(*) FROM annual_facts WHERE accession = '0000320193-24-000006'").fetchone() == (0,)
    assert ("10-Q",) in con.execute("SELECT form FROM filings").fetchall()
    missing = load_companyfacts(db, [FACTS], {"x.htm": {**q1, "fiscal_period": "Q2"}})["warnings"]
    assert "10-Q for fiscal year 2024 Q2 not found" in missing[0]
