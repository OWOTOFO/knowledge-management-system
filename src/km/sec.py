"""Load SEC XBRL "companyfacts" JSON (downloaded by hand) into SQL tables, tied to downloaded 10-Ks/10-Qs.

Every fact kept comes from one of the 10-K / 10-Q documents in the library, so every number in the
database can be traced to a document the user has. Offline: stdlib json + sqlite3 only.
"""

from __future__ import annotations

import json
import re
import sqlite3
from contextlib import closing
from datetime import date
from pathlib import Path

DDL = """
DROP VIEW IF EXISTS annual_facts;
DROP VIEW IF EXISTS quarterly_facts;
DROP TABLE IF EXISTS facts;
DROP TABLE IF EXISTS filings;
DROP TABLE IF EXISTS km_notes;
DROP TABLE IF EXISTS km_glossary;
CREATE TABLE filings (accession TEXT PRIMARY KEY, ticker TEXT, company TEXT, form TEXT,
                      fiscal_year INTEGER, report_date TEXT, filed TEXT, document TEXT, url TEXT);
CREATE TABLE facts (ticker TEXT, company TEXT, taxonomy TEXT, concept TEXT, label TEXT, unit TEXT,
                    value NUMERIC, period_start TEXT, period_end TEXT, duration_days INTEGER,
                    period_fiscal_year INTEGER, fiscal_period TEXT, form TEXT, accession TEXT, filed TEXT);
CREATE INDEX facts_lookup ON facts (ticker, concept, period_fiscal_year);
CREATE VIEW annual_facts AS
SELECT ticker, company, concept, label, unit, value, period_start, period_end, period_fiscal_year, accession, filed
FROM (SELECT f.*, ROW_NUMBER() OVER (PARTITION BY ticker, concept, unit, period_start, period_end
                                     ORDER BY filed DESC) AS rn
      FROM facts f WHERE form = '10-K' AND (duration_days IS NULL OR duration_days BETWEEN 350 AND 380))
WHERE rn = 1;
CREATE VIEW quarterly_facts AS
SELECT ticker, company, concept, label, unit, value, period_start, period_end, period_fiscal_year,
       fiscal_period AS fiscal_quarter, accession, filed
FROM (SELECT f.*, ROW_NUMBER() OVER (PARTITION BY ticker, concept, unit, period_start, period_end
                                     ORDER BY filed DESC) AS rn
      FROM facts f WHERE form = '10-Q' AND duration_days BETWEEN 80 AND 100)
WHERE rn = 1;
CREATE TABLE km_notes (note TEXT);
CREATE TABLE km_glossary (term TEXT PRIMARY KEY, description TEXT);
"""

FORMS = ("10-K", "10-Q")

NOTES = [
    "facts holds XBRL financial data from the companies' 10-K (annual) and 10-Q (quarterly) filings. "
    "Prefer the views. annual_facts: 10-K full-year and point-in-time values, one row per period "
    "(latest filing wins, so restatements apply). quarterly_facts: 3-month periods from 10-Qs, "
    "fiscal_quarter is Q1-Q3 (there is no 10-Q for Q4).",
    "10-Q year-to-date totals and quarter-end balances are only in facts: form = '10-Q', "
    "duration_days about 180 or 270 for year-to-date, NULL for balances.",
    "value is in the unit given by the unit column. USD values are raw dollars, NOT millions; "
    "per-share values use unit USD/shares.",
    "Use period_fiscal_year for fiscal years (Apple's fiscal 2024 ended in September 2024). "
    "Do not derive years from period_end.",
    "concept is the XBRL tag. Companies tag the same metric differently (e.g. Revenues vs "
    "RevenueFromContractWithCustomerExcludingAssessedTax); use the concept hints and filter with concept IN (...).",
    "Always select ticker, concept, period_fiscal_year, value, unit and accession so every number "
    "can be traced to its filing.",
    "filings lists the loaded 10-K and 10-Q filings with their SEC URLs.",
]


def _filing_url(cik: int, accession: str, filename: str) -> str:
    folder = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession.replace('-', '')}/"
    filename = re.sub(r"\.html$", ".htm", filename)   # browsers' "Save As" often renames .htm to .html
    # shortcut: SEC document names are lowercase without spaces; anything else was renamed, so link the folder.
    return folder + filename if re.fullmatch(r"[a-z0-9._-]+\.htm", filename) else folder


def _key(form: str, fy, fp) -> tuple:
    """One 10-K per fiscal year, one 10-Q per fiscal quarter."""
    return form, fy, fp if form == "10-Q" else "FY"


def load_companyfacts(db_path: Path, json_paths: list[Path], docs: dict[str, dict]) -> dict:
    """docs: {file name: {cik, fiscal_year, fiscal_period, form, ticker, company}} from the dei tags.

    Returns {"urls": {file name: filing url}, "filings": [...], "facts": n, "warnings": [...]}.
    """
    filings, facts, glossary, warnings, urls = [], [], {}, [], {}
    for path in json_paths:
        data = json.loads(path.read_text())
        cik, company = int(data["cik"]), data.get("entityName", "")
        my_docs = {f: m for f, m in docs.items() if m.get("cik") == cik and m.get("form", "10-K") in FORMS}
        if not my_docs:
            warnings.append(f"{path.name}: no 10-K/10-Q document for CIK {cik} in the library, "
                            "so its facts were skipped.")
            continue

        rows = [(tax, concept, body, unit, it)
                for tax, concepts in data.get("facts", {}).items()
                for concept, body in concepts.items()
                for unit, items in body.get("units", {}).items()
                for it in items if it.get("form") in FORMS and it.get("fy") and it.get("accn")]
        latest: dict[tuple, tuple[str, str]] = {}     # (form, fy, fp) -> (filed, accession) of that filing
        for *_, it in rows:
            k = _key(it["form"], it["fy"], it.get("fp"))
            if (it["filed"], it["accn"]) > latest.get(k, ("", "")):
                latest[k] = (it["filed"], it["accn"])

        chosen: dict[str, tuple[str, dict, str]] = {}  # accession -> (file name, doc meta, filed)
        for fname, m in my_docs.items():
            k = _key(m.get("form", "10-K"), m.get("fiscal_year"), m.get("fiscal_period"))
            if k in latest:
                chosen[latest[k][1]] = (fname, m, latest[k][0])
            else:
                warnings.append(f"{fname}: {k[0]} for fiscal year {k[1]}"
                                f"{' ' + str(k[2]) if k[0] == '10-Q' else ''} not found in {path.name}.")

        report_end: dict[str, str] = {}               # accession -> end of the period the filing reports on
        for *_, it in rows:
            if it["accn"] in chosen and it.get("start") and _days(it) >= (80 if it["form"] == "10-Q" else 350):
                report_end[it["accn"]] = max(report_end.get(it["accn"], ""), it["end"])

        for accn, (fname, m, filed) in chosen.items():
            urls[fname] = _filing_url(cik, accn, fname)
            filings.append((accn, m.get("ticker"), company, m.get("form", "10-K"), m["fiscal_year"],
                            report_end.get(accn), filed, fname, urls[fname]))

        for tax, concept, body, unit, it in rows:
            if it["accn"] not in chosen:
                continue
            m = chosen[it["accn"]][1]
            end_year = int(it["end"][:4])
            filing_year = int((report_end.get(it["accn"]) or it["end"])[:4])
            # XBRL fy is the FILING's year.
            # shortcut: wrong for prior-year-end balances in a Q1 10-Q of a company whose fiscal year doesn't end
            # in December (Apple's Q1 ends in December); fix with the fiscal-year-end month if such 10-Qs are added.
            period_fy = m["fiscal_year"] - (filing_year - end_year)
            facts.append((m.get("ticker"), company, tax, concept, body.get("label"), unit, it["val"],
                          it.get("start"), it["end"], _days(it) if it.get("start") else None, period_fy,
                          it.get("fp"), it["form"], it["accn"], it["filed"]))
            glossary.setdefault(concept, f"{body.get('label') or ''}. {body.get('description') or ''}"[:300])

    with closing(sqlite3.connect(db_path)) as con:
        con.executescript(DDL)
        con.executemany("INSERT INTO filings VALUES (?,?,?,?,?,?,?,?,?)", filings)
        con.executemany("INSERT INTO facts VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", facts)
        con.executemany("INSERT INTO km_notes VALUES (?)", [(n,) for n in NOTES])
        con.executemany("INSERT INTO km_glossary VALUES (?,?)", glossary.items())
        con.commit()
    return {"urls": urls, "filings": [dict(zip(("accession", "ticker", "company", "form", "fiscal_year",
                                                "report_date", "filed", "document", "url"), f)) for f in filings],
            "facts": len(facts), "warnings": warnings}


def _days(it: dict) -> int:
    return (date.fromisoformat(it["end"]) - date.fromisoformat(it["start"])).days
