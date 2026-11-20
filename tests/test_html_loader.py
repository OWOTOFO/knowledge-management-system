from pathlib import Path

from km.ingest.chunker import chunk_sections
from km.ingest.loaders import load_html

FIXTURE = Path(__file__).parent / "fixtures" / "aapl-20240928.htm"


def test_10k_html_paragraphs_items_and_metadata():
    sections = load_html(FIXTURE)
    by = {(s.heading, s.para): s.text for s in sections}

    assert sections[0].meta == {"company": "Apple Inc.", "ticker": "AAPL", "cik": 320193,
                                "fiscal_year": 2024, "form": "10-K"}
    assert sections[0].title == "Apple Inc. 10-K FY2024"
    # Items come from body headings, not from the table of contents (which is a table)
    assert [h for h in dict.fromkeys(s.heading for s in sections)] == [
        "Cover page", "Item 1. Business", "Item 7. Management’s Discussion and Analysis"]
    assert by[("Item 1. Business", 1)].startswith("The Company designs")
    assert by[("Item 1. Business", 2)].startswith("The Company’s fiscal year")
    item7 = "Item 7. Management’s Discussion and Analysis"
    assert by[(item7, 1)] == "Total net sales increased 2% or $7.8 billion during 2024 compared to 2023."
    assert by[(item7, 2)] == "Net sales | 391,035 | 383,285\nChange | (2)%"      # table = one block, $ cells dropped
    assert by[(item7, 3)] == "Nested blocks keep only the innermost text block."
    texts = " ".join(s.text for s in sections)
    assert "HIDDEN HEADER TEXT" not in texts                  # ix:header dropped
    assert "2024 Form 10-K | 1" not in texts and (item7, 4) not in by   # page footers dropped


def test_chunks_are_one_paragraph_with_readable_ids():
    chunks = chunk_sections(load_html(FIXTURE), size=1500, overlap=100)
    c = next(c for c in chunks if c.id == "aapl-20240928.htm::item-7::p1")
    assert c.text.splitlines()[0] == ("Apple Inc. 10-K FY2024 (fiscal year 2024) | "
                                      "Item 7. Management’s Discussion and Analysis, paragraph 1")
    assert c.metadata["ticker"] == "AAPL" and c.metadata["para"] == 1


TEN_Q = """<html><body>
<div style="display:none"><ix:header>
<ix:nonNumeric name="dei:EntityRegistrantName">Meta Platforms, Inc.</ix:nonNumeric>
<ix:nonNumeric name="dei:TradingSymbol">META</ix:nonNumeric>
<ix:nonNumeric name="dei:EntityCentralIndexKey">0001326801</ix:nonNumeric>
<ix:nonNumeric name="dei:DocumentFiscalYearFocus">2026</ix:nonNumeric>
<ix:nonNumeric name="dei:DocumentFiscalPeriodFocus">Q3</ix:nonNumeric>
<ix:nonNumeric name="dei:DocumentType">10-Q</ix:nonNumeric>
</ix:header></div>
<div>PART I—FINANCIAL INFORMATION</div>
<div><span>Item 1.</span><span>Financial Statements</span></div>
<p>Revenue was up this quarter.</p>
<div>PART II—OTHER INFORMATION</div>
<div><span>Item 1.</span><span>Legal Proceedings</span></div>
<p>See Part I, Item 1 for the legal matters described in the notes.</p>
</body></html>"""


def test_10q_headings_carry_the_part(tmp_path):
    path = tmp_path / "meta-20260930.html"
    path.write_text(TEN_Q)
    sections = load_html(path)
    assert sections[0].title == "Meta Platforms, Inc. 10-Q Q3 FY2026"
    assert [(s.heading, s.para) for s in sections] == [
        ("Part I, Item 1. Financial Statements", 1), ("Part II, Item 1. Legal Proceedings", 1)]
    ids = [c.id for c in chunk_sections(sections, size=1500, overlap=100)]
    assert ids == ["meta-20260930.html::part-i-item-1::p1", "meta-20260930.html::part-ii-item-1::p1"]
