"""Load documents into paragraph-level `Section`s that keep provenance (file, Item/heading, page, paragraph)."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Section:
    source: str            # file name
    title: str             # document title
    text: str              # one paragraph (or one table)
    page: int | None = None
    heading: str | None = None
    para: int | None = None  # paragraph number within its heading / page
    meta: dict = field(default_factory=dict)  # e.g. ticker, company, cik, fiscal_year, form, url


def _unwrap(text: str) -> str:
    """PDF text comes back hard-wrapped; re-join lines that don't end a sentence."""
    out: list[str] = []
    for line in (ln.strip() for ln in text.splitlines()):
        if not line:
            continue
        if out and not out[-1].endswith((".", "!", "?", ":")) and not re.match(r"\d+\.\s+[A-Z]", line):
            out[-1] += " " + line
        else:
            out.append(line)
    return "\n".join(out)


def load_pdf(path: Path) -> list[Section]:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    title = (reader.metadata.title if reader.metadata and reader.metadata.title else None) or path.stem
    sections = []
    for i, page in enumerate(reader.pages, start=1):
        paragraphs = _unwrap((page.extract_text() or "").strip()).splitlines()
        sections += [Section(path.name, title, p, page=i, para=j) for j, p in enumerate(paragraphs, 1)]
    return sections


def load_docx(path: Path) -> list[Section]:
    """One Section per paragraph, numbered under its nearest heading; a table is one Section."""
    import docx
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    d = docx.Document(str(path))
    title = d.core_properties.title or path.stem
    raw: list[tuple[str | None, str]] = []
    heading = None
    for child in d.element.body.iterchildren():
        tag = child.tag.rsplit("}", 1)[-1]
        if tag == "p":
            p = Paragraph(child, d)
            text = p.text.strip()
            style = (p.style.name or "") if p.style is not None else ""
            if not text:
                continue
            if style == "Title":
                title = text
            elif style.startswith("Heading"):
                heading = text
            else:
                raw.append((heading, text))
        elif tag == "tbl":
            rows = [" | ".join(c.text.strip() for c in row.cells) for row in Table(child, d).rows]
            raw.append((heading, "\n".join(rows)))
    return _number(path.name, title, raw)


def load_text(path: Path) -> list[Section]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", path.read_text(encoding="utf-8")) if p.strip()]
    return _number(path.name, path.stem, [(None, p) for p in paragraphs])


def _number(source: str, title: str, raw: list[tuple[str | None, str]], meta: dict | None = None) -> list[Section]:
    """Number paragraphs 1..n, restarting at every new heading."""
    sections, counts = [], {}
    for heading, text in raw:
        counts[heading] = counts.get(heading, 0) + 1
        sections.append(Section(source, title, text, heading=heading, para=counts[heading], meta=dict(meta or {})))
    return sections


# --- SEC 10-K / 10-Q (inline XBRL HTML) -------------------------------------------------------------

_BLOCK_TAGS = ["p", "div", "li", "h1", "h2", "h3", "h4", "h5", "h6", "table"]
_ITEM = re.compile(r"^item\s+\d+[a-c]?\b", re.I)
_PART = re.compile(r"^part\s+(iv|i{1,3})\s*(?:[—–.:-]|$)", re.I)   # "PART I—FINANCIAL INFORMATION", not "Part I, Item 1A"
_FOOTER = re.compile(r"^(\d{1,3}|table of contents|.{0,80}\|\s*\d{1,3})$", re.I)
_SKIP_CELLS = {"", "$", "•", "●", "◦"}


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("\xa0", " ")).strip()


def _table_text(table) -> str:
    rows = []
    for tr in table.find_all("tr"):
        cells: list[str] = []
        opening = ""
        for td in tr.find_all(["td", "th"]):
            c = _clean(td.get_text(" "))   # cells stack lines ("September 28," / "2024")
            if c == "(":                            # 10-K tables split "(2)%" into "(", "2", ")%"
                opening = "("
            elif c in (")", "%", ")%") and cells:
                cells[-1] += c
            elif c not in _SKIP_CELLS:
                cells.append(opening + c)
                opening = ""
        if cells:
            rows.append(" | ".join(cells))
    return "\n".join(rows)


def read_dei(soup) -> dict:
    """Company metadata from the hidden inline-XBRL `dei:` tags every 10-K carries."""
    dei: dict[str, str] = {}
    for tag in soup.find_all(["ix:nonnumeric", "ix:nonfraction"]):
        name = tag.get("name", "")
        if name.startswith("dei:") and name[4:] not in dei:
            dei[name[4:]] = _clean(tag.get_text())
    meta: dict = {}
    if dei.get("EntityRegistrantName"):
        meta["company"] = dei["EntityRegistrantName"]
    if dei.get("TradingSymbol"):
        meta["ticker"] = dei["TradingSymbol"].upper()
    if dei.get("EntityCentralIndexKey", "").isdigit():
        meta["cik"] = int(dei["EntityCentralIndexKey"])
    if dei.get("DocumentFiscalYearFocus", "").isdigit():
        meta["fiscal_year"] = int(dei["DocumentFiscalYearFocus"])
    if dei.get("DocumentType"):
        meta["form"] = dei["DocumentType"]
    if dei.get("DocumentFiscalPeriodFocus"):
        meta["fiscal_period"] = dei["DocumentFiscalPeriodFocus"].upper()   # FY, or Q1-Q3 for a 10-Q
    return meta


def load_html(path: Path) -> list[Section]:
    """10-K / 10-Q style HTML: one Section per paragraph or table, numbered within each 'Item'."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(path.read_bytes(), "html.parser")
    meta = read_dei(soup)
    for hidden in soup.find_all(["ix:header", "script", "style"]):
        hidden.decompose()
    for hidden in soup.find_all(style=re.compile(r"display:\s*none", re.I)):
        hidden.decompose()

    if {"company", "form", "fiscal_year"} <= meta.keys():
        quarter = f" {meta['fiscal_period']}" if meta.get("fiscal_period", "FY") != "FY" else ""
        title = f"{meta['company']} {meta['form']}{quarter} FY{meta['fiscal_year']}"
    else:
        title = _clean(soup.title.get_text()) if soup.title else path.stem

    raw: list[tuple[str | None, str]] = []
    heading, part = "Cover page", None
    with_part = meta.get("form") == "10-Q"   # a 10-Q restarts at Item 1 in Part II, so the Part goes in the heading
    for el in soup.find_all(_BLOCK_TAGS):
        if el.find_parent("table") is not None:
            continue                         # table content is handled with its table
        if el.name == "table":
            text = _table_text(el)
        elif el.find(_BLOCK_TAGS) is None:   # leaf block only
            # shortcut: text sitting directly in a non-leaf block next to child blocks is dropped; rare in SEC filings.
            text = _clean(el.get_text())
        else:
            continue
        is_text = el.name != "table"
        if not text or (is_text and _FOOTER.match(text)):
            continue
        if is_text and len(text) < 100 and (m := _PART.match(text)):
            part = f"Part {m.group(1).upper()}"
            continue
        if is_text and len(text) < 200 and _ITEM.match(text):
            # the table of contents is a table, so it never sets a heading
            text = re.sub(r"^(item\s+\d+[a-c]?\.)(?=\S)", r"\1 ", text, flags=re.I)   # "Item 1.Business"
            heading = f"{part}, {text}" if with_part and part else text
            continue
        raw.append((heading, text))
    return _number(path.name, title, raw, meta)


LOADERS = {".pdf": load_pdf, ".docx": load_docx, ".txt": load_text, ".md": load_text,
           ".htm": load_html, ".html": load_html}


def load_documents(docs_dir: Path) -> list[Section]:
    sections: list[Section] = []
    for path in sorted(docs_dir.iterdir()):
        loader = LOADERS.get(path.suffix.lower())
        if loader:
            sections.extend(loader(path))
    return sections
