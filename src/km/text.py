"""Small, dependency-free text utilities shared by retrieval, routing and validation."""

from __future__ import annotations

import re

STOPWORDS = frozenset(
    """a an the and or but if of to in on for with by from at as is are was were be been being
    do does did done i we you our your my me us it its this that these those what which who whom
    how when where why can could should would will shall may might must there their they them
    than then so not no yes any all each per vs versus about into over under within between
    have has had get got also more most much many some such only own same other s while during
    after before""".split()
)

_WORD = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")
_NUMBER = re.compile(
    r"(?<![A-Za-z0-9])(\d[\d,]*(?:\.\d+)?)\s*(%|percent\b|thousand\b|million\b|billion\b|trillion\b|bn\b)?", re.I)
SCALE = {"thousand": 1e3, "million": 1e6, "billion": 1e9, "bn": 1e9, "trillion": 1e12}
_CITATION = re.compile(r"\[\s*[DS]\d+\b[^\[\]]*\]")   # [D1], [D1, S2], [D8: "quoted words"]
_CITATION_ID = re.compile(r"\b[DS]\d+\b")


def stem(word: str) -> str:
    """Very light suffix stripping, enough to match submit/submitted/submitting."""
    if len(word) > 4:
        if word.endswith("ies"):
            return word[:-3] + "y"
        for suffix in ("ing", "ed"):
            if word.endswith(suffix):
                return word[: -len(suffix)]
        if word.endswith("s") and not word.endswith("ss"):
            return word[:-1]
    return word


def words(text: str) -> list[str]:
    return _WORD.findall(text.lower().replace("'s", ""))


def content_tokens(text: str) -> list[str]:
    return [stem(w) for w in words(text) if w not in STOPWORDS and len(w) > 1]


def split_sentences(text: str) -> list[str]:
    out: list[str] = []
    for line in text.splitlines():
        line = line.strip(" -*\t")
        if not line:
            continue
        out.extend(s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\[(])", line) if s.strip())
    return out


def extract_numbers(text: str) -> list[tuple[float, int, str]]:
    """Return (value, decimals, unit) per number, ignoring citation labels like [D1].

    unit is "" or one of "%", "thousand", "million", "billion", "bn", "trillion".
    """
    text = _CITATION.sub(" ", text)
    found = []
    for m in _NUMBER.finditer(text):
        raw = m.group(1).replace(",", "").rstrip(".")
        if not raw:
            continue
        unit = (m.group(2) or "").lower()
        unit = "%" if unit == "percent" else unit
        decimals = len(raw.split(".")[1]) if "." in raw else 0
        found.append((float(raw), decimals, unit))
    return found


def normalize_quote(text: str) -> str:
    """Normalise for verbatim-quote matching: case, curly quotes, dashes, whitespace, table separators."""
    text = text.translate(str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"', "—": "-", "–": "-",
                                         "\xa0": " ", "|": " "}))
    return re.sub(r"\s+", " ", text).strip().lower()


def extract_citations(text: str) -> list[str]:
    return [i for group in _CITATION.findall(text) for i in _CITATION_ID.findall(group.split(":")[0])]


def strip_citations(text: str) -> str:
    return _CITATION.sub("", text).strip()
