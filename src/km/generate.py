"""Answer synthesis constrained to retrieved evidence: inline citations ([D1], [S1]) plus, for every
cited document passage, the exact supporting words, which the validator checks verbatim."""

from __future__ import annotations

import math

from pydantic import BaseModel, Field

from km.llm.providers import LLM
from km.text import content_tokens, split_sentences

GEN_SYSTEM = """You are an internal knowledge assistant. Answer the question using ONLY the evidence.
Rules:
- End every sentence that states a fact with its citation(s), e.g. "... within 30 days [D1]." or "[S1]".
- For EVERY document id you cite, add a quote: copy the exact supporting words from that passage
  (5-40 words, character for character, from the passage body, not from its header line).
  Database results [S...] need no quote.
- Use only numbers that appear in the evidence. Do not compute new figures or estimate; derived
  figures (growth, differences) are only allowed if a database result contains them.
- Do not use outside knowledge, even if you are confident.
- If part of the question cannot be answered from the evidence, say what is missing in a sentence
  that starts with "The evidence does not".
- Set answers_question=false only if the evidence answers no part of the question (all you could say is
  that it is missing, or offer related facts instead). If you can answer part of it, answer that part,
  keep answers_question=true and say what is missing.
- Follow the format the question asks for. For "one line per year / per company", write a bullet list
  ("- " at the start of each line), one bullet per item, each with its own citation.
- Be concise: at most 6 sentences."""


class Quote(BaseModel):
    id: str = Field(description="Evidence id, e.g. D2")
    quote: str = Field(description="Exact words copied from that evidence passage")


class GeneratedAnswer(BaseModel):
    answer: str
    quotes: list[Quote] = Field(default_factory=list)
    answers_question: bool = Field(default=True, description="False only if the evidence answers no part of the question")


def format_evidence(evidence: list[dict]) -> str:
    blocks = []
    for e in evidence:
        src = e["source"]
        where = src.get("title") or src.get("file") or "SQL database"
        if src.get("location"):
            where += f", {src['location']}"
        blocks.append(f"[{e['id']}] ({where})\n{e['text']}")
    return "\n---\n".join(blocks)


def _body(text: str) -> str:
    """Evidence text without its contextual header line."""
    return text.split("\n", 1)[1] if "\n" in text else text


def _mock_answer(question: str, evidence: list[dict]) -> GeneratedAnswer:
    """Extractive answer: best-matching evidence sentences, quoted verbatim with citations."""
    def keys(text: str) -> set[str]:   # 5-char prefixes: resolution ~ resolved
        return {t[:5] for t in content_tokens(text)}

    q = keys(question)
    candidates = []
    for e in evidence:
        if e["kind"] != "doc":
            continue
        for sent in split_sentences(_body(e["text"])):
            if " | " in sent or len(sent.split()) < 6:
                continue   # skip table rows, metadata lines and title-like fragments
            candidates.append((sent, e["id"], keys(sent)))
    # IDF-weight query terms so rare, specific matches (e.g. "SLA") beat generic ones ("policy").
    n = len(candidates) or 1
    df = {t: sum(t in c[2] for c in candidates) for t in q}
    scored = []
    for sent, eid, toks in candidates:
        score = sum(math.log(1 + n / df[t]) for t in q & toks)
        if score:
            scored.append((score, sent, eid))
    scored.sort(key=lambda t: t[0], reverse=True)
    top = scored[:4]
    parts = [f"{s.rstrip('.')} [{eid}]." for _, s, eid in top]
    quotes = [Quote(id=eid, quote=s) for _, s, eid in top]

    for e in evidence:
        if e["kind"] == "sql":
            cols, rows = e["columns"], e["rows"][:5]
            row_txt = "; ".join(", ".join(f"{c} = {v}" for c, v in zip(cols, r)) for r in rows)
            more = f" (showing 5 of {len(e['rows'])} rows)" if len(e["rows"]) > 5 else ""
            parts.append(f"The database returned: {row_txt}{more} [{e['id']}].")
    return GeneratedAnswer(answer=" ".join(parts), quotes=quotes)


class AnswerGenerator:
    def __init__(self, llm: LLM | None):
        self.llm = llm

    def generate(self, question: str, evidence: list[dict], feedback: str | None = None) -> GeneratedAnswer:
        if self.llm is None:
            return _mock_answer(question, evidence)
        user = f"Evidence:\n{format_evidence(evidence)}\n\nQuestion: {question}"
        if feedback:
            user += f"\n\nA previous draft failed fact-checking:\n{feedback}\nFix these problems."
        return self.llm.structured(GEN_SYSTEM, user, GeneratedAnswer)
