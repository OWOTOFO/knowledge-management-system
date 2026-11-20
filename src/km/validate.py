"""Groundedness validation: is every claim in the answer supported by the retrieved evidence?

Deterministic checks (cannot be argued with; any failure blocks the answer):
  - every citation refers to evidence that was actually retrieved,
  - every quote is a verbatim substring of the passage it cites, and every cited passage has one,
  - every number in the answer appears in a SQL result row, or in the body of a cited paragraph that has
    a verified quote (scale-aware).
Model-judged check (best effort, reported as a confidence level):
  - each atomic claim is labelled supported / unsupported / contradicted with a 0-1 confidence and a
    one-sentence reason. With an LLM this is an LLM-as-judge call; otherwise a lexical judge.

groundedness = supported / total claims.  Passes only if groundedness >= threshold and every
deterministic check is clean.  Confidence = the weakest claim's confidence (High/Medium/Low).
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field

from km.generate import _body, format_evidence
from km.llm.providers import LLM
from km.text import (SCALE, content_tokens, extract_citations, extract_numbers, normalize_quote,
                     split_sentences, strip_citations)

Verdict = Literal["supported", "unsupported", "contradicted"]

# Filing and section labels ("Form 10-K", "Item 7", "Part II, Item 1A") are names, not figures to check.
_LABELS = re.compile(r"\b(?:Form\s+)?10-[KQ]\b|\b(?:Item|Part)\s+(?:[IVX]+|\d+[A-C]?)\b", re.I)
_ABSTENTION = re.compile(
    r"\b(the evidence does not|insufficient|no (information|evidence|data)|"
    r"not (mention|stat|specif|contain|cover|includ)\w*)\b", re.I)


class ClaimCheck(BaseModel):
    claim: str
    citations: list[str] = Field(default_factory=list)
    verdict: Verdict
    confidence: float = Field(ge=0.0, le=1.0, description="How directly the cited evidence states the claim")
    reason: str = Field(description="One sentence explaining the verdict and confidence")


class JudgeOutput(BaseModel):
    claims: list[ClaimCheck]
    answers_question: bool = Field(default=True, description="False if the answer does not give what the "
                                   "question asks for: it only says the information is missing, or offers related facts")


NOT_ANSWERED = "the evidence found does not answer the question"


class ValidationResult(BaseModel):
    passed: bool
    groundedness: float
    confidence: float
    confidence_label: Literal["High", "Medium", "Low"]
    confidence_explanation: str
    claims: list[ClaimCheck]
    verified_quotes: list[dict]
    invalid_citations: list[str]
    invalid_quotes: list[str]
    missing_quotes: list[str]
    unsupported_numbers: list[str]
    uncited_claims: list[str]
    reasons: list[str]

    def feedback(self) -> str:
        lines = list(self.reasons)
        lines += [f"- Unsupported claim: {c.claim} ({c.reason})" for c in self.claims if c.verdict != "supported"]
        lines += [f"- Quote not found verbatim: {q}" for q in self.invalid_quotes]
        return "\n".join(lines)


JUDGE_SYSTEM = """You are a strict fact-checker for an internal knowledge assistant.
Split the ANSWER into atomic factual claims. For each claim list the citation ids it uses and judge
it using ONLY the evidence with those ids:
- verdict "supported": the cited evidence states it (paraphrase is fine; numbers, years and
  company names must match), "unsupported": it does not state it or there is no citation,
  "contradicted": the cited evidence says something different.
- confidence (0-1) = how directly the evidence states the claim: about 1.0 if stated word for word
  or as an exact figure, about 0.8 for a faithful paraphrase, 0.6 or lower if it needs an inference.
- reason: one sentence explaining the verdict and confidence.
Skip sentences that only say information is missing. Do not use outside knowledge.
Finally set answers_question=false if the ANSWER does not give what the QUESTION asks for (it only says the
information is missing, or offers related facts instead). A partial answer that gives the main thing asked
for counts as answering."""


def _number_matches(value: float, decimals: int, unit: str, pool: list[float]) -> bool:
    """value matches a pool number at the answer's stated precision. '391.0 billion' matches the raw value
    391035000000 and also 391,035 from a filing table stated in millions (or 391,035,000 in thousands)."""
    tol = 0.5 * 10 ** -decimals + 1e-9
    scales = [1.0] + ([SCALE[unit], SCALE[unit] / 1e6, SCALE[unit] / 1e3] if unit in SCALE else [])
    scales = [s for s in scales if s >= 1]
    for e in pool:
        if any(abs(e / s - value) <= tol for s in scales):
            return True
        if unit == "%" and abs(e * 100 - value) <= tol:   # ratio 0.25 stated as 25%
            return True
    return False


def _quote_in(quote: str, e: dict | None) -> bool:
    """The quote appears word for word in the passage body (not its label line)."""
    nq = normalize_quote(strip_citations(quote)).rstrip(".")
    return e is not None and len(nq) >= 8 and nq in normalize_quote(_body(e["text"]))


def repair_citations(answer: str, quotes: list[dict], evidence: list[dict]) -> tuple[str, list[dict]]:
    """Point a mislabelled quote at the paragraph that really contains it.

    Models sometimes copy the right words but cite the wrong id ([D25] for D19 of 24). If the quote is found
    word for word in exactly one retrieved paragraph, and the cited id has no verified quote of its own, the
    id is corrected in the answer and the quote. The reference shown is then the paragraph the words are in."""
    by_id = {e["id"]: e for e in evidence}
    docs = [e for e in evidence if e["kind"] == "doc"]
    ok_ids = {q["id"] for q in quotes if _quote_in(q["quote"], by_id.get(q["id"]))}
    fixed = []
    for q in quotes:
        homes = [] if q["id"] in ok_ids else [e["id"] for e in docs if _quote_in(q["quote"], e)]
        if len(homes) == 1 and re.fullmatch(r"[DS]\d+", q["id"]):
            old, new = q["id"], homes[0]
            answer = _CITED.sub(lambda m: re.sub(rf"\b{old}\b", new, m.group(0)), answer)
            q = {**q, "id": new}
        fixed.append(q)
    return answer, fixed


_CITED = re.compile(r"\[[^\[\]]*\]")


def _numbers(text: str) -> list[float]:
    return [v for v, _, _ in extract_numbers(text)]


def _sql_numbers(e: dict) -> list[float]:
    """Numbers in a SQL result's rows and column names (resolved_after_30_days, q3_fy2025_value), not its query text."""
    return _numbers(re.sub(r"[A-Za-z_]+", " ", " ".join(e["columns"]))) + [n for row in e["rows"] for v in row if v is not None
                                                for n in _numbers(str(v))]


def lexical_judge(answer: str, evidence_by_id: dict[str, dict]) -> list[ClaimCheck]:
    """Offline judge: supported if >=60% of a sentence's content words occur in its cited evidence;
    a sentence citing only database results is supported when all its numbers are in those rows."""
    checks = []
    for sent in split_sentences(answer):
        if _ABSTENTION.search(sent):
            continue
        cites = extract_citations(sent)
        valid = [c for c in cites if c in evidence_by_id]
        claim = strip_citations(sent)
        if not valid:
            checks.append(ClaimCheck(claim=claim, citations=cites, verdict="unsupported", confidence=0.0,
                                     reason="The sentence has no valid citation."))
            continue
        if all(evidence_by_id[c]["kind"] == "sql" for c in valid):
            pool = [n for c in valid for n in _sql_numbers(evidence_by_id[c])]
            nums = extract_numbers(claim)
            if nums and all(_number_matches(v, d, u, pool) for v, d, u in nums):
                checks.append(ClaimCheck(claim=claim, citations=cites, verdict="supported", confidence=1.0,
                                         reason="Values taken directly from the database rows."))
                continue
        tokens = set(content_tokens(claim))
        ev_tokens = set().union(*(set(content_tokens(evidence_by_id[c]["text"])) for c in valid))
        ratio = len(tokens & ev_tokens) / max(len(tokens), 1)
        checks.append(ClaimCheck(claim=claim, citations=cites, verdict="supported" if ratio >= 0.6 else "unsupported",
                                 confidence=round(min(ratio, 1.0), 2),
                                 reason=f"{ratio:.0%} of its terms appear in the cited evidence."))
    return checks


def _explain(claims: list[ClaimCheck], n_quotes: int) -> str:
    if not claims:
        return "The answer contains no verifiable claims."
    ok = sum(c.verdict == "supported" for c in claims)
    text = f"{ok} of {len(claims)} claims verified against the cited sources"
    if n_quotes:
        text += f" ({n_quotes} exact quote{'s' if n_quotes > 1 else ''} checked)"
    weakest = min(claims, key=lambda c: c.confidence)
    if weakest.confidence >= 0.85:
        return text + "; every claim is stated directly in its source."
    return text + f'; weakest: "{weakest.claim[:90]}" ({weakest.confidence:.2f}): {weakest.reason}'


class Validator:
    def __init__(self, llm: LLM | None, threshold: float):
        self.llm = llm
        self.threshold = threshold

    def _judge(self, answer: str, evidence: list[dict], by_id: dict[str, dict],
               question: str | None) -> tuple[list[ClaimCheck], str | None, bool]:
        """(claims, error, answers_question). With a model configured, a failed judge call fails closed: never
        fall back to word overlap, which cannot tell "increased 13%" from "decreased 13%"."""
        if self.llm is None:
            return lexical_judge(answer, by_id), None, True
        asked = f"QUESTION:\n{question}\n\n" if question else ""
        try:
            out = self.llm.structured(
                JUDGE_SYSTEM, f"{asked}EVIDENCE:\n{format_evidence(evidence)}\n\nANSWER:\n{answer}", JudgeOutput)
            return out.claims, None, out.answers_question
        except Exception as e:
            return [], f"The fact-check model call failed ({type(e).__name__}: {str(e)[:200]})", True

    def validate(self, answer: str, quotes: list[dict], evidence: list[dict],
                 question: str | None = None) -> ValidationResult:
        by_id = {e["id"]: e for e in evidence}
        reasons: list[str] = []

        cited = extract_citations(answer)
        invalid = sorted({c for c in cited if c not in by_id})
        if invalid:
            reasons.append(f"Citations to evidence that was not retrieved: {', '.join(invalid)}")

        verified, bad = [], []
        for q in quotes:
            if _quote_in(q["quote"], by_id.get(q["id"])):
                verified.append(q)
            else:
                bad.append(f'{q["id"]}: "{q["quote"][:100]}"')
        if bad:
            reasons.append(f"{len(bad)} quote(s) not found word for word in the cited passage")
        quoted_ids = {q["id"] for q in verified}
        missing = sorted({c for c in cited if c in by_id and by_id[c]["kind"] == "doc" and c not in quoted_ids})
        if missing:
            reasons.append(f"No verified quote for cited passage(s): {', '.join(missing)}")

        # Numbers must come from SQL result ROWS or from the body of a cited paragraph with a verified quote
        # (references are paragraph-precise), or a cited filing's fiscal year. Never from the SQL text itself
        # (LIMIT 50, > 30 ...), from uncited passages, or from a paragraph's label line ("Item 7", "10-K").
        pool = [n for e in evidence if e["kind"] == "sql" for n in _sql_numbers(e)]
        pool += [n for i in quoted_ids & set(cited) for n in _numbers(_body(by_id[i]["text"]))]
        for c in set(cited):
            src = by_id[c]["source"] if c in by_id and by_id[c]["kind"] == "doc" else {}
            pool += _numbers(str(src.get("fiscal_year", "")))
        unsupported_numbers = []
        for sent in split_sentences(answer):
            if _ABSTENTION.search(sent):
                continue
            for value, dec, unit in extract_numbers(_LABELS.sub(" ", re.sub(r"^\d+[.)]\s+", "", sent))):
                if not _number_matches(value, dec, unit, pool):
                    unsupported_numbers.append(f"{value:g}{' ' + unit if unit else ''}")
        if unsupported_numbers:
            reasons.append(f"Numbers not found in the evidence: {', '.join(unsupported_numbers)}")

        claims, judge_error, answers_question = self._judge(answer, evidence, by_id, question)
        if not answers_question:   # true but beside the point ("X is not mentioned; Y is") is no answer
            reasons.insert(0, NOT_ANSWERED)
        # A claim that isn't supported has no support, whatever confidence the judge put on its verdict.
        claims = [c if c.verdict == "supported" else c.model_copy(update={"confidence": 0.0}) for c in claims]
        total = len(claims)
        supported = sum(c.verdict == "supported" for c in claims)
        contradicted = [c for c in claims if c.verdict == "contradicted"]
        score = supported / total if total else 0.0
        if judge_error:
            reasons.append(judge_error)
        elif total == 0:
            reasons.append("Answer contains no verifiable claims.")
        if contradicted:
            reasons.append(f'{len(contradicted)} claim(s) contradict the evidence, e.g. "{contradicted[0].claim[:100]}"')
        elif score < self.threshold:
            reasons.append(f"Only {supported} of {total} claims are supported (need {self.threshold:.0%})")

        confidence = min((c.confidence for c in claims), default=0.0)
        label = "High" if confidence >= 0.85 else "Medium" if confidence >= 0.6 else "Low"
        passed = (total > 0 and score >= self.threshold and not invalid and not bad and not missing
                  and not unsupported_numbers and not contradicted and answers_question)
        return ValidationResult(
            passed=passed, groundedness=round(score, 3), confidence=round(confidence, 2),
            confidence_label=label, confidence_explanation=_explain(claims, len(verified)), claims=claims,
            verified_quotes=verified, invalid_citations=invalid, invalid_quotes=bad, missing_quotes=missing,
            unsupported_numbers=unsupported_numbers, uncited_claims=[c.claim for c in claims if not c.citations],
            reasons=reasons)
