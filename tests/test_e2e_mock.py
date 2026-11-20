"""End-to-end pipeline in offline mock mode, driven by the labelled eval set."""

from pathlib import Path

import pytest
import yaml

from km.graph import FAILED, paragraph_link
from km.validate import ValidationResult

CASES = yaml.safe_load((Path(__file__).parents[1] / "eval" / "questions.yaml").read_text())


@pytest.mark.parametrize("case", CASES, ids=[c["question"][:40] for c in CASES])
def test_eval_case(pipeline, case):
    r = pipeline.ask(case["question"])
    assert r["route"] == case["expected_route"], r["route_rationale"]
    if case.get("expect_abstain"):
        assert r["status"] == "abstained"
        assert r["answer"].startswith(f"**{FAILED}**") and "Why:" in r["answer"]
        assert r["citations"] == []
    else:
        assert r["status"] == "answered", r["answer"]
        assert r["groundedness"] >= 0.8
        assert r["confidence"] in ("High", "Medium", "Low") and r["confidence_explanation"]
        assert r["citations"], "answers must cite evidence"
        for c in r["citations"]:
            if c["kind"] == "doc":
                assert c["quotes"] and "paragraph" in c["location"]
        for phrase in case.get("must_contain", []):
            assert phrase.lower() in r["answer"].lower(), r["answer"]


def test_failed_validation_retries_once_then_says_failed(pipeline, monkeypatch):
    real = pipeline.validator.validate
    calls = []

    def always_fail(answer, quotes, evidence, question=None):
        calls.append(1)
        res = real(answer, quotes, evidence, question)
        return ValidationResult(**{**res.model_dump(), "passed": False, "reasons": ["forced failure"]})

    monkeypatch.setattr(pipeline.validator, "validate", always_fail)
    r = pipeline.ask("What should I do if I lose my laptop while working remotely?")
    assert len(calls) == 2 and r["retries"] == 1 and r["final_route"] == "hybrid"
    assert r["status"] == "abstained"
    assert "could not be verified: forced failure" in r["answer"]


def test_paragraph_link_uses_text_fragment():
    link = paragraph_link("https://sec.gov/x.htm", "Total net sales increased 2% or $7.8 billion during 2024")
    assert link == "https://sec.gov/x.htm#:~:text=Total%20net%20sales%20increased%202%25%20or%20%247.8%20billion"
    assert paragraph_link(None, "anything") is None


def test_audit_log_written(pipeline, settings):
    pipeline.ask("How many complaints were logged per category?")
    lines = settings.audit_log.read_text().strip().splitlines()
    assert lines and '"route": "structured"' in lines[-1]


def test_save_env_updates_and_keeps_comments(tmp_path):
    from km.config import save_env

    env = tmp_path / ".env"
    env.write_text("# my comment\nLLM_PROVIDER=mock\nOTHER=1\n")
    save_env({"LLM_PROVIDER": "azure_openai", "AZURE_OPENAI_API_KEY": "k3y"}, env)
    assert env.read_text() == "# my comment\nLLM_PROVIDER='azure_openai'\nOTHER=1\nAZURE_OPENAI_API_KEY='k3y'\n"
    with pytest.raises(ValueError):
        save_env({"X": "a'b"}, env)


def test_an_answer_that_only_says_it_is_missing_is_a_failure(pipeline, monkeypatch):
    from km.generate import GeneratedAnswer
    real = pipeline.generator.generate

    def beside_the_point(question, evidence, feedback=None):
        g = real(question, evidence, feedback)
        return GeneratedAnswer(answer=g.answer, quotes=g.quotes, answers_question=False)

    monkeypatch.setattr(pipeline.generator, "generate", beside_the_point)
    r = pipeline.ask("What should I do if I lose my laptop while working remotely?")
    assert r["status"] == "abstained" and "does not answer the question" in r["answer"]
