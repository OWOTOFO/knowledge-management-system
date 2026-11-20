"""Smoke test of the web page, driven headlessly: empty library -> demo data -> ask -> failure box."""

from pathlib import Path

from streamlit.testing.v1 import AppTest

from km.config import get_settings

APP = str(Path(__file__).parents[1] / "src" / "km" / "app.py")


def test_web_page_end_to_end(tmp_path, monkeypatch):
    for key, value in {"LLM_PROVIDER": "mock", "EMBEDDING_PROVIDER": "mock", "DOCS_DIR": tmp_path / "docs",
                       "DB_PATH": tmp_path / "km.db", "INDEX_DIR": tmp_path / "index",
                       "AUDIT_LOG": tmp_path / "audit.jsonl"}.items():
        monkeypatch.setenv(key, str(value))
    get_settings.cache_clear()

    at = AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception
    assert "No documents indexed yet" in at.info[0].value

    at = next(b for b in at.button if b.label.startswith("Load demo data")).click().run()
    assert not at.exception and "Indexed" in at.success[0].value

    at.text_input[0].input("What should I do if I lose my laptop while working remotely?")
    at = at.button[0].click().run()
    assert not at.exception and not at.error
    assert any("confidence**]" in m.value for m in at.markdown)
    assert any(m.value.startswith("**[D") and "paragraph" in m.value for m in at.markdown)

    at.text_input[0].input("What is the bank's policy on cryptocurrency custody for retail clients?")
    at = at.button[0].click().run()
    assert at.error[0].value.startswith("**Failed to find an answer.**")
    get_settings.cache_clear()


def test_settings_page_never_shows_a_saved_key(tmp_path, monkeypatch):
    for key, value in {"LLM_PROVIDER": "openrouter", "OPENROUTER_API_KEY": "sk-or-v1-SECRET", "DOCS_DIR": tmp_path / "docs",
                       "DB_PATH": tmp_path / "km.db", "INDEX_DIR": tmp_path / "index"}.items():
        monkeypatch.setenv(key, str(value))
    get_settings.cache_clear()
    at = AppTest.from_file(APP, default_timeout=60).run()
    key_box = next(t for t in at.text_input if t.label.startswith("API key"))
    assert key_box.value == "" and "saved" in key_box.placeholder
    assert not any("SECRET" in str(t.value) for t in at.text_input)
    get_settings.cache_clear()
