"""Web page: ask questions, manage the document library, and change settings, all without a terminal.

Start with start.command / start.bat (or `km ui`).
"""

from __future__ import annotations

import json
import re
import sqlite3
from contextlib import closing
from pathlib import Path

import streamlit as st

from km.config import get_settings, save_env

st.set_page_config(page_title="Knowledge Assistant", page_icon="📚", layout="wide")

UPLOAD_TYPES = ["htm", "html", "pdf", "docx", "txt", "md", "json"]
BADGE = {"High": "green", "Medium": "orange", "Low": "red"}
SEC_HELP = """
**Per company, download these from SEC EDGAR (free, no account):**
1. **The 10-K documents** (one per year, e.g. the last 3). Open the company's filing list, for example
   [Apple](https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=0000320193&type=10-K) or
   [Meta](https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=0001326801&type=10-K). Click a 10-K and then
   its main document (the `.htm`, e.g. `aapl-20240928.htm`). If the page opens in the "Inline XBRL viewer",
   use *Menu → Open as HTML*. Then save it: *File → Save Page As → "Web Page, HTML Only"*, keeping its file name.
   If this year's 10-K isn't out yet, add the latest **10-Q** (quarterly report) the same way (`type=10-Q` in the link).
2. **The financial figures** (one file per company):
   [Apple](https://data.sec.gov/api/xbrl/companyfacts/CIK0000320193.json) ·
   [Meta](https://data.sec.gov/api/xbrl/companyfacts/CIK0001326801.json). Save the file (`CIK##########.json`).

Drop all files below, click **Save to library**, then **Build index**.
"""
PROVIDERS = {"mock": "Offline demo (no key needed)", "azure_openai": "Azure OpenAI",
             "openai": "OpenAI or an internal OpenAI-compatible gateway",
             "openrouter": "OpenRouter (one key for many models)", "bedrock": "AWS Bedrock"}
FIELDS = {  # env key, label, is_secret
    "mock": [],
    "azure_openai": [
        ("AZURE_OPENAI_ENDPOINT", "Endpoint, e.g. https://my-resource.openai.azure.com/", False),
        ("AZURE_OPENAI_API_KEY", "API key (leave empty to sign in with Entra ID / managed identity)", True),
        ("AZURE_OPENAI_CHAT_DEPLOYMENT", "Chat model deployment name, e.g. gpt-4o", False),
        ("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "Embedding deployment name, e.g. text-embedding-3-small", False),
        ("AZURE_OPENAI_API_VERSION", "API version", False)],
    "openai": [
        ("OPENAI_API_KEY", "API key", True),
        ("OPENAI_BASE_URL", "Base URL (only for an internal gateway; leave empty for OpenAI)", False),
        ("OPENAI_CHAT_MODEL", "Chat model, e.g. gpt-4o", False),
        ("OPENAI_EMBEDDING_MODEL", "Embedding model, e.g. text-embedding-3-small", False)],
    "openrouter": [
        ("OPENROUTER_API_KEY", "API key (create one at https://openrouter.ai/settings/keys)", True),
        ("OPENROUTER_CHAT_MODEL", "Chat model, e.g. deepseek/deepseek-v4.1-flash-1017 or openai/gpt-4.1", False),
        ("OPENROUTER_EMBEDDING_MODEL", "Embedding model, e.g. qwen/qwen3-embedding-8b or openai/text-embedding-3-large", False)],
    "bedrock": [
        ("AWS_REGION", "AWS region (credentials come from your normal AWS login)", False),
        ("BEDROCK_CHAT_MODEL_ID", "Chat model id", False),
        ("BEDROCK_EMBEDDING_MODEL_ID", "Embedding model id", False)],
}


@st.cache_resource(show_spinner="Loading the knowledge base...")
def get_pipeline():
    from km.graph import KMPipeline

    return KMPipeline(get_settings())


def reset() -> None:
    get_settings.cache_clear()
    get_pipeline.clear()


def md(text: str) -> str:
    return text.replace("$", "\\$")   # "$391 billion ... $383" would otherwise render as math


def read_catalog(s) -> dict:
    path = s.index_path / "catalog.json"
    return json.loads(path.read_text()) if path.exists() else {}


def library_files(s) -> list[Path]:
    return sorted(p for p in s.docs_dir.iterdir() if p.is_file() and p.suffix.lstrip(".").lower() in UPLOAD_TYPES) \
        if s.docs_dir.exists() else []


def safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", Path(name).name)


# ---------------------------------------------------------------- Ask
def render_result(r: dict) -> None:
    if r["status"] == "answered":
        st.markdown(f":{BADGE[r['confidence']]}[**{r['confidence']} confidence**] · {md(r['confidence_explanation'])}")
        st.markdown(md(r["answer"]))
    else:
        st.error(md(r["answer"]))
    if r["citations"]:
        st.subheader("Sources")
        for c in r["citations"]:
            with st.container(border=True):
                st.markdown(f"**[{c['id']}] {md(c['title'] or '')}** · {md(c['location'] or '')}")
                for q in c["quotes"]:
                    st.markdown(f"> {md(q)}")
                if c["link"]:
                    st.markdown(f"[Open the source at this paragraph ↗]({c['link']})")
    if r["sql"]:
        sql_only = r["status"] == "answered" and all(c["kind"] == "sql" for c in r["citations"])
        with st.expander("Database query and result rows", expanded=sql_only):
            st.code(r["sql"], language="sql")
            if r["sql_rows"]:
                import pandas as pd

                df = pd.DataFrame(r["sql_rows"], columns=r["sql_columns"])
                config = {}
                if r["sql_row_links"]:
                    df["filing"] = r["sql_row_links"]
                    config["filing"] = st.column_config.LinkColumn("filing", display_text="open filing")
                st.dataframe(df, column_config=config, hide_index=True)
    with st.expander("How this answer was checked"):
        st.write(f"Route: **{r['route']}** ({r['route_rationale']})"
                 + (f" · companies: {', '.join(r['companies'])}" if r["companies"] else "")
                 + (f" · retried as {r['final_route']}" if r["retries"] else ""))
        st.json(r["validation"] or {}, expanded=False)
        st.caption(f"Model: {r['provider']} · {r['latency_s']} s · run id {r['run_id']} (see logs/audit.jsonl)")


def ask_tab(s) -> None:
    catalog = read_catalog(s)
    if not catalog and not s.db_path.exists():
        st.info("No documents indexed yet. Open the **Library** tab to add documents (or load the demo data).")
        return
    if s.llm_provider == "mock":
        st.warning("Offline demo mode: answers are simple extracts and database questions are limited. "
                   "Add an API key in **Settings** for full answers.")
    if catalog and (catalog.get("embedding_provider") != s.effective_embedding_provider
                    or catalog.get("embedding_model", s.embedding_id) != s.embedding_id):
        st.warning("The model settings changed since the index was built. Open **Library** and click "
                   "**Build index** before asking.")
        return
    with st.form("ask"):
        question = st.text_input("Your question", placeholder="e.g. What were Apple's total net sales in fiscal 2024?")
        submitted = st.form_submit_button("Ask", type="primary")
    if submitted and question.strip():
        with st.spinner("Searching and verifying the answer..."):
            try:
                st.session_state["last"] = get_pipeline().ask(question.strip())
            except Exception as e:
                st.session_state.pop("last", None)
                st.error(f"Something went wrong: {e}. Check the **Settings** tab (Test connection).")
    if "last" in st.session_state:
        render_result(st.session_state["last"])


# ---------------------------------------------------------------- Library
def build(s) -> None:
    from km.ingest.index import build_index
    from km.llm.providers import get_embeddings

    reset()   # release the old index before rebuilding it
    bar = st.progress(0.0, "Reading documents...")
    try:
        stats = build_index(s, get_embeddings(s),
                            on_progress=lambda d, t: bar.progress(d / t, f"Indexed {d} of {t} paragraphs"))
    except Exception as e:
        st.error(f"Building the index failed: {e}")
        return
    finally:
        reset()
    msg = f"Indexed {stats['chunks']} paragraphs from {stats['documents']} documents."
    if stats["sec"]:
        msg += f" Loaded {stats['sec']['facts']} financial facts from {len(stats['sec']['filings'])} filings."
    st.session_state["built"] = (msg, stats["sec"]["warnings"] if stats["sec"] else [])
    st.rerun()   # redraw every tab, so Ask sees the new index


def library_tab(s) -> None:
    if "built" in st.session_state:
        msg, warnings = st.session_state.pop("built")
        st.success(msg)
        for w in warnings:
            st.warning(w)
    with st.expander("How to get Apple / Meta 10-K and 10-Q filings from SEC", expanded=not library_files(s)):
        st.markdown(SEC_HELP)
    uploads = st.file_uploader("Drop files here (10-K/10-Q .htm, PDF, Word, text, SEC CIK*.json)",
                               type=UPLOAD_TYPES, accept_multiple_files=True)
    if uploads:
        replace = st.checkbox("Replace files that already exist")
        if st.button("Save to library"):
            s.docs_dir.mkdir(parents=True, exist_ok=True)
            for f in uploads:
                name = safe_name(f.name)
                dest = s.docs_dir / name
                if name.lower().endswith(".json") and not re.fullmatch(r"CIK\d{10}\.json", name):
                    st.warning(f"{name}: only SEC financial-facts files named CIK##########.json are supported.")
                elif dest.exists() and not replace:
                    st.warning(f"{name} already exists - tick 'Replace files that already exist' to overwrite it.")
                else:
                    dest.write_bytes(f.getvalue())
                    st.success(f"Saved {name}")

    files = library_files(s)
    catalog = read_catalog(s)
    st.subheader(f"Library ({len(files)} files)")
    if not files:
        st.write("The library is empty.")
        if st.button("Load demo data (4 sample bank policies + a small database)"):
            from km.sample_data import generate

            generate(s.docs_dir, s.db_path)
            build(s)
    for p in files:
        c1, c2, c3 = st.columns([6, 2, 1])
        c1.write(p.name)
        c2.caption(f"{p.stat().st_size / 1e6:.1f} MB")
        with c3.popover("Delete"):
            st.write(f"Delete {p.name} from the library?")
            if st.button("Yes, delete", key=f"del-{p.name}"):
                p.unlink()
                st.rerun()

    built = s.index_path / "catalog.json"
    stale = bool(files) and (not built.exists() or {f.name for f in files if f.suffix != ".json"}
                             != {d["source"] for d in catalog.get("documents", [])}
                             or any(f.stat().st_mtime > built.stat().st_mtime for f in files))
    if stale:
        st.info("The library changed since the index was built.")
    if files and st.button("Build index" if not built.exists() else "Rebuild index", type="primary"):
        build(s)

    if catalog.get("documents"):
        st.subheader("Indexed documents")
        st.dataframe([{"document": d["title"], "file": d["source"], "paragraphs": d["paragraphs"],
                       "ticker": d.get("ticker", ""), "fiscal year": d.get("fiscal_year", "")}
                      for d in catalog["documents"]], hide_index=True)
    if s.db_path.exists():
        with closing(sqlite3.connect(f"file:{s.db_path}?mode=ro", uri=True)) as con:
            if con.execute("SELECT 1 FROM sqlite_master WHERE name = 'filings'").fetchone():
                rows = con.execute("SELECT f.ticker, f.fiscal_year, f.document, COUNT(x.accession) "
                                   "FROM filings f LEFT JOIN facts x ON x.accession = f.accession "
                                   "GROUP BY f.accession ORDER BY f.ticker, f.fiscal_year").fetchall()
                st.subheader("Financial facts loaded")
                st.dataframe([{"ticker": t, "fiscal year": y, "document": d, "facts": n} for t, y, d, n in rows],
                             hide_index=True)


# ---------------------------------------------------------------- Settings
def settings_tab(s) -> None:
    provider = st.selectbox("Language model provider", list(PROVIDERS), format_func=PROVIDERS.get,
                            index=list(PROVIDERS).index(s.llm_provider))
    with st.form("settings"):
        values = {}
        for key, label, secret in FIELDS[provider]:
            current = str(getattr(s, key.lower()) or "")
            if secret:   # a saved key is never sent back to the browser; leaving the box empty keeps it
                values[key] = st.text_input(label, type="password",
                                            placeholder="saved - leave empty to keep it" if current else "")
            else:
                values[key] = st.text_input(label, value=current)
        no_temp = st.checkbox("The model rejects the 'temperature' setting (reasoning models such as o-series / GPT-5)",
                              value=s.llm_temperature is None) if provider != "mock" else False
        thinking = st.checkbox("Let the model think before answering (3-7x slower; models with a thinking mode only)",
                               value=s.llm_thinking) if provider == "openrouter" else s.llm_thinking
        saved = st.form_submit_button("Save", type="primary")
    st.caption("Saved on this computer only, in the .env file (never committed to git).")
    if saved:
        secrets = {key for key, _, secret in FIELDS[provider] if secret}
        try:
            save_env({"LLM_PROVIDER": provider, "LLM_TEMPERATURE": "none" if no_temp else "0",
                      "LLM_THINKING": "true" if thinking else "false",
                      **{k: v.strip() for k, v in values.items() if v.strip() or k not in secrets}})
        except ValueError as e:
            st.error(str(e))
        else:
            reset()
            st.session_state.pop("last", None)
            st.success("Saved. If you changed the provider or embedding model, rebuild the index in the Library tab.")

    if st.button("Test connection"):
        from km.llm.providers import get_embeddings, get_llm

        fresh = get_settings()
        try:
            llm = get_llm(fresh)
            if llm is None:
                st.info("Offline demo mode needs no connection.")
            else:
                from km.router import RouteDecision   # every question needs structured output; check it here

                llm.text("Reply with the single word OK.", "ping")
                llm.structured("Route this question.", "What was Apple's revenue?", RouteDecision)
                dims = len(get_embeddings(fresh).embed_query("ping"))
                st.success(f"Connected: chat model {llm.name} (with structured output) and embeddings "
                           f"({dims} dimensions) all work.")
        except Exception as e:
            st.error(f"Connection failed: {e}")


s = get_settings()
st.title("📚 Knowledge Assistant")
st.caption("Answers only from your documents and database, with the exact paragraph for every claim.")
tab_ask, tab_lib, tab_set = st.tabs(["Ask", "Library", "Settings"])
with tab_ask:
    ask_tab(s)
with tab_lib:
    library_tab(s)
with tab_set:
    settings_tab(s)
