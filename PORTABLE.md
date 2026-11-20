# Run the assessment on another machine

Use Python 3.12 on macOS, Windows, or Linux. Extract the submission archive and open a terminal
in the folder containing `pyproject.toml`. Create a virtual environment with `python -m venv .venv`
using the Python 3.12 interpreter. Activate it with the command for your shell.

| Shell | Activation |
|---|---|
| macOS or Linux | `source .venv/bin/activate` |
| Windows PowerShell | `.\.venv\Scripts\Activate.ps1` |
| Windows Command Prompt | `.\.venv\Scripts\activate.bat` |

```text
python -m pip install -c constraints.txt -e ".[dev]"
python eval/offline_demo.py
```

The offline command builds a temporary PDF, Word and SQLite demo and checks document, SQL,
hybrid, and abstain paths. The initial dependency installation requires package index access.

## Run the SEC demo with a hosted model

The archive includes the six public filings and both SEC companyfacts JSON files in
`data/docs/`. Use a provider that your environment can reach and that supports structured JSON
Schema outputs. Create a local `.env` from `.env.example` and set these values for OpenRouter.
Use your own [OpenRouter API key](https://openrouter.ai/settings/keys) and enable billing for
the model you select. These are the models the demo was tested with; the Settings tab writes the same lines.

```dotenv
LLM_PROVIDER=openrouter
OPENROUTER_API_KEY=YOUR_OPENROUTER_API_KEY
OPENROUTER_CHAT_MODEL=deepseek/deepseek-v4.1-flash-1017
OPENROUTER_EMBEDDING_MODEL=openai/text-embedding-3-large
```

This reproduces the model configuration for the README's CLI acceptance run. To use the
configuration shown in the Streamlit screenshots, set
`OPENROUTER_EMBEDDING_MODEL=qwen/qwen3-embedding-8b` and rebuild the index.

```text
python -m km.cli ingest
python -m km.cli ask "What was Meta's total revenue in Q3 2026 alone? Use the quarterly financial facts and report the period and amount in USD." --json
python -m km.cli eval --file eval/sec_questions.yaml --report sec-live.json
```

Ingestion should load 6 filings and 2,730 facts. The quarterly revenue check should return
USD 64,601,000,000 in a SQL row. The evaluation writes its report even when cases fail and
returns exit status 1 if any case fails. Keep the API key local and share the archive without
your `.env`. On the next machine, create a fresh `.venv`, supply that machine's credentials,
and rebuild the derived data with `ingest`.
