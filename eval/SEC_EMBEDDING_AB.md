# SEC filing retrieval A/B evaluation

Compare `qwen/qwen3-embedding-8b` and `openai/text-embedding-3-large` through OpenRouter. Use the same six filings and two SEC companyfacts JSON files for both models. The questions, gold evidence, and per-case rationale are in [sec_embedding_ab.yaml](sec_embedding_ab.yaml).

## Test set

| Cases | Count | Purpose |
|---|---:|---|
| D01–D08 | 8 | Document retrieval across paraphrases, adjacent fiscal years, closely related concepts, and evidence split across paragraphs |
| H01–H03 | 3 | Fixed document queries for hybrid questions covering one-time tax drivers, quarterly profit measures, and evidence from two companies |
| S01–S02 | 2 | End-to-end SQL controls for per-share units and signed cash flows |
| N01–N02 | 2 | End-to-end abstention controls when country-level figures are missing |
| L01 | 1 | Chinese-only tokenizer diagnostic with the same evidence target as D03. The YAML stores the input as Unicode escapes. |

Each case includes `why` and `expected_answer`. Structured and hybrid cases include `sql_gold` checked against SQLite. The scorer matches an evidence group when one retrieved passage has the specified filing, section prefix, and all body anchors. Paragraph numbers help locate hits. Record alternative passages that support the same conclusion as manual review notes and preserve the original automated scores.

## Stage 1 · Retrieval only

Run these commands in the project root with the virtual environment activated. They use the configured local OpenRouter key and build two indexes and SQLite databases in a temporary directory. Each scored query calls `DocRetriever.search` once with fixed input and reads the final Top 8 passages. This stage measures retrieval before chat routing, SQL generation, or validation.

```bash
ab_run_dir="$(mktemp -d)"
printf 'Reports directory: %s\n' "$ab_run_dir"

run_retrieval_arm() (
  export DOCS_DIR="$PWD/data/docs"
  export LLM_PROVIDER=mock EMBEDDING_PROVIDER=openrouter OPENROUTER_EMBEDDING_MODEL="$1"
  export INDEX_DIR="$ab_run_dir/$2/index" DB_PATH="$ab_run_dir/$2/km.db"
  export AUDIT_LOG="$ab_run_dir/$2/audit.jsonl"
  export CHUNK_SIZE=1500 CHUNK_OVERLAP=150 TOP_K=8
  export DOC_MIN_SIMILARITY=0.35 DOC_MIN_COVERAGE=0.5
  python -m km.cli ingest &&
    python eval/score_sec_embedding.py --report "$ab_run_dir/$2-retrieval.json"
)

run_retrieval_arm qwen/qwen3-embedding-8b qwen8b &&
  run_retrieval_arm openai/text-embedding-3-large openai3large
```

Before comparing results, check that the reports have identical `chunks_sha256` and `settings`, and that both have an empty `query_errors` list. Report API failures separately. Use the per-case ranks to inspect wrong fiscal years, wrong sections, and results concentrated in one company.

| Metric | Meaning |
|---|---|
| `complete_at_8` | Primary metric. Number of the 11 scored cases with every required evidence group in the Top 8 |
| `hit_at_8` | Number of cases with at least one evidence group in the Top 8 |
| `mean_group_coverage_at_8` | Average the fraction of required groups found in each case across the 11 cases |
| `mrr_at_8` | Mean reciprocal rank of the first correct evidence group. A miss scores zero. |

Report whether D02 and D03 both retrieve the correct year, D08 finds both paragraphs, and H03 covers both companies. If models tie on complete hits, inspect the missing evidence and the rank of the first correct passage. Treat 11 cases as evidence about this corpus. If L01 returns no hits, check for an early return in the English tokenizer path before attributing the result to the embedding model.

After building an index, `python eval/score_sec_embedding.py --verify-only` checks the gold anchors locally.

## Stage 2 · End-to-end spot checks

Keep both indexes and fix the chat model to `deepseek/deepseek-v4.1-flash-1017`. Use the `question` field for **D03, D08, H01, H02, H03, S02, and N02**. Run `python -m km.cli ask QUESTION --json` for each arm and save each result under its case ID. Use Stage 1 scores for the retrieval comparison.

- For D cases, check that the answer covers the question and cites the requested filing, fiscal year, and section.
- For H cases, require document citations and an actual SQL result. SQL may return the correct base rows or the correct calculated change. The answer should preserve the requested amounts and directions. H03 requires document citations for both companies.
- For S02, check that the SQL result is **-20,370,000,000 USD** for Meta FY2025 financing cash flow and that the answer calls it a net outflow.
- For N02, check that the answer states that Brazil-specific advertising revenue is unavailable in this corpus. A Rest of World total does not establish a Brazil percentage.
- Record initial and final routes, retries, actual citations, and the result. If a case fails, identify the stage where it failed.

Reuse each arm's `DB_PATH`, `INDEX_DIR`, and `OPENROUTER_EMBEDDING_MODEL`. Set `LLM_PROVIDER=openrouter`, `EMBEDDING_PROVIDER=openrouter`, and `OPENROUTER_CHAT_MODEL=deepseek/deepseek-v4.1-flash-1017`. The existing indexes are sufficient for Stage 2. Keep credentials out of reports.
