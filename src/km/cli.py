"""Command line entry point (the web page is the main way in; this is for tests and power users):
km ui | sample-data | ingest | ask "<question>" | eval"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from km.config import ROOT, get_settings


def cmd_ui(_args) -> None:
    import threading
    import webbrowser

    app = Path(__file__).with_name("app.py")
    threading.Timer(4, webbrowser.open, ["http://localhost:8501"]).start()
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(app), "--server.headless", "true",
                    "--server.port", "8501", "--browser.gatherUsageStats", "false"], cwd=ROOT, check=False)


def cmd_sample_data(_args) -> None:
    from km.sample_data import generate

    s = get_settings()
    out = generate(s.docs_dir, s.db_path)
    print(f"Wrote {len(out['docs'])} documents to {s.docs_dir} and database {out['db']}")


def cmd_ingest(_args) -> None:
    from km.ingest.index import build_index
    from km.llm.providers import get_embeddings

    s = get_settings()
    stats = build_index(s, get_embeddings(s), on_progress=lambda d, t: print(f"  embedded {d}/{t}", end="\r"))
    print(f"Indexed {stats['chunks']} paragraphs from {stats['documents']} documents "
          f"(embeddings: {s.effective_embedding_provider})")
    for c in stats["catalog"]:
        print(f"  - {c['title']} ({c['source']}): {c['paragraphs']} paragraphs")
    if stats["sec"]:
        print(f"Loaded {stats['sec']['facts']} financial facts from {len(stats['sec']['filings'])} filings")
        for w in stats["sec"]["warnings"]:
            print(f"  warning: {w}")


def _print_result(r: dict, show_evidence: bool) -> None:
    print(f"\nQ: {r['question']}")
    print(f"Route: {r['route']} (confidence {r['route_confidence']:.2f}) - {r['route_rationale']}"
          + (f" | companies: {', '.join(r['companies'])}" if r["companies"] else ""))
    if r["retries"]:
        print(f"Retried with broadened retrieval -> {r['final_route']}")
    if r["sql"]:
        print(f"SQL: {r['sql']}")
    elif r["sql_error"]:
        print(f"SQL: not used ({r['sql_error']})")
    if r["status"] == "answered":
        print(f"\n[ANSWER]  confidence: {r['confidence']} - {r['confidence_explanation']}")
        print(r["answer"])
    else:
        print("\n" + r["answer"].replace("**", ""))
    if r["citations"]:
        print("\nSources:")
        for c in r["citations"]:
            print(f"  [{c['id']}] {c['title']}, {c['location']}")
            for q in c["quotes"]:
                print(f'      "{q}"')
            if c["link"]:
                print(f"      {c['link']}")
    if show_evidence:
        print("\nAll retrieved evidence:")
        for e in r["evidence"]:
            print(f"  [{e['id']}] " + e["text"].replace("\n", "\n      "))


def cmd_ask(args) -> None:
    from km.graph import KMPipeline

    result = KMPipeline().ask(args.question)
    if args.json:
        print(json.dumps(result, indent=2, default=str))
    else:
        _print_result(result, args.show_evidence)


def cmd_eval(args) -> None:
    import yaml

    from km.evaluate import answer_issues
    from km.graph import KMPipeline

    cases = yaml.safe_load(Path(args.file).read_text())
    if not isinstance(cases, list) or not cases:
        raise ValueError("The evaluation file must contain a nonempty list of cases.")
    pipe = KMPipeline()
    results = []
    route_ok = final_route_ok = status_ok = content_ok = passed = 0
    final_route_cases = sum(bool(case.get("expected_final_route")) for case in cases)
    print(f"Provider: {pipe.provider} | Embeddings: {pipe.s.effective_embedding_provider}")
    for c in cases:
        question = c["question"]
        case_id = c.get("id", question)
        expected_status = "abstained" if c.get("expect_abstain") else "answered"
        expected_route = c["expected_route"]
        expected_final = c.get("expected_final_route")
        try:
            r = pipe.ask(question)
        except Exception as exc:
            issue = f"Query failed with {type(exc).__name__}. Check the audit log."
            results.append({"id": case_id, "question": question, "status": "error", "passed": False, "issues": [issue]})
            print(f"[FAIL] {case_id} — {issue}")
            continue

        route_issues = []
        if r["route"] != expected_route:
            route_issues.append(f"Expected initial route {expected_route}, got {r['route']}.")
        final_route_issues = []
        if expected_final and r["final_route"] != expected_final:
            final_route_issues.append(f"Expected final route {expected_final}, got {r['final_route']}.")
        status_issues = []
        if r["status"] != expected_status:
            status_issues.append(f"Expected {expected_status}, got {r['status']}.")
        content_issues = answer_issues(c, r)
        issues = route_issues + final_route_issues + status_issues + content_issues
        checks = {"route": not route_issues, "status": not status_issues,
                  "content": not content_issues and not status_issues}
        if expected_final:
            checks["final_route"] = not final_route_issues
        route_ok += checks["route"]
        final_route_ok += checks.get("final_route", False)
        status_ok += checks["status"]
        content_ok += checks["content"]
        passed += not issues
        results.append({
            "id": case_id, "question": question, "passed": not issues, "checks": checks, "issues": issues,
            **{key: r[key] for key in ("route", "final_route", "status", "confidence", "answer", "citations",
                                      "sql", "sql_error", "sql_columns", "sql_rows", "sql_row_links", "latency_s")},
        })
        print(f"[{'FAIL' if issues else 'PASS'}] {case_id} ({r['route']} -> {r['final_route']}, {r['status']})")
        for issue in issues:
            print(f"  - {issue}")
        if issues:   # enough to see why, without opening the report
            print(f"    SQL: {r['sql'] or 'not used (' + str(r['sql_error']) + ')'}"[:400])
            print("    answer: " + " ".join(r["answer"].split())[:600])
    n = len(cases)
    print(f"\nRoute accuracy:      {route_ok}/{n}")
    if final_route_cases:
        print(f"Final route:         {final_route_ok}/{final_route_cases}")
    print(f"Answer/abstain:      {status_ok}/{n}")
    print(f"Expected content:    {content_ok}/{n}")
    print(f"All checks passed:   {passed}/{n}")

    if args.report:
        report = {
            "suite": Path(args.file).name, "provider": pipe.provider, "embeddings": pipe.s.effective_embedding_provider,
            "summary": {"total": n, "passed": passed, "route": route_ok, "status": status_ok, "content": content_ok,
                        "final_route": final_route_ok, "final_route_cases": final_route_cases},
            "results": results,
        }
        Path(args.report).write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    if passed != n:
        sys.exit(1)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="km", description="Hybrid knowledge-management Q&A")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("ui", help="open the web page").set_defaults(fn=cmd_ui)
    sub.add_parser("sample-data", help="generate synthetic documents + SQLite DB").set_defaults(fn=cmd_sample_data)
    sub.add_parser("ingest", help="parse, chunk, embed and index documents (+ SEC facts)").set_defaults(fn=cmd_ingest)
    a = sub.add_parser("ask", help="ask a question")
    a.add_argument("question")
    a.add_argument("--json", action="store_true", help="print the full result as JSON")
    a.add_argument("--show-evidence", action="store_true", help="print all retrieved evidence text")
    a.set_defaults(fn=cmd_ask)
    e = sub.add_parser("eval", help="run a labelled evaluation set")
    e.add_argument("--file", default=str(ROOT / "eval" / "questions.yaml"))
    e.add_argument("--report", help="save the summary and per-question results as JSON")
    e.set_defaults(fn=cmd_eval)
    args = parser.parse_args(argv)
    try:
        args.fn(args)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
