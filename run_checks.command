#!/bin/bash
# Double-click to test the live setup (macOS): asks every question in eval/sec_questions.yaml with the
# model settings saved in the app. If you changed the embedding model, click Rebuild index in the app first.
# Each run writes new files logs/checks-<time>.log / .json / -audit.jsonl (no API key in them).
cd "$(dirname "$0")" || exit 1
[ -x .venv/bin/python ] || { echo "Double-click start.command once first."; read -r -p "Press Enter to close."; exit 1; }
mkdir -p logs
out="logs/checks-$(date +%Y%m%d-%H%M%S)"
start=0; [ -f logs/audit.jsonl ] && start=$(wc -l < logs/audit.jsonl)
.venv/bin/python -u -m km.cli eval --file eval/sec_questions.yaml --report "$out.json" 2>&1 | tee "$out.log"
tail -n +$((start + 1)) logs/audit.jsonl > "$out-audit.jsonl" 2>/dev/null   # this run's full validation details
echo
echo "Saved: $out.log, $out.json and $out-audit.jsonl"
read -r -p "Press Enter to close."
