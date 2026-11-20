"""Append-only JSONL audit trail: what was asked, how it was routed, what evidence was used, verdict."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


def write_audit(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"ts": datetime.now(timezone.utc).isoformat(), **record}
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")
