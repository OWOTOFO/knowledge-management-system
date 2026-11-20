"""Run the no-key bank demo in a temporary directory on any supported OS."""

import os
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    with TemporaryDirectory(prefix="km-offline-") as directory:
        scratch = Path(directory)
        env = dict(os.environ, LLM_PROVIDER="mock", EMBEDDING_PROVIDER="mock",
                   DOCS_DIR=str(scratch / "docs"), DB_PATH=str(scratch / "km.db"),
                   INDEX_DIR=str(scratch / "index"), AUDIT_LOG=str(scratch / "audit.jsonl"))
        for args in (["sample-data"], ["ingest"], ["eval", "--file", str(root / "eval" / "questions.yaml")]):
            result = subprocess.run([sys.executable, "-m", "km.cli", *args], cwd=root, env=env)
            if result.returncode:
                sys.exit(result.returncode)
