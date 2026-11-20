"""Run the local KM CLI with a signed-in Codex CLI for chat and mock embeddings."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from langchain_core.utils.function_calling import convert_to_openai_function

from km.cli import main

MAC_CODEX = Path("/Applications/ChatGPT.app/Contents/Resources/codex-cli/CodexCLI.app/Contents/MacOS/codex")


def remove_defaults(node) -> None:
    if isinstance(node, dict):
        node.pop("default", None)
        for value in node.values():
            remove_defaults(value)
    elif isinstance(node, list):
        for value in node:
            remove_defaults(value)


class CodexCli:
    name = "codex_cli"

    def __init__(self) -> None:
        executable = shutil.which("codex") or (str(MAC_CODEX) if MAC_CODEX.is_file() else None)
        if executable is None:
            raise RuntimeError("Codex CLI not found. Install it or add it to PATH.")
        self.executable = executable
        self.env = {key: value for key, value in os.environ.items() if key not in {"CODEX_API_KEY", "OPENAI_API_KEY"}}
        login = subprocess.run([executable, "login", "status"], env=self.env, capture_output=True, text=True,
                               timeout=15)
        if login.returncode or "Logged in using ChatGPT" not in login.stdout + login.stderr:
            raise RuntimeError("Sign in to Codex CLI with ChatGPT first, then run this script again.")

    def structured(self, system: str, user: str, schema):
        output_schema = convert_to_openai_function(schema, strict=True)["parameters"]
        remove_defaults(output_schema)
        with TemporaryDirectory(prefix="km-codex-") as directory:
            folder = Path(directory)
            schema_file = folder / "schema.json"
            answer_file = folder / "answer.json"
            schema_file.write_text(json.dumps(output_schema), encoding="utf-8")
            command = [
                self.executable, "-a", "never", "exec", "--ignore-user-config", "--ignore-rules", "--ephemeral",
                "--skip-git-repo-check", "--sandbox", "read-only", "--disable", "shell_tool",
                "--disable", "unified_exec", "-c", 'web_search="disabled"',
                "--output-schema", str(schema_file), "--output-last-message", str(answer_file), "-",
            ]
            prompt = f"{system}\n\nUse only the input below. Do not use tools. Return the requested JSON.\n\n{user}"
            result = subprocess.run(command, input=prompt, cwd=folder, env=self.env, capture_output=True, text=True,
                                    timeout=120)
            if result.returncode or not answer_file.exists():
                errors = [line.strip() for line in result.stderr.splitlines() if line.strip().startswith("ERROR:")]
                detail = f" {errors[-1]}" if errors else ""
                raise RuntimeError(f"Codex CLI did not return structured output (exit {result.returncode}).{detail}")
            return schema.model_validate_json(answer_file.read_text(encoding="utf-8"))


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in {"ask", "eval"}:
        sys.exit("Usage: python eval/run_with_codex.py ask QUESTION [--json] | eval [--file PATH] [--report PATH]")
    os.environ["EMBEDDING_PROVIDER"] = "mock"
    try:
        provider = CodexCli()
    except (OSError, RuntimeError, subprocess.TimeoutExpired) as exc:
        sys.exit(str(exc))
    with patch("km.graph.get_llm", return_value=provider):
        main(sys.argv[1:])
