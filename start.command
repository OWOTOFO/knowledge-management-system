#!/bin/bash
# Double-click to start (macOS). The first run installs everything into .venv (a few minutes).
cd "$(dirname "$0")" || exit 1
fail() { echo; echo "$1"; read -r -p "Press Enter to close."; exit 1; }
if [ ! -x .venv/bin/python ]; then
  python3 -c 'import sys; sys.exit(sys.version_info < (3, 10))' 2>/dev/null \
    || fail "Python 3.10 or newer is needed. Install it from https://www.python.org/downloads/ and try again."
  python3 -m venv .venv || fail "Could not create .venv - see the message above."
fi
# (Re)install on the first run, after the folder moved (the install points at this folder's code),
# or when pyproject.toml (the package list) changed.
if [ "$(cat .venv/.installed 2>/dev/null)" != "$PWD" ] || [ pyproject.toml -nt .venv/.installed ]; then
  echo "Installing (the first time takes a few minutes)..."
  .venv/bin/python -m pip install -q --upgrade pip && .venv/bin/python -m pip install -q -e . \
    || fail "Install failed - see the message above."
  echo "$PWD" > .venv/.installed
fi
# An app window left open from before holds port 8501 and keeps running the old code: stop it.
for pid in $(lsof -ti tcp:8501 -sTCP:LISTEN); do
  ps -o command= -p "$pid" | grep -q streamlit && kill "$pid" && sleep 1
done
echo "Starting the app. To stop it, close this window."
(sleep 4; open "http://localhost:8501") &
exec .venv/bin/python -m streamlit run src/km/app.py
