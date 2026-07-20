#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"

"$PYTHON_BIN" - <<'PY'
import sys
if sys.version_info < (3, 8, 2):
    raise SystemExit("Python >= 3.8.2 is required")
print("Using Python", sys.version)
PY

if [ ! -d .venv ]; then
  "$PYTHON_BIN" -m venv .venv
fi

. .venv/bin/activate
python -m pip install --upgrade "pip<25.1" setuptools wheel
python -m pip install --no-cache-dir -r requirements.txt
python -m pip check

echo "Installation complete. Activate with: source .venv/bin/activate"
