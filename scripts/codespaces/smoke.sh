#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$REPO_ROOT/backend"

echo "[smoke] Running quick code smoke checks..."
if [[ -f "tests/unit/test_qdrant_shared.py" ]]; then
  uv run pytest tests/unit/test_qdrant_shared.py tests/unit/test_run_ingestion.py tests/integration/test_api.py -m integration -q
else
  uv run pytest tests/unit/test_run_ingestion.py tests/integration/test_api.py -q
fi

echo "[smoke] Listing ingestion state..."
uv run python -m ingestion.run_ingestion --list

echo "[smoke] Optional: start API locally with `uv run uvicorn api.main:app --host 0.0.0.0 --port 8000`"
