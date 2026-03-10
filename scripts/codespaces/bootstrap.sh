#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$REPO_ROOT"

printf '\n[bootstrap] Rendering environment from .env.example\n'
"$SCRIPT_DIR/render_env.sh"

printf '\n[bootstrap] Hydrating FinanceBench dataset\n'
"$SCRIPT_DIR/fetch_financebench.sh"

PDF_DIR="$REPO_ROOT/data/financebench/pdfs"
if [[ -d "$PDF_DIR" ]]; then
  pdf_count="$(find "$PDF_DIR" -maxdepth 1 -name '*.pdf' | wc -l | tr -d ' ')"
  echo "[bootstrap] FinanceBench PDFs discovered: ${pdf_count}"
else
  echo "[bootstrap] WARNING: FinanceBench PDFs not found at $PDF_DIR"
fi

if [[ -n "${QDRANT_API_KEY:-}" ]]; then
  echo "[bootstrap] Qdrant mode: cloud (${QDRANT_URL:-http://localhost:6333})"
else
  echo "[bootstrap] Qdrant mode: local (${QDRANT_URL:-http://localhost:6333})"
fi

echo "[bootstrap] Smoke status check"
(
  cd "$REPO_ROOT/backend"
  uv run python -m ingestion.run_ingestion --list
)
