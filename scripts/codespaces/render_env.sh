#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

EXAMPLE_FILE="$REPO_ROOT/.env.example"
ENV_FILE="$REPO_ROOT/.env"

if [[ ! -f "$EXAMPLE_FILE" ]]; then
  echo "[render_env] ERROR: missing .env.example"
  exit 1
fi

cp "$EXAMPLE_FILE" "$ENV_FILE"

set_value() {
  local key="$1"
  local value="${2:-}"
  if [[ -z "$value" ]]; then
    return
  fi
  if grep -qE "^${key}=" "$ENV_FILE"; then
    if sed --version >/dev/null 2>&1; then
      sed -i "s#^${key}=.*#${key}=${value}#" "$ENV_FILE"
    else
      sed -i '' "s#^${key}=.*#${key}=${value}#" "$ENV_FILE"
    fi
  else
    printf '\n%s=%s\n' "$key" "$value" >> "$ENV_FILE"
  fi
}

set_value "QDRANT_URL" "${QDRANT_URL:-}"
set_value "QDRANT_API_KEY" "${QDRANT_API_KEY:-}"
set_value "QDRANT_COLLECTION" "${QDRANT_COLLECTION:-}"
set_value "OPENROUTER_API_KEY" "${OPENROUTER_API_KEY:-}"
set_value "LANGFUSE_PUBLIC_KEY" "${LANGFUSE_PUBLIC_KEY:-}"
set_value "LANGFUSE_SECRET_KEY" "${LANGFUSE_SECRET_KEY:-}"
set_value "LANGFUSE_HOST" "${LANGFUSE_HOST:-}"

echo "[render_env] .env refreshed from .env.example and environment overrides."
