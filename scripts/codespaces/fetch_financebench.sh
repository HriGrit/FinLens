#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

TARGET_DIR="$REPO_ROOT/data/financebench"
FINANCEBENCH_REPO="${FINANCEBENCH_REPO:-https://github.com/patronus-ai/financebench}"
FINANCEBENCH_REF="${FINANCEBENCH_REF:-main}"

mkdir -p "$REPO_ROOT/data"

if [[ -d "$TARGET_DIR/.git" ]]; then
  if [[ "${FINANCEBENCH_UPDATE:-0}" == "1" ]]; then
    echo "[financebench] update requested, refreshing repository..."
    git -C "$TARGET_DIR" fetch --all --prune
    git -C "$TARGET_DIR" checkout "$FINANCEBENCH_REF"
    git -C "$TARGET_DIR" pull --ff-only origin "$FINANCEBENCH_REF"
  else
    echo "[financebench] existing checkout found, skipping update."
  fi
elif [[ -d "$TARGET_DIR" ]]; then
  echo "[financebench] directory exists without git metadata; leaving as-is."
else
  echo "[financebench] cloning ${FINANCEBENCH_REPO} into $TARGET_DIR (ref=${FINANCEBENCH_REF})"
  git clone --filter=blob:none --depth 1 --branch "$FINANCEBENCH_REF" "$FINANCEBENCH_REPO" "$TARGET_DIR"
fi

if [[ ! -d "$TARGET_DIR/pdfs" ]]; then
  echo "[financebench] ERROR: expected directory $TARGET_DIR/pdfs."
  exit 1
fi

printf '[financebench] ready at %s\n' "$TARGET_DIR"
