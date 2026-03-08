#!/usr/bin/env bash
# Codex notification hook — fires on agent-turn-complete
# Sends a native macOS notification via osascript (no extra tools needed)

PAYLOAD="$1"

TYPE=$(echo "$PAYLOAD" | python3 -c "import sys,json; print(json.load(sys.stdin).get('type',''))" 2>/dev/null)
MSG=$(echo "$PAYLOAD"  | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('last-assistant-message','Turn complete')[:100])" 2>/dev/null)

if [ "$TYPE" = "agent-turn-complete" ]; then
  # Escape any double quotes in the message
  SAFE_MSG="${MSG//\"/\\\"}"
  osascript -e "display notification \"$SAFE_MSG\" with title \"Codex\" sound name \"Glass\""
fi
