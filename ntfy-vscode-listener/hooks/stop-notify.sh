#!/usr/bin/env bash
# Claude Code "Stop" hook — fires when the agent finishes a turn and
# pauses for human input.  Posts a push notification via ntfy.sh so
# the operator can respond from their phone.
#
# Installation:
#   1. Copy or symlink this file to ~/.claude/hooks/stop-notify.sh
#   2. chmod +x ~/.claude/hooks/stop-notify.sh
#   3. Add to ~/.claude/settings.json:
#        "hooks": {
#          "Stop": [{ "matcher": "", "hooks": [{ "type": "command",
#                      "command": "~/.claude/hooks/stop-notify.sh" }] }]
#        }
#   4. Create ~/.config/ntfy-vscode/config with the variables below.

set -euo pipefail

CONFIG_DIR="${HOME}/.config/ntfy-vscode"
CONFIG_FILE="${CONFIG_DIR}/config"

# ── Load configuration ────────────────────────────────────────────────────────
if [[ ! -f "${CONFIG_FILE}" ]]; then
  exit 0  # Not configured — silently skip
fi

# Expected config file format (key=value, one per line):
#   NTFY_SERVER=https://ntfy.sh
#   NTFY_TOPIC=my-secret-topic
#   NTFY_TOKEN=tk_xxxxxxxxxxxx   (optional)
source "${CONFIG_FILE}"

NTFY_SERVER="${NTFY_SERVER:-https://ntfy.sh}"
NTFY_TOPIC="${NTFY_TOPIC:-}"
NTFY_TOKEN="${NTFY_TOKEN:-}"

if [[ -z "${NTFY_TOPIC}" ]]; then
  exit 0
fi

# ── Parse stop reason from stdin (Claude Code provides JSON) ──────────────────
STOP_JSON=""
if read -r -t 1 line 2>/dev/null; then
  STOP_JSON="${line}"
fi

STOP_REASON="unknown"
TRANSCRIPT_PATH=""
if command -v python3 &>/dev/null && [[ -n "${STOP_JSON}" ]]; then
  STOP_REASON=$(python3 -c "
import json, sys
try:
    d = json.loads('''${STOP_JSON}''')
    print(d.get('stop_reason', 'unknown'))
except:
    print('unknown')
" 2>/dev/null || echo "unknown")
  TRANSCRIPT_PATH=$(python3 -c "
import json, sys
try:
    d = json.loads('''${STOP_JSON}''')
    print(d.get('transcript_path', ''))
except:
    print('')
" 2>/dev/null || echo "")
fi

# ── Build notification message ────────────────────────────────────────────────
HOSTNAME_SHORT=$(hostname -s 2>/dev/null || echo "VS Code")
MESSAGE="Claude Code stopped on ${HOSTNAME_SHORT} (reason: ${STOP_REASON}). Reply via ntfy to continue."

PRIORITY=3
TAGS="claude,bell"
if [[ "${STOP_REASON}" == "error" ]]; then
  PRIORITY=4
  TAGS="claude,warning"
fi

# ── Send notification ─────────────────────────────────────────────────────────
AUTH_HEADER=""
if [[ -n "${NTFY_TOKEN}" ]]; then
  AUTH_HEADER="-H \"Authorization: Bearer ${NTFY_TOKEN}\""
fi

# shellcheck disable=SC2086  (intentional word-splitting for optional header)
curl -s --max-time 10 \
  -X POST \
  -H "Title: Claude Code — Input Required" \
  -H "Priority: ${PRIORITY}" \
  -H "Tags: ${TAGS}" \
  ${AUTH_HEADER} \
  -d "${MESSAGE}" \
  "${NTFY_SERVER}/${NTFY_TOPIC}" \
  > /dev/null 2>&1 || true   # Never fail the Claude hook pipeline

exit 0
