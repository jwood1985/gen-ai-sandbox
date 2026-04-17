#!/usr/bin/env bash
# Codespaces keepalive via ntfy.sh SSE subscription.
#
# Two keepalive mechanisms run in parallel:
#   1. The ntfy SSE connection itself — a live HTTP stream that counts as
#      network activity and resets the Codespace inactivity timer.
#   2. A periodic heartbeat that writes a timestamp to /tmp, ensuring the
#      Codespace stays alive even if ntfy is unreachable.
#
# Configuration (set as Codespaces secrets in repo/org settings):
#   NTFY_SERVER          ntfy server base URL  (default: https://ntfy.sh)
#   NTFY_TOPIC           topic name            (required — skip ntfy if unset)
#   NTFY_TOKEN           bearer token          (optional — for private topics)
#   NTFY_KEYWORD         command keyword       (optional — enables terminal injection)
#   KEEPALIVE_INTERVAL_SECS  heartbeat period  (default: 240)

set -euo pipefail

NTFY_SERVER="${NTFY_SERVER:-https://ntfy.sh}"
NTFY_TOPIC="${NTFY_TOPIC:-}"
NTFY_TOKEN="${NTFY_TOKEN:-}"
NTFY_KEYWORD="${NTFY_KEYWORD:-}"
INTERVAL="${KEEPALIVE_INTERVAL_SECS:-240}"
LOG="/tmp/ntfy-keepalive.log"
HEARTBEAT="/tmp/.codespace-keepalive"

log() { echo "[$(date -u +%FT%TZ)] $*" | tee -a "$LOG"; }

# ── Heartbeat loop ────────────────────────────────────────────────────────────
heartbeat() {
  while true; do
    date -u +%FT%TZ > "$HEARTBEAT"
    sleep "$INTERVAL"
  done
}

# ── ntfy SSE subscriber ───────────────────────────────────────────────────────
subscribe() {
  [[ -z "$NTFY_TOPIC" ]] && { log "NTFY_TOPIC not set — running heartbeat only."; return; }

  local url="${NTFY_SERVER}/${NTFY_TOPIC}/sse?since=now"
  local auth_header=""
  [[ -n "$NTFY_TOKEN" ]] && auth_header="Authorization: Bearer ${NTFY_TOKEN}"

  log "Subscribing to ${NTFY_SERVER}/${NTFY_TOPIC}"

  while true; do
    curl -s --no-buffer --max-time 600 \
      ${auth_header:+-H "$auth_header"} \
      -H "Accept: text/event-stream" \
      "$url" \
    | while IFS= read -r line; do
        # Only process data lines that carry a JSON payload
        [[ "$line" != data:* ]] && continue
        local json="${line#data:}"
        local event msg
        event=$(echo "$json" | python3 -c "import json,sys; print(json.loads(sys.stdin.read()).get('event',''))" 2>/dev/null || true)
        [[ "$event" != "message" ]] && continue

        msg=$(echo "$json" | python3 -c "import json,sys; print(json.loads(sys.stdin.read()).get('message',''))" 2>/dev/null || true)
        [[ -z "$msg" ]] && continue

        # Touch heartbeat — any ntfy message extends the keepalive
        date -u +%FT%TZ > "$HEARTBEAT"
        log "Received: $msg"

        # Keyword-gated terminal injection
        if [[ -n "$NTFY_KEYWORD" && "$msg" == "${NTFY_KEYWORD}:"* ]]; then
          local cmd="${msg#${NTFY_KEYWORD}:}"
          cmd="${cmd# }"  # strip leading space

          # Sanitize: reject shell metacharacters
          if echo "$cmd" | grep -qE '[;&|`$\\<>]'; then
            log "BLOCKED (metacharacters): $cmd"
            continue
          fi
          # Reject if too long
          if [[ ${#cmd} -gt 256 ]]; then
            log "BLOCKED (too long): ${cmd:0:40}..."
            continue
          fi

          log "Injecting into terminal: $cmd"
          # Write to a FIFO that a terminal session can tail, or use
          # the VS Code CLI if available
          if command -v code &>/dev/null; then
            # Send text to the active VS Code terminal via the helper command
            # (requires the ntfy-vscode-listener extension installed in Codespaces)
            echo "$cmd" >> /tmp/ntfy-inject.fifo 2>/dev/null || true
          fi
        fi
      done

    log "SSE stream ended — reconnecting in 5s…"
    sleep 5
  done
}

# ── Main ──────────────────────────────────────────────────────────────────────
log "Codespaces keepalive started (interval=${INTERVAL}s)"

heartbeat &
HEARTBEAT_PID=$!

subscribe &
SUBSCRIBE_PID=$!

# Publish startup notification to phone if configured
if [[ -n "$NTFY_TOPIC" ]]; then
  auth_header=""
  [[ -n "$NTFY_TOKEN" ]] && auth_header="Authorization: Bearer ${NTFY_TOKEN}"
  CODESPACE_LABEL="${CODESPACE_NAME:-codespace}"
  curl -s --max-time 10 \
    ${auth_header:+-H "$auth_header"} \
    -H "Title: Codespace Started" \
    -H "Priority: 2" \
    -H "Tags: rocket,vscode" \
    -d "Codespace '${CODESPACE_LABEL}' is live and keepalive is running." \
    "${NTFY_SERVER}/${NTFY_TOPIC}" > /dev/null 2>&1 || true
fi

# Clean up background jobs on exit
trap 'kill $HEARTBEAT_PID $SUBSCRIBE_PID 2>/dev/null; log "Keepalive stopped."' EXIT TERM INT

wait
