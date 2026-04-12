#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

: "${LITELLM_URL:=http://localhost:4000}"
: "${PROXY_HOST:=0.0.0.0}"
: "${PROXY_PORT:=8787}"
: "${PROXY_LOG_LEVEL:=INFO}"
: "${PROXY_LOG_FILE:=llm_proxy.log}"
: "${PROXY_BUFFER_TOOLS:=true}"
: "${PROXY_VALIDATE_JSON:=true}"
: "${PROXY_STREAM_TIMEOUT:=120}"
: "${PROXY_STREAM_MAX_DURATION:=600}"
: "${PROXY_MAX_UPSTREAM_CONNECTIONS:=100}"
: "${PROXY_REQUEST_TIMEOUT:=300}"
: "${PROXY_CONNECT_TIMEOUT:=15}"
: "${PROXY_KEEPALIVE_TIMEOUT:=30}"
: "${PROXY_MAX_ARGS_SIZE:=1048576}"
: "${PROXY_MAX_TOOL_CALLS:=32}"
: "${PROXY_MAX_CONCURRENT_STREAMS:=50}"

export LITELLM_URL
export PROXY_HOST
export PROXY_PORT
export PROXY_LOG_LEVEL
export PROXY_LOG_FILE
export PROXY_BUFFER_TOOLS
export PROXY_VALIDATE_JSON
export PROXY_STREAM_TIMEOUT
export PROXY_STREAM_MAX_DURATION
export PROXY_MAX_UPSTREAM_CONNECTIONS
export PROXY_REQUEST_TIMEOUT
export PROXY_CONNECT_TIMEOUT
export PROXY_KEEPALIVE_TIMEOUT
export PROXY_MAX_ARGS_SIZE
export PROXY_MAX_TOOL_CALLS
export PROXY_MAX_CONCURRENT_STREAMS

mkdir -p "$(dirname "$PROXY_LOG_FILE")" 2>/dev/null || true

echo "Starting llm-toolstream-proxy"
echo "  LITELLM_URL                    = $LITELLM_URL"
echo "  PROXY_HOST                      = $PROXY_HOST"
echo "  PROXY_PORT                      = $PROXY_PORT"
echo "  PROXY_LOG_LEVEL                 = $PROXY_LOG_LEVEL"
echo "  PROXY_LOG_FILE                  = $PROXY_LOG_FILE"
echo "  PROXY_BUFFER_TOOLS              = $PROXY_BUFFER_TOOLS"
echo "  PROXY_VALIDATE_JSON             = $PROXY_VALIDATE_JSON"
echo "  PROXY_STREAM_TIMEOUT            = $PROXY_STREAM_TIMEOUT"
echo "  PROXY_STREAM_MAX_DURATION       = $PROXY_STREAM_MAX_DURATION"
echo "  PROXY_MAX_UPSTREAM_CONNECTIONS  = $PROXY_MAX_UPSTREAM_CONNECTIONS"
echo "  PROXY_REQUEST_TIMEOUT           = $PROXY_REQUEST_TIMEOUT"
echo "  PROXY_CONNECT_TIMEOUT           = $PROXY_CONNECT_TIMEOUT"
echo "  PROXY_KEEPALIVE_TIMEOUT         = $PROXY_KEEPALIVE_TIMEOUT"
echo "  PROXY_MAX_ARGS_SIZE             = $PROXY_MAX_ARGS_SIZE"
echo "  PROXY_MAX_TOOL_CALLS            = $PROXY_MAX_TOOL_CALLS"
echo "  PROXY_MAX_CONCURRENT_STREAMS    = $PROXY_MAX_CONCURRENT_STREAMS"
echo ""

# IMPORTANT: Do NOT use gunicorn for this proxy.
# Gunicorn's worker timeout kills workers mid-stream, leaving orphaned TCP
# connections with large Send-Q buffers that stall the machine.
# Use direct aiohttp with uvloop instead.
exec python -m llm_toolstream_proxy.main

# ──────────────────────────────────────────────────────────────────────────────
# ALTERNATIVE: gunicorn deployment (NOT RECOMMENDED for streaming proxies)
#
# Gunicorn's --timeout kills workers that haven't responded within N seconds.
# For streaming SSE responses that can take minutes, this means gunicorn will
# SIGKILL workers mid-stream, leaving orphaned TCP connections with unacked
# data in their Send-Q. These accumulate and eventually stall the machine.
#
# If you MUST use gunicorn (e.g., for multi-worker process management), set
# --timeout to a very large value (at least 2x STREAM_MAX_DURATION) and
# accept the risk of zombie connections:
#
# : "${PROXY_WORKERS:=4}"
# : "${PROXY_MAX_REQUESTS:=10000}"
# : "${PROXY_MAX_REQUESTS_JITTER:=1000}"
#
# exec gunicorn \
#     llm_toolstream_proxy.main:create_app() \
#     --bind "${PROXY_HOST}:${PROXY_PORT}" \
#     --workers "$PROXY_WORKERS" \
#     --worker-class aiohttp.worker.GunicornWebWorker \
#     --log-level "$PROXY_LOG_LEVEL" \
#     --timeout 1200 \
#     --graceful-timeout 30 \
#     --max-requests "$PROXY_MAX_REQUESTS" \
#     --max-requests-jitter "$PROXY_MAX_REQUESTS_JITTER"